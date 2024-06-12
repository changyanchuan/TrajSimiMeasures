import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence,pack_sequence
import math


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 1601):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class T3S(nn.Module):
    def __init__(self, ninput, nhidden, cellspace, nlayer_structure_enc = 2):
        super(T3S, self).__init__()
        self.cellspace = cellspace
        
        self.spatial_encoder = nn.LSTM(2, ninput, num_layers = 1, batch_first = False)

        self.cell_encoder = nn.Embedding(self.cellspace.size(), ninput).requires_grad_(False) # follows the paper 
        self.pos_encoder = PositionalEncoding(ninput, 0.1)
        self.structure_encoder = nn.ModuleList( [nn.MultiheadAttention(ninput, num_heads=16) for _ in range(nlayer_structure_enc)] )

        self.gamma_param = nn.Parameter(data = torch.Tensor(1), requires_grad = True)

        
    def forward(self, trajs_coor, trajs_cell, traj_cell_attn_mask, traj_cell_padding_mask, trajs_len):
        # src = [ [[]] ] 3D list, list of traj coors
        # trajs_coor = [seq, batch_size, nfeat], PackedSequence
        # trajs_cell: [seq_len, batch_size, emb_size]
        # traj_cell_attn_mask: [seq_len, seq_len]
        # traj_cell_padding_mask: [batch_size, seq_len]
        # trajs_len: [batch_size]

        # trajs_coor, trajs_cell, traj_cell_attn_mask, \
        #         traj_cell_padding_mask, trajs_len = self.input_processing(src)
        
        
        _, (h_n, _) = self.spatial_encoder(trajs_coor)
        coor_emb = h_n[-1].squeeze(0) # last layer. size = [batch_size, nhidden]

        trajs_cell = self.cell_encoder(trajs_cell)
        trajs_cell = self.pos_encoder(trajs_cell)
        for i, _ in enumerate(self.structure_encoder):
            trajs_cell = self.structure_encoder[i](trajs_cell, trajs_cell, trajs_cell, 
                                                    attn_mask=traj_cell_attn_mask,
                                                    key_padding_mask=traj_cell_padding_mask)[0]
            
        cell_emb = trajs_cell
        mask = 1 - traj_cell_padding_mask.T.unsqueeze(-1).expand(cell_emb.shape).float()
        cell_emb = torch.sum(mask * cell_emb, 0)
        cell_emb = cell_emb / trajs_len.unsqueeze(-1).expand(cell_emb.shape) # [batch_size, traj_emb]
        gamma = torch.sigmoid(self.gamma_param)
        embs = gamma * coor_emb + (1.0 - gamma) * cell_emb
        return embs
    
    @torch.no_grad()
    def interpret(self, inputs):
        device = next(self.parameters()).device
        
        trajs_coor, trajs_cell, traj_cell_attn_mask, \
                traj_cell_padding_mask, trajs_len = inputs
        trajs_coor = trajs_coor.to(device)
        trajs_cell = trajs_cell.to(device)
        traj_cell_padding_mask = traj_cell_padding_mask.to(device)
        trajs_len = trajs_len.to(device)

        embs = self.forward(trajs_coor, trajs_cell, traj_cell_attn_mask, \
                            traj_cell_padding_mask, trajs_len)
        return embs

    @torch.no_grad()
    def trajsimi_interpret(self, inputs, inputs2):
        # inputs, inputs2: both are tuple
        device = next(self.parameters()).device

        trajs_coor, trajs_cell, traj_cell_attn_mask, \
                traj_cell_padding_mask, trajs_len = inputs
        trajs_coor = trajs_coor.to(device)
        trajs_cell = trajs_cell.to(device)
        traj_cell_padding_mask = traj_cell_padding_mask.to(device)
        trajs_len = trajs_len.to(device)
        
        trajs_coor2, trajs_cell2, traj_cell_attn_mask2, \
                traj_cell_padding_mask2, trajs_len2 = inputs2
        trajs_coor2 = trajs_coor2.to(device)
        trajs_cell2 = trajs_cell2.to(device)
        traj_cell_padding_mask2 = traj_cell_padding_mask2.to(device)
        trajs_len2 = trajs_len2.to(device)

        embs = self.forward(trajs_coor, trajs_cell, traj_cell_attn_mask, \
                            traj_cell_padding_mask, trajs_len)
        embs2 = self.forward(trajs_coor2, trajs_cell2, traj_cell_attn_mask2, \
                            traj_cell_padding_mask2, trajs_len2)
        dists = F.pairwise_distance(embs, embs2, p = 1)
        return dists.detach().cpu().tolist()


def input_processing(trajs, cellspace):
    # src = list of trajs in merc; size = [[[lon, lat], [lon, lat], ...] ]
    
    trajs_coor = [torch.tensor(list(map(cellspace.point_norm, traj)), \
                    dtype = torch.float) for traj in trajs]
    trajs_coor = pack_sequence(trajs_coor, enforce_sorted = False)

    trajs_cell = [ torch.tensor([cellspace.get_cellid_by_point(p[0], p[1]) for p in traj], \
                    dtype = torch.long) for traj in trajs]
    trajs_cell = pad_sequence(trajs_cell, batch_first = False) # [seq_len, batch_size, 2]
    
    trajs_len = torch.tensor([len(traj) for traj in trajs], dtype = torch.long)
    
    max_trajs_len = max(trajs_len)
    traj_cell_padding_mask = torch.arange(max_trajs_len)[None, :] >= trajs_len[:, None]

    return trajs_coor, trajs_cell, None, traj_cell_padding_mask, trajs_len


# for trajsimi 
def collate_fn(batch, cellspace):
    src, src2 = zip(*batch)
    inputs = input_processing(src, cellspace)
    inputs2 = input_processing(src2, cellspace)
    return inputs, inputs2 # two tuples


# for knn
def collate_fn_single(src, cellspace):
    inputs = input_processing(src, cellspace)
    return inputs

