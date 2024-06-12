# code ref: TrajCL
# Transformer example: https://github.com/pytorch/examples/blob/master/word_language_model/model.py

import sys
sys.path.append('..')

import math
from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from nn.utils.traj import merc2cell2, generate_spatial_features
from nn.moco import MoCo

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


class TrajCL(nn.Module):
    def __init__(self, cellspace, ninput, nhidden = 2048, nhead = 4, nlayer = 2, \
                        attn_dropout = 0.1, pos_droput = 0.1):
        super(TrajCL, self).__init__()
        self.cellspace = cellspace
        self.embs = nn.Parameter(data = torch.randn(self.cellspace.size(), ninput, dtype = torch.float32), requires_grad = True)
        
        self.ninput = ninput
        self.nhidden = nhidden
        self.nhead = nhead
        self.pos_encoder = PositionalEncoding(ninput, pos_droput)
        trans_encoder_layers = nn.TransformerEncoderLayer(ninput, nhead, nhidden, attn_dropout)
        self.trans_encoder = nn.TransformerEncoder(trans_encoder_layers, nlayer)
        
        self.spatial_attn = SpatialAttentionEncoder(4, 32, 1, 3, attn_dropout, pos_droput)
        self.gamma_param = nn.Parameter(data = torch.tensor(0.5), requires_grad = True)

    def forward(self, src, attn_mask, src_padding_mask, src_len, srcspatial):
        # src: [seq_len, batch_size, emb_size]
        # attn_mask: [seq_len, seq_len]
        # src_padding_mask: [batch_size, seq_len]
        # src_len: [batch_size]
        # srcspatial : [seq_len, batch_size, 4]

        if srcspatial is not None:
            _, attn_spatial = self.spatial_attn(srcspatial, attn_mask, src_padding_mask, src_len)
            attn_spatial = attn_spatial.repeat(self.nhead, 1, 1)
            gamma = torch.sigmoid(self.gamma_param) * 10
            attn_spatial = gamma * attn_spatial
        else:
            attn_spatial = None

        src = self.pos_encoder(src)
        rtn = self.trans_encoder(src, attn_spatial, src_padding_mask)

        # average pooling # another implementation; faster
        mask = 1 - src_padding_mask.T.unsqueeze(-1).expand(rtn.shape).float()
        rtn = torch.sum(mask * rtn, 0)
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape)

        # return traj embeddings
        return rtn


    @torch.no_grad()
    def interpret(self, inputs):
        device = next(self.parameters()).device
        trajs1_emb, trajs1_emb_p, trajs1_len = inputs
        trajs1_emb = trajs1_emb.to(device)
        trajs1_emb_p = trajs1_emb_p.to(device)
        trajs1_len = trajs1_len.to(device)
        max_trajs1_len = trajs1_len.max().item() # trajs1_len[0]
        src_padding_mask1 = torch.arange(max_trajs1_len, device = device)[None, :] >= trajs1_len[:, None]
        traj_embs = self.forward(trajs1_emb, None, src_padding_mask1, trajs1_len, trajs1_emb_p)
        return traj_embs


    @torch.no_grad()
    def trajsimi_interpret(self, inputs1, inputs2):
        device = next(self.parameters()).device
        trajs1_emb, trajs1_emb_p, trajs1_len = inputs1
        trajs2_emb, trajs2_emb_p, trajs2_len = inputs2
        
        trajs1_emb = trajs1_emb.to(device)
        trajs1_emb_p = trajs1_emb_p.to(device)
        trajs1_len = trajs1_len.to(device)
        
        trajs2_emb = trajs2_emb.to(device)
        trajs2_emb_p = trajs2_emb_p.to(device)
        trajs2_len = trajs2_len.to(device)
        
        max_trajs1_len = trajs1_len.max().item() # trajs1_len[0]
        src_padding_mask1 = torch.arange(max_trajs1_len, device = device)[None, :] >= trajs1_len[:, None]
        max_trajs2_len = trajs2_len.max().item() # trajs2_len[0]
        src_padding_mask2 = torch.arange(max_trajs2_len, device = device)[None, :] >= trajs2_len[:, None]
        
        traj_embs = self.forward(trajs1_emb, None, src_padding_mask1, 
                                 trajs1_len, trajs1_emb_p)
        traj_embs2 = self.forward(trajs2_emb, None, src_padding_mask2, 
                                  trajs2_len, trajs2_emb_p)

        dists = F.pairwise_distance(traj_embs, traj_embs2, p = 1)
        return dists.detach().cpu().tolist()


def input_processing(trajs, cellspace, embs):
    # src = list of trajs in merc; size = [[[lon, lat], [lon, lat], ...] ]
    
    trajs2_cell, trajs2_p = zip(*[merc2cell2(t, cellspace) for t in trajs])
    trajs2_emb_p = [torch.tensor(generate_spatial_features(t, cellspace), dtype = torch.float32) for t in trajs2_p]
    trajs2_emb_p = pad_sequence(trajs2_emb_p, batch_first = False)

    trajs2_emb_cell = [embs[list(t)] for t in trajs2_cell]
    trajs2_emb_cell = pad_sequence(trajs2_emb_cell, batch_first = False) # [seq_len, batch_size, emb_dim]

    trajs2_len = torch.tensor(list(map(len, trajs2_cell)), dtype = torch.long)
    
    # return: padded tensor and their length
    # return trajs2_emb, trajs2_len, trajs2_idx
    return trajs2_emb_cell, trajs2_emb_p, trajs2_len


# for trajsimi 
def collate_fn(batch, cellspace, embs):
    src, src2 = zip(*batch)
    inputs = input_processing(src, cellspace, embs)
    inputs2 = input_processing(src2, cellspace, embs)
    return inputs, inputs2 # two tuples


# for knn
def collate_fn_single(src, cellspace, embs):
    # src, _ = zip(*batch)
    inputs = input_processing(src, cellspace, embs)
    return inputs


class SpatialAttentionEncoder(nn.Module):
    def __init__(self, ninput, nhidden, nhead, nlayer, attn_dropout, pos_droput):
        super(SpatialAttentionEncoder, self).__init__()
        self.ninput = ninput
        self.nhidden = nhidden
        self.pos_encoder = PositionalEncoding(ninput, pos_droput)
        trans_encoder_layers = MyTransformerEncoderLayer(ninput, nhead, nhidden, attn_dropout)
        self.trans_encoder = MyTransformerEncoder(trans_encoder_layers, nlayer)
        
    
    def forward(self, src, attn_mask, src_padding_mask, src_len):
        # src: [seq_len, batch_size, emb_size]
        # attn_mask: [seq_len, seq_len]
        # src_padding_mask: [batch_size, seq_len]
        # src_len: [batch_size]

        src = self.pos_encoder(src)
        rtn, attn = self.trans_encoder(src, attn_mask, src_padding_mask)

        # average pooling # another implementation; faster
        mask = 1 - src_padding_mask.T.unsqueeze(-1).expand(rtn.shape).float()
        rtn = torch.sum(mask * rtn, 0)
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape)

        # rtn = [batch_size, traj_emb]
        # attn = [batch_size, seq_len, seq_len]
        return rtn, attn



class MyTransformerEncoder(nn.Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(MyTransformerEncoder, self).__init__()
        self.layers = nn.modules.transformer._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        output = src

        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn


class MyTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(MyTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.modules.activation.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.modules.transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(MyTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # attn = [batch, seq, seq] # masked
        return src, attn




class TrajCLMoCo(nn.Module):

    def __init__(self, encoder_q, encoder_k):
        super(TrajCLMoCo, self).__init__()

        seq_emb_dim = encoder_q.ninput

        self.clmodel = MoCo(encoder_q, encoder_k, 
                            seq_emb_dim,
                            seq_emb_dim // 2, 
                            2048,
                            temperature = 0.05)


    def forward(self, trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len):
        device = next(self.parameters()).device
        # create kwargs inputs for TransformerEncoder
        
        # generate mask: https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
        max_trajs1_len = trajs1_len.max().item() # trajs1_len[0]
        max_trajs2_len = trajs2_len.max().item() # trajs2_len[0]
        src_padding_mask1 = torch.arange(max_trajs1_len, device = device)[None, :] >= trajs1_len[:, None]
        src_padding_mask2 = torch.arange(max_trajs2_len, device = device)[None, :] >= trajs2_len[:, None]
        
        logits, targets = self.clmodel({'src': trajs1_emb, 'attn_mask': None, 'src_padding_mask': src_padding_mask1, 'src_len': trajs1_len, 'srcspatial': trajs1_emb_p},  
                {'src': trajs2_emb, 'attn_mask': None, 'src_padding_mask': src_padding_mask2, 'src_len': trajs2_len, 'srcspatial': trajs2_emb_p})
        return logits, targets


    def loss(self, logits, targets):
        return self.clmodel.loss(logits, targets)
