import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class MLP(nn.Module):
    def __init__(self, ninput, noutput, nlayer = 2):
        super(MLP, self).__init__()
        
        lst_m = [nn.Linear(ninput, noutput)]
        for _ in range(nlayer - 1):
            lst_m.append(nn.ReLU())
            lst_m.append(nn.Linear(noutput, noutput))
        self.encoder = nn.Sequential(*lst_m)


    def forward(self, src, src_padding_mask, src_len):
        # src: [seq_len, batch_size, emb_size]
        # src_padding_mask: [batch_size, seq_len]
        # src_len: [batch_size]
        rtn = self.encoder(src)
        
        mask = 1 - src_padding_mask.T.unsqueeze(-1).expand(rtn.shape).float()
        rtn = torch.sum(mask * rtn, 0)
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape)
        return rtn
    
    
    @torch.no_grad()
    def interpret(self, inputs1):
        device = next(self.parameters()).device
        trajs1_emb, trajs1_len = inputs1
        
        trajs1_emb = trajs1_emb.to(device)
        trajs1_len = trajs1_len.to(device)
        
        max_trajs1_len = trajs1_len.max().item() # trajs1_len[0]
        src_padding_mask1 = torch.arange(max_trajs1_len, device = device)[None, :] >= trajs1_len[:, None]
        
        traj_embs = self.forward(trajs1_emb, src_padding_mask1, trajs1_len)
        return traj_embs

    
    @torch.no_grad()
    def trajsimi_interpret(self, inputs1, inputs2):
        device = next(self.parameters()).device
        trajs1_emb, trajs1_len = inputs1
        trajs2_emb, trajs2_len = inputs2
        
        trajs1_emb = trajs1_emb.to(device)
        trajs1_len = trajs1_len.to(device)
        
        trajs2_emb = trajs2_emb.to(device)
        trajs2_len = trajs2_len.to(device)
        
        max_trajs1_len = trajs1_len.max().item() # trajs1_len[0]
        src_padding_mask1 = torch.arange(max_trajs1_len, device = device)[None, :] >= trajs1_len[:, None]
        max_trajs2_len = trajs2_len.max().item() # trajs2_len[0]
        src_padding_mask2 = torch.arange(max_trajs2_len, device = device)[None, :] >= trajs2_len[:, None]
        
        traj_embs = self.forward(trajs1_emb, src_padding_mask1, trajs1_len)
        traj_embs2 = self.forward(trajs2_emb, src_padding_mask2, trajs2_len)

        dists = F.pairwise_distance(traj_embs, traj_embs2, p = 1)
        return dists.detach().cpu().tolist()


def input_processing(trajs):
    # src = list of trajs in merc; size = [[[lon, lat], [lon, lat], ...] ]
    
    trajs_t = [torch.tensor(t) for t in trajs]
    trajs_t = pad_sequence(trajs_t, batch_first = False) # [seq_len, batch_size, 2]
    trajs_len = torch.tensor(list(map(len, trajs)), dtype = torch.long)
    
    # return: padded tensor and their length
    return trajs_t, trajs_len

# for trajsimi 
def collate_fn(batch):
    src, src2 = zip(*batch)
    inputs = input_processing(src)
    inputs2 = input_processing(src2)
    return inputs, inputs2 # two tuples

# for knn
def collate_fn_single(src):
    inputs = input_processing(src)
    return inputs
