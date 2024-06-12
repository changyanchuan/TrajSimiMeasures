# code ref: https://github.com/PeilunYang/TMN
# I removed some low-relavance codes and reorganized the code.
# I guess that the original TMN implementation is based on Neutraj, whiel NeuTraj's
# implementation has room for improvement, and hence TMN is inefficient as expected.
# -- yc

import sys
sys.path.append('..')

import torch.nn.functional as F
from torch.nn import Module
import torch
import numpy as np
from nn.NEUTRAJ_utils import trajcoor_to_trajpadinput

class TMN(Module):
    def __init__(self, input_size, target_size, grid_size, sampling_num, 
                    lon_range, lat_range, cell_size, sam_spatial_width = 2):
        
        super(TMN, self).__init__()
        self.input_size = input_size
        self.target_size = target_size
        self.grid_size = grid_size
        # self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.cell_size = cell_size

        # self.hidden = (autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
        #                                      requires_grad=False),
        #                    autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
        #                                      requires_grad=False))
        self.smn = SMNEncoder(self.input_size, self.target_size, self.grid_size, sam_spatial_width)


    def forward(self, inputs_arrays, inputs_len_arrays):
        device = next(self.parameters()).device
        anchor_input = torch.tensor(inputs_arrays[0], dtype = torch.float, device = device, requires_grad = False)
        trajs_input = torch.tensor(inputs_arrays[1], dtype = torch.float, device = device, requires_grad = False)
        negative_input = torch.tensor(inputs_arrays[2], dtype = torch.float, device = device, requires_grad = False)

        anchor_input_len = inputs_len_arrays[0]
        trajs_input_len = inputs_len_arrays[1]
        negative_input_len = inputs_len_arrays[2]

        num_trajs = len(anchor_input) # batch_size*(1+self.sampling_num)

        anchor_embedding, trajs_embedding = self.matching_forward(anchor_input, anchor_input_len, trajs_input, trajs_input_len)
        trajs_loss = torch.exp(F.pairwise_distance(anchor_embedding, trajs_embedding, p=2))

        anchor_embedding, negative_embedding = self.matching_forward(anchor_input, anchor_input_len, negative_input, negative_input_len)
        negative_loss = torch.exp(F.pairwise_distance(anchor_embedding, negative_embedding, p=2))
        return trajs_loss, negative_loss


    def matching_forward(self, src1_t, src1_len, src2_t, src2_len):
        emb1, emb2 = self.smn.f(src1_t, src1_len, src2_t, src2_len)
        return emb1, emb2


    @torch.no_grad()
    def trajsimi_interpret(self, inputs1, inputs2):
        device = next(self.parameters()).device
        src1_t, src1_len = inputs1
        src2_t, src2_len = inputs2
        src1_t = src1_t.to(device)
        src2_t = src2_t.to(device)
        embs1, embs2 = self.matching_forward(src1_t, src1_len, src2_t, src2_len)
        
        # copied from TMN Github Code/test_methods.py::test_matching_model()
        embs1 = embs1.cpu().numpy()
        embs2 = embs2.cpu().numpy()
        return [(j, float(np.exp(-np.sum(np.square(embs1[j] - embs2[j])))))
                             for j, e in enumerate(embs1)]
    
# for trajsimi 
def collate_fn(batch, lon_range, lat_range, cell_size):
    src, src2 = zip(*batch)
    src1_padded, src1_len = trajcoor_to_trajpadinput(src, lon_range, lat_range, cell_size)
    src1_t = torch.Tensor(src1_padded)
    src2_padded, src2_len = trajcoor_to_trajpadinput(src2, lon_range, lat_range, cell_size)
    src2_t = torch.Tensor(src2_padded)
    return (src1_t, src1_len), (src2_t, src2_len)


# for trajsimi_DQ, not for kNN, since TMN does not support KNN
def collate_fn_single(src, lon_range, lat_range, cell_size):
    src1_padded, src1_len = trajcoor_to_trajpadinput(src, lon_range, lat_range, cell_size)
    src1_t = torch.Tensor(src1_padded)
    return src1_t, src1_len


class SMNEncoder(Module):
    def __init__(self, input_size, hidden_size, grid_size, pooling_size, sam_spatial_width = 2):
        super(SMNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.sam_spatial_width = sam_spatial_width
        self.mlp_ele = torch.nn.Linear(2, hidden_size//2)
        self.nonLeaky = torch.nn.LeakyReLU(0.1)
        self.nonTanh = torch.nn.Tanh()
        self.point_pooling = torch.nn.AvgPool1d(pooling_size)

        self.seq_model_layer = 1
        self.seq_model = torch.nn.LSTM(hidden_size, hidden_size, num_layers=self.seq_model_layer)
        self.res_linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.res_linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.res_linear3 = torch.nn.Linear(hidden_size, hidden_size)


    def f(self, src1, src1_len, src2, src2_len):
        input_a, input_len_a = src1, src1_len,  # porto inputs:220x149x4 inputs_len:list
        input_b, input_len_b = src2, src2_len
        time_steps_a = input_a.size(1)
        time_steps_b = input_b.size(1)
        
        mlp_input_a = self.nonLeaky(self.mlp_ele(input_a[:, :, :-2]))
        mlp_input_b = self.nonLeaky(self.mlp_ele(input_b[:, :, :-2]))
        sam_offset = self.grid_size[1] * self.sam_spatial_width + self.sam_spatial_width
        input_grid_a = (input_a[:, :, 3] * self.grid_size[1] + input_a[:, :, 2] - sam_offset + 1).clamp(0, self.grid_size[0] * self.grid_size[1]).long()  # porto:220x149
        mask_a = (input_grid_a != 0).unsqueeze(-2)  # porto:220x1x149
        input_grid_b = (input_b[:, :, 3] * self.grid_size[1] + input_b[:, :, 2] - sam_offset + 1).clamp(0, self.grid_size[0] * self.grid_size[1]).long()  # porto:220x149
        mask_b = (input_grid_b != 0).unsqueeze(-2)  # porto:220x1x149

        out_a, state_a = self.init_hidden(self.hidden_size, input_a.size(0))
        out_a = out_a.to(input_a.device)
        state_a = state_a.to(input_a.device)
        
        scores_a_o = torch.matmul(mlp_input_a, mlp_input_b.transpose(-2, -1))  # porto:220x149x149
        scores_a_o = scores_a_o.masked_fill(mask_b == 0, -1e9).transpose(-2, -1)
        scores_a_o = scores_a_o.masked_fill(mask_a == 0, -1e9).transpose(-2, -1)
        scores_a = scores_a_o  # porto:220x149x149
        p_attn_a = F.softmax(scores_a, dim=-1)
        p_attn_a = p_attn_a.masked_fill(mask_b == 0, 0.0).transpose(-2, -1)
        p_attn_a = p_attn_a.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
        attn_ab = p_attn_a.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size // 2)
        sum_traj_b = mlp_input_b.unsqueeze(-3).expand(-1, time_steps_a, -1, -1).mul(attn_ab).sum(dim=-2)
        cell_input_a = torch.cat((mlp_input_a, (mlp_input_a-sum_traj_b)), dim=-1)  # porto:220x149x128

        outputs_a, (hn_a, cn_a) = self.seq_model(cell_input_a.permute(1, 0, 2), (out_a, state_a))
        outputs_ca = torch.sigmoid(self.res_linear1(outputs_a)) * self.nonLeaky(self.res_linear2(outputs_a)) #F.tanh(self.res_linear2(outputs_a))
        outputs_hata = torch.sigmoid(self.res_linear3(outputs_a)) * self.nonLeaky(outputs_ca) #F.tanh(outputs_ca)
        outputs_fa = outputs_a + outputs_hata
        mask_out_a = []
        for b, v in enumerate(input_len_a):
            mask_out_a.append(outputs_fa[v - 1][b, :].view(1, -1))
        fa_outputs = torch.cat(mask_out_a, dim=0)

        out_b, state_b = self.init_hidden(self.hidden_size, input_b.size(0))
        out_b = out_a.to(input_a.device)
        state_b = state_b.to(input_a.device)

       
        scores_b = scores_a_o.permute(0, 2, 1)
        ## scores_b = scores_b.masked_fill(mask_b == 0, -1e9)
        p_attn_b = F.softmax(scores_b, dim=-1)
        p_attn_b = p_attn_b.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
        p_attn_b = p_attn_b.masked_fill(mask_b == 0, 0.0).transpose(-2, -1)
        attn_ba = p_attn_b.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size // 2)
        sum_traj_a = mlp_input_a.unsqueeze(-3).expand(-1, time_steps_b, -1, -1).mul(attn_ba).sum(dim=-2)
        cell_input_b = torch.cat((mlp_input_b, (mlp_input_b - sum_traj_a)), dim=-1)  # porto:220x149x128
        
        outputs_b, (hn_b, cn_b) = self.seq_model(cell_input_b.permute(1, 0, 2), (out_b, state_b))
        outputs_cb = torch.sigmoid(self.res_linear1(outputs_b)) * self.nonLeaky(self.res_linear2(outputs_b)) #F.tanh(self.res_linear2(outputs_b))
        outputs_hatb = torch.sigmoid(self.res_linear3(outputs_b)) * self.nonLeaky(outputs_cb) #F.tanh(outputs_cb)
        outputs_fb = outputs_b + outputs_hatb
        mask_out_b = []
        for b, v in enumerate(input_len_b):
            mask_out_b.append(outputs_b[v - 1][b, :].view(1, -1))
        fb_outputs = torch.cat(mask_out_b, dim=0)

        return fa_outputs, fb_outputs


    def init_hidden(self, hidden_dim, batch_size=1):
        # (num_layers, mini_batch_size, hidden_dim)
        return (torch.zeros(self.seq_model_layer, batch_size, hidden_dim),
                torch.zeros(self.seq_model_layer, batch_size, hidden_dim))


