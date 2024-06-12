# based on the code at https://github.com/yaodi833/NeuTraj
# the original github implementation is inefficient and unpythonic, 
# i rewrite some parts -- yc

import sys
sys.path.append('..')

from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math
import numpy as np

from config import Config as Config
from nn.NEUTRAJ_utils import neutraj_trajs_preprocess, neutraj_trajs_process_for_model_input, trajcoor_to_trajpadinput



class NeuTraj_Network(nn.Module):
    def __init__(self, input_size, target_size, grid_size, cell_size, x_range, y_range, \
                        sampling_num = 10, stard_LSTM = False, incell = True):
        super(NeuTraj_Network, self).__init__()
        self.input_size = input_size # 4
        self.target_size = target_size # 128
        self.grid_size = grid_size # [1100, 1100] number of cells in x-axis and y-axis
        self.cell_size = cell_size
        self.x_range = x_range
        self.y_range = y_range
        self.sampling_num = sampling_num
        
        self.rnn = RNNEncoder(self.input_size, self.target_size, self.grid_size, stard_LSTM = stard_LSTM, incell = incell)
    
    def forward(self, inputs_arrays, inputs_len_arrays):
        device = next(self.parameters()).device
        
        anchor_input = torch.tensor(inputs_arrays[0], dtype = torch.float, device = device, requires_grad = False)
        trajs_input = torch.tensor(inputs_arrays[1], dtype = torch.float, device = device, requires_grad = False)
        negative_input = torch.tensor(inputs_arrays[2], dtype = torch.float, device = device, requires_grad = False)
        # anchor_input = inputs_arrays[0]
        # trajs_input = inputs_arrays[1]
        # negative_input = inputs_arrays[2]

        anchor_input_len = inputs_len_arrays[0]
        trajs_input_len = inputs_len_arrays[1]
        negative_input_len = inputs_len_arrays[2]

        num_trajs = len(anchor_input) # batch_size*(1+self.sampling_num)

        # hidden = (autograd.Variable( torch.zeros(self.batch_size*(1+self.sampling_num), self.target_size),requires_grad=False ).to(Config.device),
        #                     autograd.Variable( torch.zeros(self.batch_size*(1+self.sampling_num), self.target_size),requires_grad=False ).to(Config.device) )
        hidden = (torch.zeros((num_trajs, self.target_size),requires_grad = False, device = device),
                            torch.zeros((num_trajs, self.target_size),requires_grad = False, device = device) )

        # anchor_embedding = self.rnn([autograd.Variable(anchor_input,requires_grad=False).to(Config.device), anchor_input_len], hidden)
        # trajs_embedding = self.rnn([autograd.Variable(trajs_input,requires_grad=False).to(Config.device), trajs_input_len], hidden)
        # negative_embedding = self.rnn([autograd.Variable(negative_input,requires_grad=False).to(Config.device), negative_input_len], hidden)

        anchor_embedding = self.rnn([anchor_input, anchor_input_len], hidden)
        trajs_embedding = self.rnn([trajs_input, trajs_input_len], hidden)
        negative_embedding = self.rnn([negative_input, negative_input_len], hidden)

        trajs_loss = torch.exp(-F.pairwise_distance(anchor_embedding, trajs_embedding, p=2))
        negative_loss = torch.exp(-F.pairwise_distance(anchor_embedding, negative_embedding, p=2))
        return trajs_loss, negative_loss

    
    def spatial_memory_update(self, inputs_arrays, inputs_len_arrays):
        device = next(self.parameters()).device
        
        batch_traj_input = torch.tensor(inputs_arrays[3], device = device)
        # batch_traj_input = inputs_arrays[3]
        batch_traj_len = inputs_len_arrays[3]
        # batch_hidden = (autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size),requires_grad=False).to(Config.device),
        #                 autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size),requires_grad=False).to(Config.device))
        batch_hidden = (torch.zeros((len(batch_traj_len), self.target_size),requires_grad=False, device=device),
                        torch.zeros((len(batch_traj_len), self.target_size),requires_grad=False, device=device))
        # self.rnn.batch_grid_state_gates([autograd.Variable(batch_traj_input).to(Config.device), batch_traj_len],batch_hidden)
        self.rnn.batch_grid_state_gates([batch_traj_input, batch_traj_len],batch_hidden)

    @torch.no_grad()
    def interpret(self, inputs1):
        device = next(self.parameters()).device
        src1_t, src1_len = inputs1
        src1_t = src1_t.to(device)
        num_trajs = len(src1_len)
        _h = torch.zeros((num_trajs, self.target_size), requires_grad = False, device = device)
        hidden1 = (_h, _h)
        embs1 = self.rnn((src1_t, src1_len), hidden1)
        return embs1
    
    @torch.no_grad()
    def trajsimi_interpret(self, inputs1, inputs2):
        device = next(self.parameters()).device
        src1_t, src1_len = inputs1
        src2_t, src2_len = inputs2
        src1_t = src1_t.to(device)
        src2_t = src2_t.to(device)
        
        num_trajs = len(src1_len)
        _h = torch.zeros((num_trajs, self.target_size), requires_grad = False, device = device)
        hidden1 = (_h, _h)
        hidden2 = (_h, _h)
        embs1 = self.rnn((src1_t, src1_len), hidden1)
        embs2 = self.rnn((src2_t, src2_len), hidden2)

        dists = F.pairwise_distance(embs1, embs2, p = 1)
        return dists.detach().cpu().tolist()
    
    
# for trajsimi 
def collate_fn(batch, x_range, y_range, cell_size):
    src, src2 = zip(*batch)
    src1_padded, src1_len = trajcoor_to_trajpadinput(src, x_range, y_range, cell_size)
    src1_t = torch.Tensor(src1_padded)
    src2_padded, src2_len = trajcoor_to_trajpadinput(src2, x_range, y_range, cell_size)
    src2_t = torch.Tensor(src2_padded)
    return (src1_t, src1_len), (src2_t, src2_len) # src1_len: list
    
# for knn
def collate_fn_single(src, x_range, y_range, cell_size):
    src1_padded, src1_len = trajcoor_to_trajpadinput(src, x_range, y_range, cell_size)
    src1_t = torch.Tensor(src1_padded)
    return (src1_t, src1_len) # src1_len: list


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, grid_size, stard_LSTM= False, incell = True):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size # 4
        self.hidden_size = hidden_size # 128 target size
        self.stard_LSTM = stard_LSTM
        self.cell = SAM_LSTMCell(input_size, hidden_size, grid_size, incell=incell)

        # print(self.cell)
        # print('in cell update: {}'.format(incell))
        # self.cell = torch.nn.LSTMCell(input_size-2, hidden_size).to(Config.device)

    # RNN-network forward
    def forward(self, inputs_a, initial_state = None):
        inputs, inputs_len = inputs_a
        time_steps = inputs.size(1)
        out = None

        out, state = initial_state

        outputs = []
        for t in range(time_steps):
            cell_input = inputs[:, t, :]
            out, state = self.cell(cell_input, (out, state))
            outputs.append(out)

        mask_out = []
        for b, v in enumerate(inputs_len):
            mask_out.append(outputs[v-1][b,:].view(1,-1))

        return torch.cat(mask_out, dim = 0)

    def batch_grid_state_gates(self, inputs_a, initial_state = None):
        device = next(self.parameters()).device
        
        inputs, inputs_len = inputs_a
        time_steps = inputs.size(1)
        out, state = initial_state
        outputs = []
        gates_out_all = []
        # batch_weight_ih = autograd.Variable(self.cell.weight_ih.data, requires_grad=False).to(Config.device)
        # batch_weight_hh = autograd.Variable(self.cell.weight_hh.data, requires_grad=False).to(Config.device)
        # batch_bias_ih = autograd.Variable(self.cell.bias_ih.data, requires_grad=False).to(Config.device)
        # batch_bias_hh = autograd.Variable(self.cell.bias_hh.data, requires_grad=False).to(Config.device)
        batch_weight_ih = torch.tensor(self.cell.weight_ih.data, requires_grad=False, device=device)
        batch_weight_hh = torch.tensor(self.cell.weight_hh.data, requires_grad=False, device=device)
        batch_bias_ih = torch.tensor(self.cell.bias_ih.data, requires_grad=False, device=device)
        batch_bias_hh = torch.tensor(self.cell.bias_hh.data, requires_grad=False, device=device)
        for t in range(time_steps):
            # cell_input = inputs[:, t, :][:,:-2]
            cell_input = inputs[:, t, :]
            self.cell.update_memory(cell_input, (out, state),
                                    batch_weight_ih, batch_weight_hh,
                                    batch_bias_ih, batch_bias_hh)


class RNNCellBase(nn.Module):
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))


class SAM_LSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, grid_size, bias=True, incell = True):
        super(SAM_LSTMCell, self).__init__()
        self.incell = incell
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.bias = bias
        self.neutraj_sam_spatial_width = 2
        self.weight_ih = Parameter(torch.Tensor(5 * hidden_size, input_size-2))
        self.weight_hh = Parameter(torch.Tensor(5 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(5 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(5 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.spatial_embedding = SpatialExternalMemory(grid_size[0]+3*self.neutraj_sam_spatial_width,
                                                                grid_size[1]+3*self.neutraj_sam_spatial_width,
                                                                hidden_size)
        self.atten = Attention(hidden_size)
        self.c_d = None
        self.sg = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        return self.spatial_lstm_cell(input, hx,
                    self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

    def spatial_lstm_cell(self, input_a, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        # self.spatial_embedding = torch.ones(self.spatial_embedding.size()).to(Config.device)
        device = next(self.parameters()).device
        input = input_a[:,:-2]
        grid_input = input_a[:,-2:].type(torch.LongTensor).to(device) + self.neutraj_sam_spatial_width

        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

        ingate, forgetgate, cellgate, outgate, spatialgate = gates.chunk(5, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        spatialgate = torch.sigmoid(spatialgate)
        cy_h = (forgetgate * cx) + (ingate * cellgate)
        cy_hh  = cy_h.data
        cs = self.spatial_embedding.find_nearby_grids(grid_input)
        atten_cs, attn_weights = self.atten(cy_hh,cs) # __call__ which is equal to forward
        c = cy_h + spatialgate * atten_cs
        # c = cy_h
        hy = outgate * torch.tanh(c)

        if self.incell:
            grid_x, grid_y = grid_input[:, 0].data, grid_input[:, 1].data
            self.sg = spatialgate.data
            self.c_d = c.data
            updates = self.sg* self.spatial_embedding.read(grid_x, grid_y) + (1-self.sg) * self.c_d
            if self.training:
                self.spatial_embedding.update(grid_x, grid_y, updates)

        return hy, c

    def batch_update_memory(self, input_a, hidden, w_ih, w_hh, b_ih=None, b_hh=None, w = 2):
        input = input_a[:,:-2]
        grid_input = input_a[:,-2:].type(torch.LongTensor)+w

        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

        ingate, forgetgate, cellgate, outgate, spatialgate = gates.chunk(5, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        spatialgate = torch.sigmoid(spatialgate)
        cy_h = (forgetgate * cx) + (ingate * cellgate)
        cy_hh  = cy_h.data
        cs = self.spatial_embedding.find_nearby_grids(grid_input)
        atten_cs, attn_weights = self.atten.grid_update_atten(cy_hh,cs)

        c = cy_h + spatialgate * atten_cs
        grid_x, grid_y = grid_input[:, 0].data, grid_input[:, 1].data
        self.sg = spatialgate.data
        self.c_d = c.data
        updates = self.sg* self.spatial_embedding.read(grid_x, grid_y) + (1-self.sg) * self.c_d
        if self.training:
            self.spatial_embedding.update(grid_x, grid_y, updates)


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None
        self.linear_weight = None
        self.linear_bias = None


    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        output = autograd.Variable(output, requires_grad=False)
        context = autograd.Variable(context, requires_grad=False)
        batch_size = output.size(0)
        output = output.view(batch_size,1,-1)
        hidden_size = output.size(2)
        input_size = context.size(1)
        attn = torch.bmm(output, context.transpose(1, 2))
        self.mask = (attn.data == 0).bool() # yc
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        oatt = (attn.data != attn.data).bool() # yc
        attn.data.masked_fill_(oatt,0.)
        mix = torch.bmm(attn, context)


        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        out = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        out = torch.squeeze(out, 1)
        return out, attn

    def grid_update_atten(self, output, context):
        output = autograd.Variable(output, requires_grad=False)
        context = autograd.Variable(context, requires_grad=False)
        batch_size = output.size(0)
        output = output.view(batch_size,1,-1)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        self.mask = (attn.data == 0).bool() # yc
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), -1).view(batch_size, -1, input_size)
        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)
        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_l
        self.linear_weight = autograd.Variable(self.linear_out.weight.data, requires_grad=False)
        self.linear_bias = autograd.Variable(self.linear_out.bias.data, requires_grad=False)

        out = torch.tanh(F.linear(combined.view(-1, 2 * hidden_size), self.linear_weight, self.linear_bias)).view(batch_size, -1, hidden_size)
        out = torch.squeeze(out, 1)
        oatt = (out.data != out.data).bool() # yc
        out.data.masked_fill_(oatt,0.)
        return out, attn


class SpatialExternalMemory(nn.Module):
    def __init__(self, N, M, H):
        super(SpatialExternalMemory, self).__init__()

        self.N = N
        self.M = M
        self.H = H

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        # This is typically used to register a buffer that should not to be considered a model parameter.
        self.register_buffer('memory', autograd.Variable(torch.Tensor(N, M, H)))

        # Initialize memory bias
        nn.init.constant_(self.memory, 0.0)

    def reset(self):
        """Initialize memory from bias, for start-of-sequence."""
        nn.init.constant(self.memory, 0.0)

    def size(self):
        return self.N, self.M, self.H

    def find_nearby_grids(self, grid_input, w=2):
        grid_x, grid_y = grid_input[:,0].data, grid_input[:,1].data
        batch_size = len(grid_x)
        mask = torch.arange(-w, w + 1, 1).unsqueeze(0).to(grid_x.device) 
        mask_x = mask.reshape(-1, 1).repeat(1, (w * 2 + 1) * batch_size).view(-1)
        mask_y = mask.reshape(-1,1).repeat((w * 2 + 1) , batch_size).view(-1)
        grid_x_bd = (grid_x.repeat((w * 2 + 1)**2)).view(-1) + mask_x
        grid_y_bd = (grid_y.repeat((w * 2 + 1)**2)).view(-1) + mask_y
        t = self.memory[grid_x_bd, grid_y_bd, :].view(len(grid_x), (2*w+1)*(2*w+1), -1)
        return t

    def update(self, grid_x, grid_y, updates):
        self.memory[grid_x, grid_y, :] = updates

    def read(self, grid_x, grid_y):
        return self.memory[grid_x, grid_y, :]


class WeightedRankingLoss(nn.Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightedRankingLoss, self).__init__()
        self.positive_loss = WeightMSELoss(batch_size, sampling_num)
        self.negative_loss = WeightMSELoss(batch_size, sampling_num)

    def forward(self, p_input, p_target, n_input, n_target):
        trajs_mse_loss = self.positive_loss(p_input, autograd.Variable(p_target), False)

        negative_mse_loss = self.negative_loss(n_input, autograd.Variable(n_target), True)

        self.trajs_mse_loss = trajs_mse_loss
        self.negative_mse_loss = negative_mse_loss
        loss = sum([trajs_mse_loss,negative_mse_loss])
        return loss


class WeightMSELoss(nn.Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightMSELoss, self).__init__()
        # self.weight = []
        # for i in range(batch_size):
        #     self.weight.append(0.)
        #     for traj_index in range(sampling_num):
        #         self.weight.append(np.array([sampling_num - traj_index]))

        # self.weight = np.array(self.weight)
        # sum = np.sum(self.weight)
        # self.weight = self.weight / sum
        # self.weight = self.weight.astype(np.float32)
        # self.weight = Parameter(torch.Tensor(self.weight).to(Config.device), requires_grad = False)
        weight_lst = [0] + list(range(sampling_num, 0, -1))
        self.register_buffer('weight', torch.tensor(weight_lst, dtype = torch.float, requires_grad = False))

        # self.batch_size = batch_size
        self.sampling_num = sampling_num

    def forward(self, input, target, isReLU = False):
        # div = target - input.view(-1,1)
        # if isReLU:
        #     div = F.relu(div.view(-1,1))
        # square = torch.mul(div.view(-1,1), div.view(-1,1))
        # weight_square = torch.mul(square.view(-1,1), self.weight) # self.weight.view(-1,1))
        div = target.squeeze(-1) - input
        if isReLU:
            div = F.relu(div)
        square = torch.mul(div, div)
        weight = self.weight.repeat(square.shape[0] // self.weight.shape[0])
        weight_square = torch.mul(square, weight / weight.sum())
        loss = torch.sum(weight_square)
        return loss

