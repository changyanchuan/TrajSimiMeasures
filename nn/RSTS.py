# code ref: https://github.com/Like-China/TrajectorySim-RSTS-model

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
import torch.nn.functional as F 
import os


class StackingGRUCell(nn.Module):
    """
    Multi-layer CRU Cell
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StackingGRUCell, self).__init__()
        self.num_layers = num_layers
        self.grus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.grus.append(nn.GRUCell(input_size, hidden_size))
        for i in range(1, num_layers):
            self.grus.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, input, h0):
        """
        Input:
        input (batch, input_size): input tensor
        h0 (num_layers, batch, hidden_size): initial hidden state
        ---
        Output:
        output (batch, hidden_size): the final layer output tensor
        hn (num_layers, batch, hidden_size): the hidden state of each layer
        """
        hn = []
        output = input
        for i, gru in enumerate(self.grus):
            hn_i = gru(output, h0[i])
            hn.append(hn_i)
            if i != self.num_layers - 1:
                output = self.dropout(hn_i)
            else:
                output = hn_i
        hn = torch.stack(hn)
        return output, hn


class GlobalAttention(nn.Module):
    """
    $$a = \sigma((W_1 q)H)$$
    $$c = \tanh(W_2 [a H, q])$$
    """
    def __init__(self, hidden_size):
        super(GlobalAttention, self).__init__()
        self.L1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.L2 = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, q, H):
        """
        Input:
        q (batch, hidden_size): query
        H (batch, seq_len, hidden_size): context
        ---
        Output:
        c (batch, hidden_size)
        """
        # (batch, hidden_size) => (batch, hidden_size, 1)
        q1 = self.L1(q).unsqueeze(2)
        # (batch, seq_len)
        a = torch.bmm(H, q1).squeeze(2)
        a = self.softmax(a)
        # (batch, seq_len) => (batch, 1, seq_len)
        a = a.unsqueeze(1)
        # (batch, hidden_size)
        c = torch.bmm(a, H).squeeze(1)
        # (batch, hidden_size * 2)
        c = torch.cat([c, q], 1)
        return self.tanh(self.L2(c))


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, num_layers, dropout,
                       bidirectional, region):

        super(Encoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers
        self.region = region
        
        self.embedding = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)

    def forward(self, input, lengths, h0=None):
        """
        Input:
        input (seq_len, batch): padded sequence tensor
        lengths (1, batch): sequence lengths
        h0 (num_layers*num_directions, batch, hidden_size): initial hidden state
        ---
        Output:
        hn (num_layers*num_directions, batch, hidden_size):
            the hidden state of each layer
        output (seq_len, batch, hidden_size*num_directions): output tensor
        """
        # (seq_len, batch) => (seq_len, batch, input_size)
        embed = self.embedding(input)
        lengths = lengths.data.view(-1).tolist()
        if lengths is not None:
            embed = pack_padded_sequence(embed, lengths, enforce_sorted = False)
        output, hn = self.rnn(embed, h0)
        if lengths is not None:
            output = pad_packed_sequence(output)[0]
        return hn, output
    
    @torch.no_grad()
    def interpret(self, inputs1):
        device = next(self.parameters()).device
        traj_cellid, traj_len = inputs1
        _, embs = self.forward(traj_cellid.to(device), traj_len.to(device))
        embs = embs[traj_len-1, range(len(traj_len)), :]
        return embs

    @torch.no_grad()
    def trajsimi_interpret(self, inputs1, inputs2):
        device = next(self.parameters()).device
        traj_cellid, traj_len = inputs1
        traj_cellid2, traj_len2 = inputs2
        _, embs = self.forward(traj_cellid.to(device), traj_len.to(device))
        _, embs2 = self.forward(traj_cellid2.to(device), traj_len2.to(device))
        embs = embs[traj_len-1, range(len(traj_len)), :]
        embs2 = embs2[traj_len2-1, range(len(traj_len2)), :]
        dists = F.pairwise_distance(embs, embs2, p = 2)
        return dists.detach().cpu().tolist()


def input_processing(src, region):
    # src lon lat -> cell ids
    traj_cellid = []
    for traj in src:
        # xy = [ (p[0], p[1]) for p in traj]
        # t = [ p[2] for p in traj]
        # traj_cellid.append(torch.tensor(region.trip2words(xy, t)))
        traj_cellid.append(torch.tensor(region.trip2words_3d(traj)))
    traj_cellid = pad_sequence(traj_cellid, batch_first = False, padding_value = 0)
    traj_len = torch.tensor([len(traj) for traj in src], dtype = torch.long)
    return traj_cellid, traj_len


# for trajsimi 
def collate_fn(batch, region):
    src, src2 = zip(*batch)
    traj_cellid, traj_len = input_processing(src, region)
    traj_cellid2, traj_len2 = input_processing(src2, region)
    return (traj_cellid, traj_len), (traj_cellid2, traj_len2)

# for knn
def collate_fn_single(src, region):
    traj_cellid, traj_len = input_processing(src, region)
    return (traj_cellid, traj_len)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, embedding):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.rnn = StackingGRUCell(input_size, hidden_size, num_layers,
                                   dropout)
        self.attention = GlobalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, input, h, H, use_attention=True):
        """
        Input:
        input (seq_len, batch): padded sequence tensor
        h (num_layers, batch, hidden_size): input hidden state
        H (seq_len, batch, hidden_size): the context used in attention mechanism
            which is the output of encoder
        use_attention: If True then we use attention
        ---
        Output:
        output (seq_len, batch, hidden_size)
        h (num_layers, batch, hidden_size): output hidden state,
            h may serve as input hidden state for the next iteration,
            especially when we feed the word one by one (i.e., seq_len=1)
            such as in translation
        """
        assert input.dim() == 2, "The input should be of (seq_len, batch)"
        # (seq_len, batch) => (seq_len, batch, input_size)
        embed = self.embedding(input)
        output = []
        # split along the sequence length dimension
        for e in embed.split(1):
            e = e.squeeze(0) # (1, batch, input_size) => (batch, input_size)
            o, h = self.rnn(e, h)
            if use_attention:
                o = self.attention(o, H.transpose(0, 1))
            o = self.dropout(o)
            output.append(o)
        output = torch.stack(output)
        return output, h


class EncoderDecoder(nn.Module):
    def __init__(self, args):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = args.vocab_size
        self.embedding_size = args.embedding_size
        # self.embedding = nn.Embedding(args.vocab_size, args.embedding_size, padding_idx=settings.PAD)
        self.encoder = Encoder(args.embedding_size, args.hidden_size, args.vocab_size,
                               args.num_layers, args.dropout, args.bidirectional) # TODO
        self.decoder = Decoder(args.embedding_size, args.hidden_size, args.num_layers,
                               args.dropout, self.embedding)
        self.num_layers = args.num_layers

    def load_pretrained_embedding(path, self=None):
        if os.path.isfile(path):
            w = torch.load(path)
            self.embedding.weight.data.copy_(w)

    def encoder_hn2decoder_h0(self, h):
        """
        Input:
        h (num_layers * num_directions, batch, hidden_size): encoder output hn
        ---
        Output:
        h (num_layers, batch, hidden_size * num_directions): decoder input h0
        """
        if self.encoder.num_directions == 2:
            num_layers, batch, hidden_size = h.size(0)//2, h.size(1), h.size(2)
            return h.view(num_layers, 2, batch, hidden_size)\
                    .transpose(1, 2).contiguous()\
                    .view(num_layers, batch, hidden_size * 2)
        else:
            return h

    def forward(self, src, lengths, trg):
        """
        Input:
        src (src_seq_len, batch): source tensor
        lengths (1, batch): source sequence lengths
        trg (trg_seq_len, batch): target tensor, the `seq_len` in trg is not
            necessarily the same as that in src
        ---
        Output:
        output (trg_seq_len, batch, hidden_size)
        """
        encoder_hn, H = self.encoder(src, lengths)
        decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)
        output, decoder_hn = self.decoder(trg[:-1], decoder_h0, H)
        return output

