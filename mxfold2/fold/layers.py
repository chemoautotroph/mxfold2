import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .embedding import OneHotEmbedding, SparseEmbedding
from .transformer import TransformerLayer

class CNNLayer(nn.Module):
    def __init__(self, n_in, num_filters=(128,), filter_size=(7,), pool_size=(1,), dilation=1, dropout_rate=0.0, resnet=False):
        super(CNNLayer, self).__init__()
        self.resnet = resnet
        self.net = nn.ModuleList()
        for n_out, ksize, p in zip(num_filters, filter_size, pool_size):
            self.net.append( 
                nn.Sequential( 
                    nn.Conv1d(n_in, n_out, kernel_size=ksize, dilation=2**dilation, padding=2**dilation*(ksize//2)),
                    nn.MaxPool1d(p, stride=1, padding=p//2) if p > 1 else nn.Identity(),
                    nn.GroupNorm(1, n_out), # same as LayerNorm?
                    nn.CELU(), 
                    nn.Dropout(p=dropout_rate) ) )
            n_in = n_out


    def forward(self, x, lengths=None): # x: (B, n_in, N); lengths: (B,) long or None
        # When `lengths` is provided, pad positions are zeroed after every conv so
        # the next conv's same-padding sees zero neighbors at the valid/pad
        # boundary — matching the per-sequence (batch=1) computation.
        mask_3d = None
        if lengths is not None:
            B_, _, N_ = x.shape
            pad_mask = torch.arange(N_, device=x.device)[None, :] >= lengths.to(x.device)[:, None]  # (B, N)
            mask_3d = pad_mask.unsqueeze(1)  # (B, 1, N) for broadcasting over channels
            x = x.masked_fill(mask_3d, 0.0)
        for net in self.net:
            x_a = net(x)
            x = x + x_a if self.resnet and x.shape[1]==x_a.shape[1] else x_a
            if mask_3d is not None:
                x = x.masked_fill(mask_3d, 0.0)
        return x


class CNNLSTMEncoder(nn.Module):
    def __init__(self, n_in, 
            num_filters=(256,), filter_size=(7,), pool_size=(1,), dilation=0,
            num_lstm_layers=0, num_lstm_units=0, num_att=0, dropout_rate=0.0, resnet=True):

        super(CNNLSTMEncoder, self).__init__()
        self.resnet = resnet
        self.n_in = self.n_out = n_in
        while len(num_filters) > len(filter_size):
            filter_size = tuple(filter_size) + (filter_size[-1],)
        while len(num_filters) > len(pool_size):
            pool_size = tuple(pool_size) + (pool_size[-1],)
        if num_lstm_layers == 0 and num_lstm_units > 0:
            num_lstm_layers = 1

        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv = self.lstm = self.att = None

        if len(num_filters) > 0 and num_filters[0] > 0:
            self.conv = CNNLayer(n_in, num_filters, filter_size, pool_size, dilation, dropout_rate=dropout_rate, resnet=self.resnet)
            self.n_out = n_in = num_filters[-1]

        if num_lstm_layers > 0:
            self.lstm = nn.LSTM(n_in, num_lstm_units, num_layers=num_lstm_layers, batch_first=True, bidirectional=True, 
                            dropout=dropout_rate if num_lstm_layers>1 else 0)
            self.n_out = n_in = num_lstm_units*2
            self.lstm_ln = nn.LayerNorm(self.n_out)

        if num_att > 0:
            self.att = nn.MultiheadAttention(self.n_out, num_att, dropout=dropout_rate)


    def forward(self, x, lengths=None): # x: (B, n_in, N); lengths: (B,) long or None
        # When `lengths` is provided, batch-pad-aware masking is applied so that
        # batched output matches per-sequence (batch=1) output at valid positions.
        # Specifically:
        #   - LSTM uses pack_padded_sequence so the backward direction does not
        #     start from a tail of pad-zero inputs and pollute every position.
        #   - MultiheadAttention uses key_padding_mask so pad keys are ignored.
        #   - Final output has pad rows zeroed so downstream layers (Transform2D,
        #     PairedLayer/UnpairedLayer with same-padded conv) see clean inputs.
        # When `lengths` is None, behavior is identical to upstream (batch=1 only).
        if self.conv is not None:
            x = self.conv(x, lengths=lengths) # (B, C, N)
        x = torch.transpose(x, 1, 2) # (B, N, C)

        pad_mask = None  # (B, N) bool, True at pad positions
        if lengths is not None:
            B_, N_, _ = x.shape
            if int(lengths.max()) > N_:
                raise RuntimeError(f"max length {int(lengths.max())} > padded N {N_}")
            arange = torch.arange(N_, device=x.device)
            pad_mask = arange[None, :] >= lengths.to(x.device)[:, None]

        if self.lstm is not None:
            # Skip pack/pad when no actual padding exists (lengths all equal to N).
            # pack_padded_sequence reorders internally and can produce slightly
            # different numerical results vs the direct path; for same-length
            # batches we want to be bit-identical to batch=1.
            if lengths is not None and not torch.all(lengths == x.shape[1]):
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False)
                x_a_packed, _ = self.lstm(packed)
                x_a, _ = nn.utils.rnn.pad_packed_sequence(
                    x_a_packed, batch_first=True, total_length=x.shape[1])
            else:
                x_a, _ = self.lstm(x)
            x_a = self.lstm_ln(x_a)
            x_a = self.dropout(F.celu(x_a)) # (B, N, H*2)
            x = x + x_a if self.resnet and x.shape[2]==x_a.shape[2] else x_a

        if self.att is not None:
            x_nbc = torch.transpose(x, 0, 1)  # (N, B, C) — MHA default layout
            if pad_mask is not None:
                x_a, _ = self.att(x_nbc, x_nbc, x_nbc, key_padding_mask=pad_mask)
                # Pad-row queries attend to *some* unmasked keys but their own
                # query vector and the resulting outputs may still be NaN/garbage
                # depending on torch version; zero them explicitly. Mask shape
                # (B, N) → (N, B, 1) for broadcasting against x_a.
                x_a = x_a.masked_fill(pad_mask.t().unsqueeze(-1), 0.0)
            else:
                x_a, _ = self.att(x_nbc, x_nbc, x_nbc)
            x = x_nbc + x_a
            x = torch.transpose(x, 0, 1)  # back to (B, N, C)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return x


class Transform2D(nn.Module):
    def __init__(self, join='cat', context_length=0):
        super(Transform2D, self).__init__()
        self.join = join


    def forward(self, x_l, x_r):
        assert(x_l.shape == x_r.shape)
        B, N, C = x_l.shape
        x_l = x_l.reshape(B, N, 1, C).expand(B, N, N, C)
        x_r = x_r.reshape(B, 1, N, C).expand(B, N, N, C)
        if self.join=='cat':
            x = torch.cat((x_l, x_r), dim=3) # (B, N, N, C*2)
        elif self.join=='add':
            x = x_l + x_r # (B, N, N, C)
        elif self.join=='mul':
            x = x_l * x_r # (B, N, N, C)

        return x


class PairedLayer(nn.Module):
    def __init__(self, n_in, n_out=1, filters=(), ksize=(), fc_layers=(), dropout_rate=0.0, exclude_diag=True, resnet=True):
        super(PairedLayer, self).__init__()

        self.resnet = resnet        
        self.exclude_diag = exclude_diag
        while len(filters) > len(ksize):
            ksize = tuple(ksize) + (ksize[-1],)

        self.conv = nn.ModuleList()
        for m, k in zip(filters, ksize):
            self.conv.append(
                nn.Sequential( 
                    nn.Conv2d(n_in, m, k, padding=k//2), 
                    nn.GroupNorm(1, m),
                    nn.CELU(), 
                    nn.Dropout(p=dropout_rate) ) )
            n_in = m

        fc = []
        for m in fc_layers:
            fc += [
                nn.Linear(n_in, m), 
                nn.LayerNorm(m),
                nn.CELU(), 
                nn.Dropout(p=dropout_rate) ]
            n_in = m
        fc += [ nn.Linear(n_in, n_out) ]
        self.fc = nn.Sequential(*fc)


    def forward(self, x, lengths=None):
        # When `lengths` is provided, pad positions of the 2D feature map are
        # zeroed after every Conv2d so subsequent same-padded convs see zero
        # neighbors at the valid/pad boundary — matching the per-sequence
        # (batch=1) computation. The mask must be (B*2, 1, N, N) to broadcast
        # over the doubled-batch (triu/tril concat).
        diag = 1 if self.exclude_diag else 0
        B, N, _, C = x.shape
        # Build 2D pad mask: True wherever i is pad OR j is pad.
        mask_2d_dup = None
        if lengths is not None:
            pad_mask = torch.arange(N, device=x.device)[None, :] >= lengths.to(x.device)[:, None]  # (B, N)
            mask_2d = pad_mask.unsqueeze(2) | pad_mask.unsqueeze(1)  # (B, N, N)
            mask_2d_dup = mask_2d.repeat(2, 1, 1).unsqueeze(1)  # (B*2, 1, N, N)
        x = x.permute(0, 3, 1, 2)
        x_u = torch.triu(x.reshape(B*C, N, N), diagonal=diag).reshape(B, C, N, N)
        x_l = torch.tril(x.reshape(B*C, N, N), diagonal=-1).reshape(B, C, N, N)
        x = torch.cat((x_u, x_l), dim=0).reshape(B*2, C, N, N)
        if mask_2d_dup is not None:
            x = x.masked_fill(mask_2d_dup, 0.0)
        for conv in self.conv:
            x_a = conv(x)
            x = x + x_a if self.resnet and x.shape[1]==x_a.shape[1] else x_a # (B*2, n_out, N, N)
            if mask_2d_dup is not None:
                x = x.masked_fill(mask_2d_dup, 0.0)
        x_u, x_l = torch.split(x, B, dim=0) # (B, n_out, N, N) * 2
        x_u = torch.triu(x_u.reshape(B, -1, N, N), diagonal=diag)
        x_l = torch.tril(x_u.reshape(B, -1, N, N), diagonal=-1)
        x = x_u + x_l # (B, n_out, N, N)
        x = x.permute(0, 2, 3, 1).reshape(B*N*N, -1)
        x = self.fc(x)
        out = x.reshape(B, N, N, -1) # (B, N, N, n_out)
        if lengths is not None:
            # Reuse the (B, N, N) pad mask for the final output too.
            out = out.masked_fill(mask_2d.unsqueeze(-1), 0.0)
        return out


class UnpairedLayer(nn.Module):
    def __init__(self, n_in, n_out=1, filters=(), ksize=(), fc_layers=(), dropout_rate=0.0, resnet=True):
        super(UnpairedLayer, self).__init__()

        self.resnet = resnet
        while len(filters) > len(ksize):
            ksize = tuple(ksize) + (ksize[-1],)

        self.conv = nn.ModuleList()
        for m, k in zip(filters, ksize):
            self.conv.append(
                nn.Sequential(
                    nn.Conv1d(n_in, m, k, padding=k//2), 
                    nn.GroupNorm(1, m),
                    nn.CELU(), 
                    nn.Dropout(p=dropout_rate) ) )
            n_in = m

        fc = []
        for m in fc_layers:
            fc += [
                nn.Linear(n_in, m), 
                nn.LayerNorm(m),
                nn.CELU(), 
                nn.Dropout(p=dropout_rate)]
            n_in = m
        fc += [ nn.Linear(n_in, n_out) ] # , nn.LayerNorm(n_out) ]
        self.fc = nn.Sequential(*fc)


    def forward(self, x, x_base=None, lengths=None):
        # 1D analogue of PairedLayer's masking — zero pad positions after every
        # Conv1d so subsequent same-padded convs see zero neighbors at the
        # valid/pad boundary.
        B, N, C = x.shape
        mask_3d = None
        if lengths is not None:
            pad_mask = torch.arange(N, device=x.device)[None, :] >= lengths.to(x.device)[:, None]  # (B, N)
            mask_3d = pad_mask.unsqueeze(1)  # (B, 1, N)
        x = x.transpose(1, 2) # (B, n_in, N)
        if mask_3d is not None:
            x = x.masked_fill(mask_3d, 0.0)
        for conv in self.conv:
            x_a = conv(x)
            x = x + x_a if self.resnet and x.shape[1]==x_a.shape[1] else x_a
            if mask_3d is not None:
                x = x.masked_fill(mask_3d, 0.0)
        x = x.transpose(1, 2).reshape(B*N, -1) # (B, N, n_out)
        x = self.fc(x)
        out = x.reshape(B, N, -1)
        if lengths is not None:
            out = out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return out


class LengthLayer(nn.Module):
    def __init__(self, n_in, layers=(), dropout_rate=0.5):
        super(LengthLayer, self).__init__()
        self.n_in = n_in
        n = n_in if isinstance(n_in, int) else np.prod(n_in)

        l = []
        for m in layers:
            l += [ nn.Linear(n, m), nn.CELU(), nn.Dropout(p=dropout_rate) ]
            n = m
        l += [ nn.Linear(n, 1) ]
        self.net = nn.Sequential(*l)

        if isinstance(self.n_in, int):
            self.x = torch.tril(torch.ones((self.n_in, self.n_in)))
        else:
            n = np.prod(self.n_in)
            x = np.fromfunction(lambda i, j, k, l: np.logical_and(k<=i ,l<=j), (*self.n_in, *self.n_in))
            self.x = torch.from_numpy(x.astype(np.float32)).reshape(n, n)


    def forward(self, x): 
        return self.net(x)


    def make_param(self):
        device = next(self.net.parameters()).device
        x = self.forward(self.x.to(device))
        return x.reshape((self.n_in,) if isinstance(self.n_in, int) else self.n_in)


class NeuralNet(nn.Module):
    def __init__(self, embed_size=0,
            num_filters=(96,), filter_size=(5,), dilation=0, pool_size=(1,), 
            num_lstm_layers=0, num_lstm_units=0, num_att=0, 
            num_transformer_layers=0, num_transformer_hidden_units=2048,
            num_transformer_att=8,
            no_split_lr=False, pair_join='cat',
            num_paired_filters=(), paired_filter_size=(),
            num_hidden_units=(32,), dropout_rate=0.0, fc_dropout_rate=0.0, 
            exclude_diag=True, n_out_paired_layers=0, n_out_unpaired_layers=0, **kwargs):

        super(NeuralNet, self).__init__()

        self.no_split_lr = no_split_lr
        self.pair_join = pair_join
        self.embedding = OneHotEmbedding() if embed_size == 0 else SparseEmbedding(embed_size)
        n_in = self.embedding.n_out

        if num_transformer_layers==0:
            self.encoder = CNNLSTMEncoder(n_in,
                num_filters=num_filters, filter_size=filter_size, pool_size=pool_size, dilation=dilation, num_att=num_att,
                num_lstm_layers=num_lstm_layers, num_lstm_units=num_lstm_units, dropout_rate=dropout_rate)
        else:
            self.encoder = TransformerLayer(n_in, n_head=num_transformer_att, 
                            n_hidden=num_transformer_hidden_units, 
                            n_layers=num_transformer_layers, dropout=dropout_rate)
        n_in = self.encoder.n_out

        if self.pair_join != 'bilinear':
            self.transform2d = Transform2D(join=pair_join)

            n_in_paired = n_in // 2 if pair_join!='cat' else n_in
            if self.no_split_lr:
                n_in_paired *= 2

            self.fc_paired = PairedLayer(n_in_paired, n_out_paired_layers,
                                    filters=num_paired_filters, ksize=paired_filter_size,
                                    exclude_diag=exclude_diag,
                                    fc_layers=num_hidden_units, dropout_rate=fc_dropout_rate)
            if n_out_unpaired_layers > 0:
                self.fc_unpaired = UnpairedLayer(n_in, n_out_unpaired_layers,
                                        filters=num_paired_filters, ksize=paired_filter_size,
                                        fc_layers=num_hidden_units, dropout_rate=fc_dropout_rate)
            else:
                self.fc_unpaired = None

        else:
            n_in_paired = n_in // 2 if not self.no_split_lr else n_in
            self.bilinear = nn.Bilinear(n_in_paired, n_in_paired, n_out_paired_layers)
            self.linear = nn.Linear(n_in, n_out_unpaired_layers)


    def forward(self, seq):
        device = next(self.parameters()).device
        # Each input gets a leading '0' token (zero-vec marker), then the embedding
        # right-pads to max length. Lengths after the '0' prefix tell the encoder
        # where each sequence ends so LSTM / attention can mask the pad tail.
        prefixed = ['0' + s for s in seq]
        lengths = torch.tensor([len(s) for s in prefixed], dtype=torch.long, device=device)
        x = self.embedding(prefixed).to(device) # (B, 4, N_padded)
        # OneHotEmbedding adds 'n' on both sides of length ksize//2, accounting for that:
        if isinstance(self.embedding, OneHotEmbedding):
            lengths = lengths + 2 * (self.embedding.ksize // 2)
        x = self.encoder(x, lengths=lengths)

        if self.no_split_lr:
            x_l, x_r = x, x
        else:
            x_l = x[:, :, 0::2]
            x_r = x[:, :, 1::2]
        x_r = x_r[:, :, torch.arange(x_r.shape[-1]-1, -1, -1)] # reverse the last axis

        if self.pair_join != 'bilinear':
            x_lr = self.transform2d(x_l, x_r)

            score_paired = self.fc_paired(x_lr, lengths=lengths)
            if self.fc_unpaired is not None:
                score_unpaired = self.fc_unpaired(x, lengths=lengths)
            else:
                score_unpaired = None

            return score_paired, score_unpaired

        else:
            B, N, C = x_l.shape
            x_l = x_l.reshape(B, N, 1, C).expand(B, N, N, C).reshape(B*N*N, -1)
            x_r = x_r.reshape(B, 1, N, C).expand(B, N, N, C).reshape(B*N*N, -1)
            score_paired = self.bilinear(x_l, x_r).reshape(B, N, N, -1)
            score_unpaired = self.linear(x)

            return score_paired, score_unpaired