"""Exact torch-only replica of the ketos VGSL spec used for training:

  [1,H,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64
   Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do O1c<classes>]

Module names and parameter names match kraken.lib.vgsl.TorchVGSLModel for this spec,
so state dicts are interchangeable. Verified by training/scripts/verify_kernel_arch.py.
This lets the Kaggle kernel run without installing kraken (no internet needed).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        p = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.co = nn.Conv2d(in_channels, out_channels, kernel_size, stride=(1, 1), padding=p)

    def forward(self, x):
        return torch.relu(self.co(x))


class Wrapped(nn.Module):
    """Boxes a submodule under `.layer` to match kraken state_dict keys (no params)."""

    def __init__(self, layer, kind):
        super().__init__()
        self.layer = layer
        self.kind = kind

    def forward(self, x):
        if self.kind == 'drop2d':
            return F.dropout2d(x, self.layer.p, self.training)
        if self.kind == 'drop':
            return F.dropout(x, self.layer.p, self.training)
        return self.layer(x)


class ReshapeS(nn.Module):
    """Replicates kraken.lib.vgsl.layers.Reshape for S1(1x0)1,3."""

    def __init__(self, src_dim, part_a, part_b, high, low):
        super().__init__()
        self.src_dim, self.part_a, self.part_b = src_dim, part_a, part_b
        self.high, self.low = high, low

    def forward(self, x):
        x = x.reshape(x.shape[:self.src_dim] + (self.part_a, self.part_b) + x.shape[self.src_dim + 1:])
        dest = self.low
        src_dim = self.src_dim
        if self.high != src_dim:
            dest = self.high
        else:
            src_dim += 1
        perm = list(range(len(x.shape)))
        step = 1 if dest > src_dim else -1
        for k in range(src_dim, dest, step):
            perm[k], perm[k + step] = perm[k + step], perm[k]
        x = x.permute(perm)
        return x.reshape(x.shape[:dest] + (x.shape[dest] * x.shape[dest + 1],) + x.shape[dest + 2:])


class TransposedSummarizingRNN(nn.Module):
    """Replicates kraken.lib.vgsl.layers.TransposedSummarizingRNN.

    `Lbx` in VGSL parses to transpose=False, summarize=False: a normal
    bidirectional LSTM scanning the width axis (seq = W), batch first.
    """

    def __init__(self, input_size, hidden_size, transpose=False, summarize=False):
        super().__init__()
        self.layer = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.output_size = 2 * hidden_size
        self.transpose = transpose
        self.summarize = summarize

    def forward(self, x):
        x = x.permute(2, 0, 3, 1)
        if self.transpose:
            x = x.transpose(0, 2)
        siz = x.size()
        x = x.contiguous().view(-1, siz[2], siz[3])
        o, _ = self.layer(x)
        o = o.view(siz[0], siz[1], siz[2], self.output_size)
        if self.summarize:
            o = o[:, :, -1, :].unsqueeze(2)
        if self.transpose:
            o = o.transpose(0, 2)
        return o.permute(1, 3, 0, 2)


class LinSoftmax(nn.Module):
    """Replicates kraken.lib.vgsl.layers.LinSoftmax (raw linear output, no softmax)."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.lin = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.transpose(1, 3)
        return self.lin(x).transpose(1, 3)


def init_weights(net):
    """Replicates kraken TorchVGSLModel.init_weights:
    convs uniform(-0.1, 0.1), LSTM weights orthogonal + forget-gate bias 1.0,
    linear xavier_uniform + zero bias."""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.LSTM):
            for p in m.parameters():
                if p.data.dim() == 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.constant_(p.data[len(p) // 4:len(p) // 2], 1.0)
        elif isinstance(m, nn.Conv2d):
            for p in m.parameters():
                nn.init.uniform_(p.data, -0.1, 0.1)


class LineNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.C_0 = ActConv2D(1, 32, (3, 13))
        self.Do_1 = Wrapped(nn.Dropout2d(0.1), 'drop2d')
        self.Mp_2 = Wrapped(nn.MaxPool2d(2, 2), 'pool')
        self.C_3 = ActConv2D(32, 32, (3, 13))
        self.Do_4 = Wrapped(nn.Dropout2d(0.1), 'drop2d')
        self.C_5 = ActConv2D(32, 64, (3, 9))
        self.Do_6 = Wrapped(nn.Dropout2d(0.1), 'drop2d')
        self.Mp_7 = Wrapped(nn.MaxPool2d(2, 2), 'pool')
        self.C_8 = ActConv2D(64, 64, (3, 9))
        self.Do_9 = Wrapped(nn.Dropout2d(0.1), 'drop2d')
        self.S_10 = ReshapeS(2, 1, -1, 2, 1)
        self.L_11 = TransposedSummarizingRNN(1024, 200)
        self.Do_12 = Wrapped(nn.Dropout2d(0.1), 'drop2d')
        self.L_13 = TransposedSummarizingRNN(400, 200)
        self.Do_14 = Wrapped(nn.Dropout2d(0.1), 'drop2d')
        self.L_15 = TransposedSummarizingRNN(400, 200)
        self.Do_16 = Wrapped(nn.Dropout(0.5), 'drop')
        self.O_17 = LinSoftmax(400, classes)
        init_weights(self)

    def forward(self, x):
        for m in self.children():
            x = m(x)
        return x, None
