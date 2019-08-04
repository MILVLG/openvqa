# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch.nn as nn
import torch


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., act=None):
        super(FC, self).__init__()
        fc = []

        fc.append(nn.Linear(in_size, out_size))

        if act is not None:
            fc.append(act())

        if dropout_r > 0:
            fc.append(nn.Dropout(dropout_r))
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., act=None):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, act=act)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))
