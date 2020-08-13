# --------------------------------------------------------
# OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# --------------------------------------------------------

from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# ------------------------------
# --- Operations and Modules ---
# ------------------------------

class RelMHAtt(nn.Module):
    def __init__(self, __C):
        super(RelMHAtt, self).__init__()
        self.__C = __C
        self.HBASE = __C.REL_HBASE
        self.HHEAD = int(__C.HIDDEN_SIZE / __C.REL_HBASE)

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_r = nn.Linear(__C.REL_SIZE, self.HHEAD, bias=True)

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, k, q, mask=None, rel_embed=None):
        assert rel_embed is not None
        n_batches = q.size(0)

        v = self.linear_v(v).view(n_batches, -1, self.HHEAD,
                                  self.HBASE).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.HHEAD,
                                  self.HBASE).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.HHEAD,
                                  self.HBASE).transpose(1, 2)
        r = self.relu(self.linear_r(rel_embed)).permute(0, 3, 1, 2)

        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        scores = torch.log(torch.clamp(r, min=1e-6)) + scores
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        atted = torch.matmul(att_map, v)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches, -1, self.__C.HIDDEN_SIZE)
        atted = self.linear_merge(atted)

        return atted


class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.HIDDEN_SIZE * 4,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.norm = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, arg1, arg2, arg3, arg4):
        x = self.norm(x + self.dropout(
            self.mlp(x)
        ))
        return x


class SA(nn.Module):
    def __init__(self, __C, size=1024):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.norm = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, arg1, y_mask, arg2, arg3):
        y = self.norm(y + self.dropout(
            self.mhatt(y, y, y, y_mask)
        ))

        return y


class RSA(nn.Module):
    def __init__(self, __C, size=1024):
        super(RSA, self).__init__()

        self.mhatt = RelMHAtt(__C)

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.norm = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, arg1, x_mask, arg2, rela):
        x = self.norm(x + self.dropout(
            self.mhatt(x, x, x, x_mask, rela)
        ))

        return x


class GA(nn.Module):
    def __init__(self, __C):
        super(GA, self).__init__()

        self.mhatt = MHAtt(__C)

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.norm = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask, rela):
        x = self.norm(x + self.dropout(
            self.mhatt(v=y, k=y, q=x, mask=y_mask)
        ))

        return x


# ------------------------------------------------
# --- Encoder-Decoder Architecture of MMNasNet ---
# ------------------------------------------------

class NAS_ED(nn.Module):
    def __init__(self, __C):
        super(NAS_ED, self).__init__()
        enc = __C.ARCH['enc']
        dec = __C.ARCH['dec']
        self.enc_list = nn.ModuleList([eval(layer)(__C) for layer in enc])
        self.dec_list = nn.ModuleList([eval(layer)(__C) for layer in dec])

    def forward(self, y, x, y_mask, x_mask, rela):
        for enc in self.enc_list:
            y = enc(y, None, y_mask, None, None)

        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask, rela)

        return y, x
