# --------------------------------------------------------
# OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# Based on the implementation of paper "Bilinear Attention Neworks", NeurIPS 2018 https://github.com/jnhwkim/ban-vqa)
# --------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import torch, math

# ------------------------------
# ----- Weight Normal MLP ------
# ------------------------------

class MLP(nn.Module):
    """
    Simple class for non-linear fully connect network
    """

    def __init__(self, dims, act='ReLU', dropout_r=0.0):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if dropout_r > 0:
                layers.append(nn.Dropout(dropout_r))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if act != '':
                layers.append(getattr(nn, act)())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# ------------------------------
# ------ Bilinear Connect ------
# ------------------------------

class BC(nn.Module):
    """
    Simple class for non-linear bilinear connect network
    """

    def __init__(self, __C, atten=False):
        super(BC, self).__init__()

        self.__C = __C
        self.v_net = MLP([__C.IMG_FEAT_SIZE,
                          __C.BA_HIDDEN_SIZE], dropout_r=__C.DROPOUT_R)
        self.q_net = MLP([__C.HIDDEN_SIZE,
                          __C.BA_HIDDEN_SIZE], dropout_r=__C.DROPOUT_R)
        if not atten:
            self.p_net = nn.AvgPool1d(__C.K_TIMES, stride=__C.K_TIMES)
        else:            
            self.dropout = nn.Dropout(__C.CLASSIFER_DROPOUT_R)  # attention

            self.h_mat = nn.Parameter(torch.Tensor(
                1, __C.GLIMPSE, 1, __C.BA_HIDDEN_SIZE).normal_())
            self.h_bias = nn.Parameter(
                torch.Tensor(1, __C.GLIMPSE, 1, 1).normal_())

    def forward(self, v, q):
        # low-rank bilinear pooling using einsum
        v_ = self.dropout(self.v_net(v))
        q_ = self.q_net(q)
        logits = torch.einsum('xhyk,bvk,bqk->bhvq',
                              (self.h_mat, v_, q_)) + self.h_bias
        return logits  # b x h_out x v x q
    
    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)  # b x v x d
        q_ = self.q_net(q)  # b x q x d
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        logits = logits.unsqueeze(1)  # b x 1 x d
        logits = self.p_net(logits).squeeze(1) * self.__C.K_TIMES  # sum-pooling
        return logits

# ------------------------------
# -------- BiAttention ---------
# ------------------------------


class BiAttention(nn.Module):
    def __init__(self, __C):
        super(BiAttention, self).__init__()

        self.__C = __C
        self.logits = weight_norm(BC(__C, True), name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True, logit=False, mask_with=-float('inf')):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(
                1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, mask_with)

        if not logit:
            p = nn.functional.softmax(
                logits.view(-1, self.__C.GLIMPSE, v_num * q_num), 2)
            return p.view(-1, self.__C.GLIMPSE, v_num, q_num), logits

        return logits

# ------------------------------
# - Bilinear Attention Network -
# ------------------------------

class BAN(nn.Module):
    def __init__(self, __C):
        super(BAN, self).__init__()

        self.__C = __C
        self.BiAtt = BiAttention(__C)
        b_net = []
        q_prj = []
        c_prj = []
        for i in range(__C.GLIMPSE):
            b_net.append(BC(__C))
            q_prj.append(MLP([__C.HIDDEN_SIZE, __C.HIDDEN_SIZE], '', __C.DROPOUT_R))
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)

    def forward(self, q, v):
        att, logits = self.BiAtt(v, q)  # b x g x v x q

        for g in range(self.__C.GLIMPSE):
            bi_emb = self.b_net[g].forward_with_weights(
                v, q, att[:, g, :, :])  # b x l x h
            q = self.q_prj[g](bi_emb.unsqueeze(1)) + q

        return q
