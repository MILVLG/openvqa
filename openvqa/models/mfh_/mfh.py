# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


# -------------------------------------------------------------
# ---- Multi-Model Hign-order Bilinear Pooling Co-Attention----
# -------------------------------------------------------------

class FC(nn.Module):
    def __init__(self, in_features, out_features, dropout_ratio):
        super(FC, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # init.xavier_normal_(self.linear.state_dict()['weight'])
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear(x)))
        return x


class ProdAttn(nn.Module):
    def __init__(self, __C, feat_size, hidden_size, dropout_ratio):
        super(ProdAttn, self).__init__()
        self.__C = __C
        self.FC_i = nn.Linear(feat_size, __C.MFB_FACTOR_NUM*__C.MFB_OUT_DIM)
        self.FC_q = nn.Linear(hidden_size, __C.MFB_FACTOR_NUM*__C.MFB_OUT_DIM)
        # init.xavier_normal_(self.FC_i.state_dict()['weight'])
        # init.xavier_normal_(self.FC_q.state_dict()['weight'])
        self.dropout = nn.Dropout(dropout_ratio)
        self.pool = nn.AvgPool1d(__C.MFB_FACTOR_NUM, stride=__C.MFB_FACTOR_NUM)

    def forward(self, outs_i, outs_q):
        '''
            outs_q.size() -> (batch, C, hidden)
            outs_i.size() -> (batch, C, feat)
            z.size() -> (batch, C, hidden)
            output.size() -> (batch, C, hidden)
        '''
        batch_size = outs_i.shape[0]
        # z = self.pool(self.dropout(F.relu(self.FC_i(outs_i)) )
        z = self.pool(self.dropout(self.FC_i(outs_i) * self.FC_q(outs_q))) * self.__C.MFB_FACTOR_NUM
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1)).view(batch_size, -1, self.__C.MFB_OUT_DIM)
        return z


class MFH(nn.Module):
    def __init__(self, __C, feat_size, hidden_size, dropout_ratio, is_first):
        super(MFH, self).__init__()
        self.__C = __C
        self.is_first = is_first
        self.FC_i = nn.Linear(feat_size, __C.MFB_FACTOR_NUM*__C.MFB_OUT_DIM)
        self.FC_q = nn.Linear(hidden_size, __C.MFB_FACTOR_NUM*__C.MFB_OUT_DIM)
        # init.xavier_normal_(self.FC_i.state_dict()['weight'])
        # init.xavier_normal_(self.FC_q.state_dict()['weight'])
        self.dropout = nn.Dropout(dropout_ratio)
        self.pool = nn.AvgPool1d(__C.MFB_FACTOR_NUM, stride=__C.MFB_FACTOR_NUM)

    def forward(self, outs_i, outs_q, exp_last = 0):
        '''
            outs_q.size() -> (batch, C, hidden)
            outs_i.size() -> (batch, C, feat)
            z.size() -> (batch, C, hidden)
            output.size() -> (batch, C, hidden)
        '''
        batch_size = outs_i.shape[0]
        # z = self.pool(self.dropout(F.relu(self.FC_i(outs_i)) * F.relu(self.FC_q(outs_q))))
        exp = self.dropout(self.FC_i(outs_i) * self.FC_q(outs_q)) if self.is_first else self.dropout(self.FC_i(outs_i) * self.FC_q(outs_q) * exp_last)
        z = self.pool(exp) * self.__C.MFB_FACTOR_NUM
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1)).view(batch_size, -1, self.__C.MFB_OUT_DIM)
        return z, exp


class QuestionAttention(nn.Module):
    def __init__(self, __C):
        super(QuestionAttention, self).__init__()
        self.__C = __C
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(input_size=__C.EMBEDDING_SIZE, hidden_size=__C.HIDDEN_SIZE, num_layers=1, bidirectional=False)
        self.dropout2 = nn.Dropout(0.1)
        self.fc = FC(__C.HIDDEN_SIZE, 512, 0.1)
        self.linear = nn.Linear(512, __C.NUM_QUESTION_GLIMPSE)
        # init.xavier_normal_(self.linear.state_dict()['weight'])

    def forward(self, x):
        '''
            x.size() -> (seq, batch, embed)
            h_state.size() -> (1, batch, hidden)
            outs.size() -> (seq, batch, hidden)
            attn.size() -> (seq, batch, Qhead)
            x_attn.size() -> (batch, hidden * Qhead)
            output.size() -> (batch, hidden * Qhead)
        '''
        batch_size = x.shape[1]
        x = self.dropout(x)
        h_state = autograd.Variable(torch.zeros(1, batch_size, self.__C.HIDDEN_SIZE)).cuda()
        c_state = autograd.Variable(torch.zeros(1, batch_size, self.__C.HIDDEN_SIZE)).cuda()
        outs, _ = self.lstm(x, (h_state, c_state))
        outs = self.dropout2(outs)
        attn = self.fc(outs)
        attn = F.softmax(self.linear(attn), dim=0)
        attn_list = []
        for i in range(self.__C.NUM_QUESTION_GLIMPSE):
            attn_list.append(torch.sum(attn[:, :, i:i+1] * outs, dim=0))
        x_attn = torch.cat(attn_list, dim=1)
        return x_attn


class ImageAttention(nn.Module):
    def __init__(self, __C):
        super(ImageAttention, self).__init__()
        self.__C = __C
        self.dropout = nn.Dropout(0.1)
        self.prodattn = ProdAttn(__C, __C.IMAGE_CHANNEL, __C.HIDDEN_SIZE * __C.NUM_QUESTION_GLIMPSE, 0.1)
        self.fc = FC(__C.MFB_OUT_DIM, 512, 0.1)
        self.linear = nn.Linear(512, __C.NUM_IMG_GLIMPSE)
        # init.xavier_normal_(self.linear.state_dict()['weight'])

    def forward(self, x, outs_q):
        '''
            x.size() -> (batch, topk, feat)
            outs_q.size() -> (batch, hidden * Qhead)
            attn.size() -> (batch, topk, Ihead)
            x_attn.size() -> (batch, hidden * Ihead)
            output.size() -> (batch, hidden * Ihead)
        '''
        x = self.dropout(x)
        attn = self.prodattn(x, outs_q.unsqueeze(1).repeat(1, x.size()[1], 1))
        attn = self.fc(attn)
        attn = F.softmax(self.linear(attn), dim=1)
        attn_list = []
        for i in range(self.__C.NUM_IMG_GLIMPSE):
            attn_list.append(torch.sum(attn[:, :, i:i+1] * x, dim=1))
        x_attn = torch.cat(attn_list, dim=1)
        return x_attn

