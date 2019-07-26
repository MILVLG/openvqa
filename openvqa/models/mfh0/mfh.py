# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

from openvqa.ops.fc import FC
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


# -------------------------------------------------------------
# ---- Multi-Model Hign-order Bilinear Pooling Co-Attention----
# -------------------------------------------------------------


# MFB Module
class MFB(nn.Module):
    def __init__(self, __C):
        super(MFB, self).__init__()
        self.__C = __C
        self.JOINT_EMB_SIZE = __C.MFB_FACTOR_NUM * __C.MFB_OUT_DIM                            # 5*1000=5000
        self.FC_q = nn.Linear(__C.HIDDEN_SIZE*__C.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)  # 1024*2 -> 5000
        self.FC_i = nn.Linear(__C.IMAGE_CHANNEL, self.JOINT_EMB_SIZE)                         # 2048 -> 5000
        self.dropout = nn.Dropout(p=__C.MFB_DROPOUT_RATIO)
        self.sumpool = nn.AvgPool1d(__C.MFB_FACTOR_NUM, stride=__C.MFB_FACTOR_NUM)

    def forward(self, que_feat, img_feat):
        # que_feat: N x 100 x 2048
        # img_feat: N x 100 x 2048
        batch_size = que_feat.shape[0]     # batch_size = N

        que_feat = self.FC_q(que_feat)                      # N x 100 x 5000
        img_feat = self.FC_i(img_feat)                      # N x 100 x 5000

        z = que_feat * img_feat                             # N x 100 x 5000
        z = self.dropout(z)                                 # N x 100 x 5000
        z = self.sumpool(z) * self.__C.MFB_FACTOR_NUM       # N x 100 x 1000
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))  # N x 1000 x 100
        z = F.normalize(z.view(batch_size, -1))             # N x 100000
        z = z.view(batch_size, self.__C.IMG_FEAT_SIZE, self.__C.MFB_OUT_DIM)  # N x 100 x 1000
        return z


# MFH Module
class MFH(nn.Module):
    def __init__(self, __C):
        super(MFH, self).__init__()
        self.__C = __C
        self.JOINT_EMB_SIZE = __C.MFB_FACTOR_NUM * __C.MFB_OUT_DIM      # 5*1000=5000
        self.FC_q = nn.Linear(__C.HIDDEN_SIZE*__C.NUM_QUESTION_GLIMPSE, self.JOINT_EMB_SIZE)    # 1024*2 -> 5000
        self.FC_i = nn.Linear(__C.IMAGE_CHANNEL*__C.NUM_IMG_GLIMPSE, self.JOINT_EMB_SIZE)       # 2048*2 -> 5000
        self.dropout = nn.Dropout(p=__C.MFB_DROPOUT_RATIO)
        self.sumpool = nn.AvgPool1d(__C.MFB_FACTOR_NUM, stride=__C.MFB_FACTOR_NUM)

    def forward(self, que_feat, img_feat, exp_in):
        # que_feat: N x 2048
        # img_feat: N x 4096
        batch_size = que_feat.shape[0]      # batch_size = N
        que_feat = self.FC_q(que_feat)      # N x 5000
        img_feat = self.FC_i(img_feat)      # N x 5000
        exp_out = que_feat * img_feat       # N x 5000
        exp_out = self.dropout(exp_out)     # N x 5000
        z = exp_out * exp_in                # N x 5000
        z = z.unsqueeze(1)                  # N x 1 x 5000
        z = self.sumpool(z) * self.__C.MFB_FACTOR_NUM       # N x 1 x 1000
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))  # N x 1 x 1000
        z = F.normalize(z.view(batch_size, -1))             # N x 1000
        return z, exp_out


# Question Attention Module
class QuestionAttention(nn.Module):
    def __init__(self, __C):
        self.__C = __C
        super(QuestionAttention, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(__C.WORD_EMBED_SIZE, __C.HIDDEN_SIZE, num_layers=1, bidirectional=False)
        self.dropout_lstm = nn.Dropout(0.1)
        self.fc = FC(__C.HIDDEN_SIZE, 512, 0.1)
        self.linear = nn.Linear(512, __C.NUM_QUESTION_GLIMPSE)

    def forward(self, que_feat):
        # que_feat:  T x N x 300
        batch_size = que_feat.shape[1]     # batch_size = N

        que_feat = self.dropout(que_feat)                       # T x N x 300
        h_state = autograd.Variable(torch.zeros(1, batch_size, self.__C.HIDDEN_SIZE)).cuda()
        c_state = autograd.Variable(torch.zeros(1, batch_size, self.__C.HIDDEN_SIZE)).cuda()
        que_feat, _ = self.lstm(que_feat, (h_state, c_state))   # T x N x 1024
        que_feat = self.dropout_lstm(que_feat)

        que_att_feat = self.fc(que_feat)                        # T x N x 512
        que_att_feat = self.linear(que_att_feat)                # T x N x 2
        que_att_feat = F.softmax(que_att_feat, dim=0)           # T x N x 2
        que_att_feat_list = []
        for i in range(self.__C.NUM_QUESTION_GLIMPSE):
            mask = que_att_feat[:, :, i:i+1]                        # T x N x 1
            mask = mask * que_feat                              # T x N x 1024
            mask = mask.sum(dim=0)                              # N x 1024
            que_att_feat_list.append(mask)
        que_att_feat = torch.cat(que_att_feat_list, dim=1)      # N x 2048

        return que_att_feat


# Image Attention Module
class ImageAttention(nn.Module):
    def __init__(self, __C):
        super(ImageAttention, self).__init__()
        self.__C = __C
        self.dropout = nn.Dropout(0.1)
        self.mfb = MFB(__C)
        self.fc = FC(__C.MFB_OUT_DIM, 512, 0.1)
        self.linear = nn.Linear(512, __C.NUM_IMG_GLIMPSE)

    def forward(self, que_feat, img_feat):
        # que_feat: N x 2048
        # img_feat: N x 100 x 2048

        que_feat = que_feat.unsqueeze(1).repeat(1, img_feat.size()[1], 1)  # N x 100 x 2048
        img_feat = self.dropout(img_feat)                   # N x 100 x 2048
        img_att_feat = self.mfb(que_feat, img_feat)         # N x 100 x 1000
        img_att_feat = self.fc(img_att_feat)                # N x 100 x 512
        img_att_feat = self.linear(img_att_feat)            # N X 100 X 2
        img_att_feat = F.softmax(img_att_feat, dim=1)       # N x 100 x 2
        img_att_feat_list = []
        for i in range(self.__C.NUM_IMG_GLIMPSE):
            mask = img_att_feat[:, :, i:i+1]                # N x 100 x 1
            mask = mask * img_feat                          # N x 100 x 2048
            mask = mask.sum(dim=1)                          # N x 2048
            img_att_feat_list.append(mask)
        img_att_feat = torch.cat(img_att_feat_list, dim=1)  # N x 4096

        return img_att_feat

