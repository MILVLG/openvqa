# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

from openvqa.ops.fc import FC, MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


# -------------------------------------------------------------
# ---- Multi-Model Hign-order Bilinear Pooling Co-Attention----
# -------------------------------------------------------------


class MFB(nn.Module):
    def __init__(self, __C, img_feat_size, ques_feat_size, is_first):
        super(MFB, self).__init__()
        self.__C = __C
        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, __C.MFB_FACTOR_NUM * __C.MFB_OUT_SIZE)
        self.proj_q = nn.Linear(ques_feat_size, __C.MFB_FACTOR_NUM * __C.MFB_OUT_SIZE)
        self.dropout = nn.Dropout(__C.MFB_DROPOUT_RATIO)
        self.pool = nn.AvgPool1d(__C.MFB_FACTOR_NUM, stride=__C.MFB_FACTOR_NUM)

    def forward(self, img_feat, ques_feat, exp_in=1):
        '''
            img_feat.size() -> (N, C, img_feat_size) C can be 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_OUT_SIZE)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]

        img_feat = self.proj_i(img_feat)    # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)  # (N, 1, K*O)

        exp_out = img_feat * ques_feat      # (N, C, K*O)
        # exp_out = self.dropout(exp_out)     # (N, C, K*O)
        # z = exp_out * exp_in                # (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)                   # (N, C, K*O)
        z = self.pool(exp_out) * self.__C.MFB_FACTOR_NUM          # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))             # (N, C*O)
        z = z.view(batch_size, -1, self.__C.MFB_OUT_SIZE)   # (N, C, O)
        return z, exp_out


class QAtt(nn.Module):
    def __init__(self, __C):
        super(QAtt, self).__init__()
        self.__C = __C
        self.mlp = MLP(
            in_size=__C.LSTM_OUT_SIZE,
            mid_size=__C.HIDDEN_SIZE,
            out_size=__C.NUM_QUES_GLIMPSES,
            dropout_r=0.1,
            use_relu=True
        )

    def forward(self, ques_feat):
        '''
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            qatt_feat.size() -> (N, LSTM_OUT_SIZE * NUM_QUES_GLIMPSES)
        '''
        batch_size = ques_feat.shape[1]
        qatt_maps = self.mlp(ques_feat)             # (N, T, NUM_QUES_GLIMPSES)
        qatt_maps = F.softmax(qatt_maps, dim=1)     # (N, T, NUM_QUES_GLIMPSES)

        qatt_feat_list = []
        for i in range(self.__C.NUM_QUES_GLIMPSES):
            mask = qatt_maps[:, :, i:i + 1]         # (N, T, 1)
            mask = mask * ques_feat                 # (N, T, LSTM_OUT_SIZE)
            mask = torch.sum(mask, dim=1)           # (N, LSTM_OUT_SIZE)
            qatt_feat_list.append(mask)
            # qatt_feat_list.append(torch.sum(qatt_maps[:, :, i:i+1] * x, dim=1))
        qatt_feat = torch.cat(qatt_feat_list, dim=1)  # (N, LSTM_OUT_SIZE*NUM_QUES_GLIMPSES)

        return qatt_feat


class IAtt(nn.Module):
    def __init__(self, __C, img_feat_size, ques_att_feat_size):
        super(IAtt, self).__init__()
        self.__C = __C
        self.dropout = nn.Dropout(0.1)
        self.mfb = MFB(__C, img_feat_size, ques_att_feat_size, True)
        self.mlp = MLP(
            in_size=__C.MFB_OUT_SIZE,
            mid_size=__C.HIDDEN_SIZE,
            out_size=__C.NUM_IMG_GLIMPSES,
            dropout_r=0.1,
            use_relu=True
        )

    def forward(self, img_feat, ques_att_feat):
        '''
            img_feats.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_att_feat.size() -> (N, LSTM_OUT_SIZE * NUM_QUES_GLIMPSES)
            iatt_feat.size() -> (N, MFB_OUT_SIZE * NUM_IMG_GLIMPSES)
        '''
        ques_att_feat = ques_att_feat.unsqueeze(1)  # (N, 1, LSTM_OUT_SIZE * NUM_QUES_GLIMPSES)
        img_feat = self.dropout(img_feat)
        z, _ = self.mfb(img_feat, ques_att_feat)  # (N, C, O)

        iatt_maps = self.mlp(z)  # (N, C, NUM_IMG_GLIMPSES)
        iatt_maps = F.softmax(iatt_maps, dim=1)  # (N, C, NUM_IMG_GLIMPSES)

        iatt_feat_list = []
        for i in range(self.__C.NUM_IMG_GLIMPSES):
            mask = iatt_maps[:, :, i:i + 1]  # (N, C, 1)
            mask = mask * img_feat  # (N, C, FRCN_FEAT_SIZE)
            mask = torch.sum(mask, dim=1)  # (N, FRCN_FEAT_SIZE)
            iatt_feat_list.append(mask)
            # iatt_feat_list.append(torch.sum(iatt_maps[:, :, i:i+1] * img_feat, dim=1))
        iatt_feat = torch.cat(iatt_feat_list, dim=1)  # (N, FRCN_FEAT_SIZE*IMG_NUM_GLIMPSES)

        return iatt_feat


class CoAtt(nn.Module):
    def __init__(self, __C):
        super(CoAtt, self).__init__()
        self.__C = __C

        img_feat_size = __C.FEAT_SIZE[__C.DATASET]['FRCN_FEAT_SIZE']  # 2048
        img_att_feat_size = img_feat_size * __C.NUM_IMG_GLIMPSES  # 4096
        ques_att_feat_size = __C.LSTM_OUT_SIZE * __C.NUM_QUES_GLIMPSES  # 2048

        self.q_att = QAtt(__C)
        self.i_att = IAtt(__C, img_feat_size, ques_att_feat_size)

        if self.__C.HIGH_ORDER:  # MFH
            self.mfh1 = MFB(__C, img_att_feat_size, ques_att_feat_size, True)
            self.mfh2 = MFB(__C, img_att_feat_size, ques_att_feat_size, False)
        else:  # MFB
            self.mfb = MFB(__C, img_att_feat_size, ques_att_feat_size, True)

    def forward(self, img_feat, ques_feat):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        ques_feat = self.q_att(ques_feat)  # (N, LSTM_OUT_SIZE*NUM_QUES_GLIMPSES)
        fuse_feat = self.i_att(img_feat, ques_feat)  # (N, FRCN_FEAT_SIZE*NUM_IMG_GLIMPSES)

        if self.__C.HIGH_ORDER:  # MFH
            z1, exp1 = self.mfh1(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))  # z1:(N, 1, O)  exp1:(N, C, K*O)
            z2, _ = self.mfh2(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1), exp1)  # z2:(N, 1, O)  _:(N, C, K*O)
            z = torch.cat((z1.squeeze(1), z2.squeeze(1)), 1)  # (N, 2*O)
        else:  # MFB
            z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))  # z:(N, 1, O)  _:(N, C, K*O)
            z = z.squeeze(1)  # (N, O)

        return z
