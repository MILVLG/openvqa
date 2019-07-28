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
        self.proj_i = nn.Linear(img_feat_size, __C.MFB_K*__C.MFB_OUT_SIZE)
        self.proj_q = nn.Linear(ques_feat_size, __C.MFB_K*__C.MFB_OUT_SIZE)
        self.dropout = nn.Dropout(__C.MFB_DROPOUT_R)
        self.pool = nn.AvgPool1d(__C.MFB_K, stride=__C.MFB_K)

    def forward(self, img_feat, ques_feat, exp_last = 1):
        '''
            ques_feat.size() -> (N, 1, ques_feat_size)
            img_feat.size() -> (N, C, img_feat_size) C can be 1 or 100
            z.size() -> (N, C, MFB_OUT_SIZE)

        '''
        batch_size = img_feat.shape[0]

        exp = self.dropout(self.proj_i(img_feat) * self.proj_q(ques_feat)) if self.is_first else self.dropout(self.proj_i(img_feat) * self.proj_q(ques_feat) * exp_last)
        z = self.pool(exp) * self.__C.MFB_K
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1)).view(batch_size, -1, self.__C.MFB_OUT_SIZE)
        return z, exp


class QAtt(nn.Module):
    def __init__(self, __C):
        super(QAtt, self).__init__()
        self.__C = __C
        self.mlp = MLP(
            in_size=__C.LSTM_OUT_SIZE,
            mid_size=__C.HIDDEN_SIZE,
            out_size=__C.NUM_QUES_GLIMPSES,
            dropout_r=0,
            use_relu=True
        )

    def forward(self, x):
        '''
            x.size() -> (N, T, LSTM_OUT_SIZE)
            qatt_feat.size() -> (N, LSTM_OUT_SIZE * NUM_QUES_GLIMPSES)
        '''
        batch_size = x.shape[1]
        
        qatt_maps = F.softmax(self.mlp(x),dim=1)
        #print('qatt_maps:', qatt_maps.shape)
        qatt_feat_list = []
        for i in range(self.__C.NUM_QUES_GLIMPSES):
            qatt_feat_list.append(torch.sum(qatt_maps[:, :, i:i+1] * x, dim=1))
        qatt_feat = torch.cat(qatt_feat_list, dim=1)
        #print('qatt_feat:', qatt_feat.shape)
        
        return qatt_feat


class IAtt(nn.Module):
    def __init__(self, __C, img_feat_size, ques_att_feat_size):
        super(IAtt, self).__init__()
        self.__C = __C
        self.mfb = MFB(__C, img_feat_size, ques_att_feat_size, True)
        self.mlp = MLP(
            in_size=__C.MFB_OUT_SIZE,
            mid_size=__C.HIDDEN_SIZE,
            out_size=__C.NUM_IMG_GLIMPSES,
            dropout_r=0,
            use_relu=True
        )
        #self.dropout = nn.Dropout(0.1)
        #self.prodattn = ProdAttn(__C, __C.IMAGE_CHANNEL, __C.HIDDEN_SIZE * __C.NUM_QUESTION_GLIMPSE, 0.1)
        #self.fc = FC(__C.MFB_OUT_DIM, 512, 0.1)
        #self.linear = nn.Linear(512, __C.NUM_IMG_GLIMPSE)
        # init.xavier_normal_(self.linear.state_dict()['weight'])

    def forward(self, img_feats, ques_att_feat):
        '''
            img_feats.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_att_feat.size() -> (N, LSTM_OUT_SIZE * NUM_QUES_GLIMPSES)
            iatt_feat.size() -> (N, MFB_OUT_SIZE * NUM_IMG_GLIMPSES)
        '''
        ques_att_feat = ques_att_feat.unsqueeze(1)
        z, _ = self.mfb(img_feats, ques_att_feat)
        #print('z:', z.shape)
        iatt_maps = F.softmax(self.mlp(z), dim=1)
        #print('iatt_maps:', iatt_maps.shape)
        iatt_feat_list = []
        for i in range(self.__C.NUM_IMG_GLIMPSES):
            iatt_feat_list.append(torch.sum(iatt_maps[:, :, i:i+1] * img_feats, dim=1))
        iatt_feat = torch.cat(iatt_feat_list, dim=1)
        #print('iatt_feat:', iatt_feat.shape)
        return iatt_feat

class CoAtt(nn.Module):
    def __init__(self, __C):
        super(CoAtt, self).__init__()
        self.__C = __C

        img_feat_size = __C.FEAT_SIZE[__C.DATASET]['FRCN_FEAT_SIZE']
        ques_att_feat_size = __C.NUM_QUES_GLIMPSES*__C.LSTM_OUT_SIZE
        
        self.q_att = QAtt(__C)
        self.i_att = IAtt(__C, img_feat_size, ques_att_feat_size)

        
        if self.__C.HIGH_ORDER:
            self.mfb1 = MFB(__C, img_feat_size*__C.NUM_IMG_GLIMPSES, ques_att_feat_size, True)
            self.mfb2 = MFB(__C, img_feat_size*__C.NUM_IMG_GLIMPSES, ques_att_feat_size, False)
        else:
            self.mfb1 = MFB(__C, img_feat_size*__C.NUM_IMG_GLIMPSES, ques_att_feat_size, True)


    def forward(self, img_feat, ques_feat):

        ques_feat = self.q_att(ques_feat)            # N x 2048
        fuse_feat = self.i_att(img_feat, ques_feat)    # N x 4096
        
        z = None

        if self.__C.HIGH_ORDER:
            z1, exp1 = self.mfb1(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))        # N x 1000  N x 5000
            z2, _ = self.mfb2(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1), exp1)  # N x 1000  N x 5000
            z = torch.cat((z1.squeeze(1), z2.squeeze(1)), 1)                       # N x 2000
        else:
            z, _ = self.mfb1(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))        # N x 1000  N x 5000
        
        return z