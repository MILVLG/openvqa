# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

from openvqa.ops.fc import FC
from openvqa.models.mfh.mfh import CoAtt
from openvqa.models.mfh.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
import torch


# -------------------------
# ---- Main MFB/MFH model with Co-Attention Learning ----
# -------------------------


class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C
        self.adapter = Adapter(__C)

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.drop1 = nn.Dropout(__C.MFB_DROPOUT_R)
        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.LSTM_OUT_SIZE,
            num_layers=1,
            batch_first=True
        )
        self.drop2 = nn.Dropout(__C.MFB_DROPOUT_R)
        self.backbone = CoAtt(__C)

        if __C.HIGH_ORDER:
            self.proj = nn.Linear(2*__C.MFB_OUT_SIZE, answer_size)
        else:
            self.proj = nn.Linear(__C.MFB_OUT_SIZE, answer_size)

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):

        img_feat, _ = self.adapter(frcn_feat, grid_feat, bbox_feat)  # N x 100 x 2048

        # Pre-process Language Feature
        ques_feat = self.embedding(ques_ix)
        
        #ques_feat = self.drop1(ques_feat)
        ques_feat, _ = self.lstm(ques_feat)
        ques_feat = self.drop2(ques_feat)

        z = self.backbone(img_feat, ques_feat)
        proj_feat = self.proj(z)

        return proj_feat

