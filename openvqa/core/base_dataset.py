# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import numpy as np
import glob, json, torch, random
import torch.utils.data as Data

class BaseDataSet(Data.Dataset):
    def __init__(self):
        self.token_to_ix = None
        self.pretrained_emb = None
        self.ans_to_ix = None
        self.ix_to_ans = None

        self.data_size = None
        self.token_size = None
        self.ans_size = None


    def load_ques_ans(self, idx):
        raise NotImplementedError()


    def load_img_feats(self, idx, iid):
        raise NotImplementedError()


    def __getitem__(self, idx):

        ques_ix_iter, ans_iter, iid = self.load_ques_ans(idx)

        frcn_feat_iter, grid_feat_iter, spat_feat_iter = self.load_img_feats(idx, iid)

        return \
            torch.from_numpy(frcn_feat_iter),\
            torch.from_numpy(grid_feat_iter),\
            torch.from_numpy(spat_feat_iter),\
            torch.from_numpy(ques_ix_iter),\
            torch.from_numpy(ans_iter)


    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)

