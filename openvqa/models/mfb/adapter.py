# --------------------------------------------------------
# OpenVQA
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

import torch.nn as nn
import torch
import torch.nn.functional as F
from openvqa.core.base_dataset import BaseAdapter
from openvqa.utils.make_mask import make_mask


class Adapter(BaseAdapter):
    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C


    def vqa_init(self, __C):
        self.frcn_linear = nn.Linear(__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][1], __C.HIDDEN_SIZE)


    def gqa_init(self, __C):
        self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
        self.frcn_linear = nn.Linear(
            __C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][1] + __C.BBOXFEAT_EMB_SIZE,
            __C.HIDDEN_SIZE
        )
        self.grid_linear = nn.Linear(__C.FEAT_SIZE['gqa']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)


    def clevr_init(self, __C):
        self.grid_linear = nn.Linear(__C.FEAT_SIZE['clevr']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)


    def vqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']

        img_feat_mask = make_mask(frcn_feat)
        img_feat = frcn_feat
        #[N, C, W] = img_feat.shape
        #img_feat = F.normalize(img_feat.view(N, -1)).view(N, C, W)
        return img_feat, img_feat_mask

    def gqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        grid_feat = feat_dict['GRID_FEAT']

        img_feat_mask = torch.cat((make_mask(frcn_feat), make_mask(grid_feat)), dim=-1)
        bbox_feat = self.bbox_linear(bbox_feat)
        frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        frcn_feat = self.frcn_linear(frcn_feat)
        grid_feat = self.grid_linear(grid_feat)
        img_feat = torch.cat((frcn_feat, grid_feat), dim=1)

        return img_feat, img_feat_mask


    def clevr_forward(self, feat_dict):
        grid_feat = feat_dict['GRID_FEAT']

        img_feat_mask = make_mask(grid_feat)
        img_feat = self.grid_linear(grid_feat)

        return img_feat, img_feat_mask



