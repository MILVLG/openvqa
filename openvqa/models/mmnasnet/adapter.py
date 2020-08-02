# --------------------------------------------------------
# OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# --------------------------------------------------------

import torch.nn as nn
import torch
from openvqa.core.base_dataset import BaseAdapter
from openvqa.utils.make_mask import make_mask


class Adapter(BaseAdapter):
    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C

    
    def relation_embedding(self, f_g):
        x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=2)  # [bs, n_obj, 1]

        cx = (x_min + x_max) * 0.5  # [bs, n_obj, 1]
        cy = (y_min + y_max) * 0.5  # [bs, n_obj, 1]
        w = (x_max - x_min) + 1.  # [bs, n_obj, 1]
        h = (y_max - y_min) + 1.  # [bs, n_obj, 1]

        delta_x = cx - cx.transpose(-1, -2)
        delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
        delta_x = torch.log(delta_x)  # [bs, n_obj, n_obj]

        delta_y = cy - cy.transpose(-1, -2)
        delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
        delta_y = torch.log(delta_y)  # [bs, n_obj, n_obj]

        delta_w = torch.log(w / w.transpose(-1, -2))  # [bs, n_obj, n_obj]
        delta_h = torch.log(h / h.transpose(-1, -2))  # [bs, n_obj, n_obj]
        size = delta_h.size()

        delta_x = delta_x.view(size[0], size[1], size[2], 1)
        delta_y = delta_y.view(size[0], size[1], size[2], 1)
        delta_w = delta_w.view(size[0], size[1], size[2], 1)
        delta_h = delta_h.view(size[0], size[1], size[2], 1)  # [bs, n_obj, n_obj, 1]
        position_mat = torch.cat(
            (delta_x, delta_y, delta_w, delta_h), -1)  # [bs, n_obj, n_obj, 4]

        return position_mat

    def vqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)


    def gqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)

        if __C.USE_AUX_FEAT:
            self.grid_linear = nn.Linear(__C.FEAT_SIZE['gqa']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)


    def clevr_init(self, __C):
        self.grid_linear = nn.Linear(__C.FEAT_SIZE['clevr']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)


    def vqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']

        img_feat_mask = make_mask(frcn_feat)

        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_proc(bbox_feat)
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        img_feat = self.frcn_linear(frcn_feat)
        rel_embed = self.relation_embedding(bbox_feat)

        return img_feat, rel_embed, img_feat_mask


    def gqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        grid_feat = feat_dict['GRID_FEAT']

        img_feat_mask = make_mask(frcn_feat)

        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        img_feat = self.frcn_linear(frcn_feat)

        if self.__C.USE_AUX_FEAT:
            grid_feat_mask = make_mask(grid_feat)
            img_feat_mask = torch.cat((img_feat_mask, grid_feat_mask), dim=-1)
            grid_feat = self.grid_linear(grid_feat)
            img_feat = torch.cat((img_feat, grid_feat), dim=1)

        rel_embed = self.relation_embedding(bbox_feat)

        return img_feat, rel_embed, img_feat_mask


    def clevr_forward(self, feat_dict):
        grid_feat = feat_dict['GRID_FEAT']

        img_feat_mask = make_mask(grid_feat)
        img_feat = self.grid_linear(grid_feat)

        rel_embed = self.relation_embedding(bbox_feat)

        return img_feat, rel_embed, img_feat_mask



