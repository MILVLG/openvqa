# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------


def feat_filter(dataset, frcn_feat, grid_feat, bbox_feat):
    feat_dict = {}

    if dataset in ['vqa']:
        feat_dict['FRCN_FEAT'] = frcn_feat
        feat_dict['BBOX_FEAT'] = bbox_feat

    elif dataset in ['gqa']:
        feat_dict['FRCN_FEAT'] = frcn_feat
        feat_dict['GRID_FEAT'] = grid_feat
        feat_dict['BBOX_FEAT'] = bbox_feat

    elif dataset in ['clevr']:
        feat_dict['GRID_FEAT'] = grid_feat

    else:
        exit(-1)

    return feat_dict


