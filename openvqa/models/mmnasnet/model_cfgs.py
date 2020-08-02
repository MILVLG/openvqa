# --------------------------------------------------------
# OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# --------------------------------------------------------

from openvqa.core.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        self.ARCH = {
            'enc': ['SA', 'SA', 'SA', 'SA', 'FFN', 'FFN', 'FFN', 'FFN', 'SA', 'FFN', 'FFN', 'FFN'],
            'dec': ['GA', 'GA', 'FFN', 'FFN', 'GA', 'FFN', 'RSA', 'GA', 'FFN', 'GA', 'RSA', 'FFN', 'RSA', 'SA', 'FFN', 'RSA', 'GA', 'FFN']
        }
        self.HIDDEN_SIZE = 512
        self.BBOXFEAT_EMB_SIZE = 2048
        self.FF_SIZE = 2048
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024
        self.USE_AUX_FEAT = False
        self.USE_BBOX_FEAT = False
        self.REL_HBASE = 64
        self.REL_SIZE = 64
