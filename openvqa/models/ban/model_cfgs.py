# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenwei Shao https://github.com/ParadoxZW
# --------------------------------------------------------

from openvqa.core.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        self.GLIMPSE = 8
        self.HIDDEN_SIZE = 1024
        self.K_TIMES = 3
        self.BA_HIDDEN_SIZE = self.K_TIMES * self.HIDDEN_SIZE
        self.DROPOUT_R = 0.2
        self.CLASSIFER_DROPOUT_R = 0.5
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 2048
