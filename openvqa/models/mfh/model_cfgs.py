# --------------------------------------------------------
# OpenVQA
# Written by Gao Pengbing https://github.com/nbgao
# --------------------------------------------------------

from openvqa.core.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        self.HIGH_ORDER = False
        self.HIDDEN_SIZE = 512
        self.MFB_FACTOR_NUM = 5
        self.MFB_OUT_SIZE = 1000
        self.LSTM_OUT_SIZE = 1024
        self.LSTM_DROPOUT_RATIO = 0.1
        self.MFB_DROPOUT_RATIO = 0.1
        self.NUM_IMG_GLIMPSES = 2
        self.NUM_QUES_GLIMPSES = 2
