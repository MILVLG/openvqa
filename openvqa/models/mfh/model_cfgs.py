# --------------------------------------------------------
# OpenVQA
# Written by Gao Pengbing https://github.com/nbgao
# --------------------------------------------------------

from openvqa.core.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        self.LOSS = 'BCE'
        self.HIGH_ORDER = False
        self.HIDDEN_SIZE = 512
        self.MFB_K = 5
        self.MFB_OUT_SIZE = 1000
        self.LSTM_OUT_SIZE = 1024
        self.LSTM_DROPOUT_R = 0.3
        self.MFB_DROPOUT_R = 0.1
        self.NUM_IMG_GLIMPSES = 2
        self.NUM_QUES_GLIMPSES = 2
