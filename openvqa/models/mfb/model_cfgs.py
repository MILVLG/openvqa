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
        self.MFB_K = 5
        self.MFB_O = 1000
        self.LSTM_OUT_SIZE = 1024
        self.DROPOUT_R = 0.1
        self.I_GLIMPSES = 2
        self.Q_GLIMPSES = 2
