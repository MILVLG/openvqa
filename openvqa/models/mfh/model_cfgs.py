# --------------------------------------------------------
# OpenVQA
# Written by Gao Pengbing https://github.com/nbgao
# --------------------------------------------------------

from openvqa.core.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        self.LOSS = 'KLDiv'
