# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from importlib import import_module


class ModelLoader:
    def __init__(self, __C):

        self.model_use = __C.MODEL_USE
        model_moudle_path = 'openvqa.models.' + self.model_use + '.net'
        self.model_moudle = import_module(model_moudle_path)

    def Net(self, __arg1, __arg2, __arg3, __arg4):
        return self.model_moudle.Net(__arg1, __arg2, __arg3, __arg4)


class CfgLoader:
    def __init__(self, model_use):

        cfg_moudle_path = 'openvqa.models.' + model_use + '.model_cfgs'
        self.cfg_moudle = import_module(cfg_moudle_path)

    def load(self):
        return self.cfg_moudle.Cfgs()
