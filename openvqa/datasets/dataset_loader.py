# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from importlib import import_module

class DatasetLoader:
    def __init__(self, __C):
        self.__C = __C

        self.dataset = __C.DATASET
        dataset_moudle_path = 'openvqa.datasets.' + self.dataset +'.' + self.dataset + '_loader'
        self.dataset_moudle = import_module(dataset_moudle_path)

    def DataSet(self):
        return self.dataset_moudle.DataSet(self.__C)


class EvalLoader:
    def __init__(self, __C):
        self.__C = __C

        self.dataset = __C.DATASET
        eval_moudle_path = 'openvqa.datasets.' + self.dataset + '.' + 'eval' + '.' + 'result_eval'
        self.eval_moudle = import_module(eval_moudle_path)

    def eval(self, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7):
        return self.eval_moudle.eval(self.__C, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7)
