# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os, copy
from openvqa.datasets.dataset_loader import DatasetLoader
from utils.train_engine import train_engine
from utils.test_engine import test_engine

class Execution:
    def __init__(self, __C):
        self.__C = __C

        print('Loading dataset........')
        self.dataset = DatasetLoader(__C).DataSet()

        # If trigger the evaluation after every epoch
        # Will create a new cfgs with RUN_MODE = 'val'
        self.dataset_eval = None
        if __C.EVAL_EVERY_EPOCH:
            __C_eval = copy.deepcopy(__C)
            setattr(__C_eval, 'RUN_MODE', 'val')

            print('Loading validation set for per-epoch evaluation........')
            self.dataset_eval = DatasetLoader(__C_eval).DataSet()


    def run(self, run_mode):
        if run_mode == 'train':
            if self.__C.RESUME is False:
                self.empty_log(self.__C.VERSION)
            train_engine(self.__C, self.dataset, self.dataset_eval)

        elif run_mode == 'val':
            test_engine(self.__C, self.dataset, validation=True)

        elif run_mode == 'test':
            test_engine(self.__C, self.dataset)

        else:
            exit(-1)


    def empty_log(self, version):
        print('Initializing log file........')
        if (os.path.exists(self.__C.LOG_PATH + '/log_run_' + version + '.txt')):
            os.remove(self.__C.LOG_PATH + '/log_run_' + version + '.txt')
        print('Finished!')
        print('')
