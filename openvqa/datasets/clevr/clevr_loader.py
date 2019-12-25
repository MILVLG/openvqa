# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import numpy as np
import glob, json, re, en_vectors_web_lg
from openvqa.core.base_dataset import BaseDataSet
from openvqa.utils.ans_punct import prep_ans


class DataSet(BaseDataSet):
    def __init__(self, __C):
        super(DataSet, self).__init__()
        self.__C = __C

        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        # Loading all image paths
        # grid_feat_path_list = \
        #     glob.glob(__C.FEATS_PATH[__C.DATASET]['train'] + '/*.npz') + \
        #     glob.glob(__C.FEATS_PATH[__C.DATASET]['val'] + '/*.npz') + \
        #     glob.glob(__C.FEATS_PATH[__C.DATASET]['test'] + '/*.npz')

        # Loading question word list
        stat_ques_list = \
            json.load(open(__C.RAW_PATH[__C.DATASET]['train'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['val'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['test'], 'r'))['questions']

        # Loading answer word list
        stat_ans_list = \
            json.load(open(__C.RAW_PATH[__C.DATASET]['train'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['val'], 'r'))['questions']

        # Loading question and answer list
        self.ques_list = []
        grid_feat_path_list = []

        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(__C.RAW_PATH[__C.DATASET][split], 'r'))['questions']
            grid_feat_path_list += glob.glob(__C.FEATS_PATH[__C.DATASET][split] + '/*.npz')

        # Define run data size
        self.data_size = self.ques_list.__len__()

        print(' ========== Dataset size:', self.data_size)


        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}
        self.iid_to_grid_feat_path = self.img_feat_path_load(grid_feat_path_list)

        # Tokenize
        self.token_to_ix, self.pretrained_emb, max_token = self.tokenize(stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)

        self.max_token = -1
        if self.max_token == -1:
            self.max_token = max_token
        print('Max token length:', max_token, 'Trimmed to:', self.max_token)

        # Answers statistic
        self.ans_to_ix, self.ix_to_ans = self.ans_stat(stat_ans_list)
        self.ans_size = self.ans_to_ix.__len__()
        print(' ========== Answer token vocab size:', self.ans_size)
        print('Finished!')
        print('')



    def img_feat_path_load(self, path_list):
        iid_to_path = {}

        for ix, path in enumerate(path_list):
            iid = path.split('/')[-1].split('.')[0]
            iid_to_path[iid] = path

        return iid_to_path


    def tokenize(self, stat_ques_list, use_glove):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        max_token = 0
        for ques in stat_ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques['question'].lower()
            ).replace('-', ' ').replace('/', ' ').split()

            if len(words) > max_token:
                max_token = len(words)

            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb, max_token


    def ans_stat(self, stat_ans_list):
        ans_to_ix = {}
        ix_to_ans = {}

        for ans_stat in stat_ans_list:
            ans = ans_stat['answer']

            if ans not in ans_to_ix:
                ix_to_ans[ans_to_ix.__len__()] = ans
                ans_to_ix[ans] = ans_to_ix.__len__()

        return ans_to_ix, ix_to_ans



    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_ques_ans(self, idx):
        # if self.__C.RUN_MODE in ['train']:
        ques = self.ques_list[idx]
        iid = str(ques['image_index'])

        # Process question
        ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=self.max_token)
        ans_iter = np.zeros(1)

        if self.__C.RUN_MODE in ['train']:
            # process answers
            ans = ques['answer']
            ans_iter = self.proc_ans(ans, self.ans_to_ix)

        return ques_ix_iter, ans_iter, iid


    def load_img_feats(self, idx, iid):
        grid_feat = np.load(self.iid_to_grid_feat_path[iid])
        grid_feat_iter = grid_feat['x']

        return np.zeros(1), grid_feat_iter, np.zeros(1)



    # ------------------------------------
    # ---- Real-Time Processing Utils ----
    # ------------------------------------

    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix


    def proc_ans(self, ans, ans_to_ix):
        ans_ix = np.zeros(1, np.int64)
        ans_ix[0] = ans_to_ix[ans]

        return ans_ix

