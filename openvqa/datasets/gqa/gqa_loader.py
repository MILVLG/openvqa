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

        ques_dict_preread = {
            'train': json.load(open(__C.RAW_PATH[__C.DATASET]['train'], 'r')),
            'val': json.load(open(__C.RAW_PATH[__C.DATASET]['val'], 'r')),
            'testdev': json.load(open(__C.RAW_PATH[__C.DATASET]['testdev'], 'r')),
            'test': json.load(open(__C.RAW_PATH[__C.DATASET]['test'], 'r')),
        }

        # Loading all image paths
        frcn_feat_path_list = glob.glob(__C.FEATS_PATH[__C.DATASET]['default-frcn'] + '/*.npz')
        grid_feat_path_list = glob.glob(__C.FEATS_PATH[__C.DATASET]['default-grid'] + '/*.npz')

        # Loading question word list
        # stat_ques_dict = {
        #     **ques_dict_preread['train'],
        #     **ques_dict_preread['val'],
        #     **ques_dict_preread['testdev'],
        #     **ques_dict_preread['test'],
        # }

        # Loading answer word list
        # stat_ans_dict = {
        #     **ques_dict_preread['train'],
        #     **ques_dict_preread['val'],
        #     **ques_dict_preread['testdev'],
        # }

        # Loading question and answer list
        self.ques_dict = {}
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            if split in ques_dict_preread:
                self.ques_dict = {
                    **self.ques_dict,
                    **ques_dict_preread[split],
                }
            else:
                self.ques_dict = {
                    **self.ques_dict,
                    **json.load(open(__C.RAW_PATH[__C.DATASET][split], 'r')),
                }

        # Define run data size
        self.data_size = self.ques_dict.__len__()
        print(' ========== Dataset size:', self.data_size)


        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}
        self.iid_to_frcn_feat_path = self.img_feat_path_load(frcn_feat_path_list)
        self.iid_to_grid_feat_path = self.img_feat_path_load(grid_feat_path_list)

        # Loading dict: question dict -> question list
        self.qid_list = list(self.ques_dict.keys())

        # Tokenize
        self.token_to_ix, self.pretrained_emb, max_token = self.tokenize('openvqa/datasets/gqa/dicts.json', __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)

        self.max_token = -1
        if self.max_token == -1:
            self.max_token = max_token
        print('Max token length:', max_token, 'Trimmed to:', self.max_token)

        # Answers statistic
        self.ans_to_ix, self.ix_to_ans = self.ans_stat('openvqa/datasets/gqa/dicts.json')
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


    # def tokenize(self, stat_ques_dict, use_glove):
    #     token_to_ix = {
    #         'PAD': 0,
    #         'UNK': 1,
    #         'CLS': 2,
    #     }
    #
    #     spacy_tool = None
    #     pretrained_emb = []
    #     if use_glove:
    #         spacy_tool = en_vectors_web_lg.load()
    #         pretrained_emb.append(spacy_tool('PAD').vector)
    #         pretrained_emb.append(spacy_tool('UNK').vector)
    #         pretrained_emb.append(spacy_tool('CLS').vector)
    #
    #     max_token = 0
    #     for qid in stat_ques_dict:
    #         ques = stat_ques_dict[qid]['question']
    #         words = re.sub(
    #             r"([.,'!?\"()*#:;])",
    #             '',
    #             ques.lower()
    #         ).replace('-', ' ').replace('/', ' ').split()
    #
    #         if len(words) > max_token:
    #             max_token = len(words)
    #
    #         for word in words:
    #             if word not in token_to_ix:
    #                 token_to_ix[word] = len(token_to_ix)
    #                 if use_glove:
    #                     pretrained_emb.append(spacy_tool(word).vector)
    #
    #     pretrained_emb = np.array(pretrained_emb)
    #
    #     return token_to_ix, pretrained_emb, max_token
    #
    #
    # def ans_stat(self, stat_ans_dict):
    #     ans_to_ix = {}
    #     ix_to_ans = {}
    #
    #     for qid in stat_ans_dict:
    #         ans = stat_ans_dict[qid]['answer']
    #         ans = prep_ans(ans)
    #
    #         if ans not in ans_to_ix:
    #             ix_to_ans[ans_to_ix.__len__()] = ans
    #             ans_to_ix[ans] = ans_to_ix.__len__()
    #
    #     return ans_to_ix, ix_to_ans


    def tokenize(self, json_file, use_glove):
        token_to_ix, max_token = json.load(open(json_file, 'r'))[2:]
        spacy_tool = None
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()

        pretrained_emb = []
        for word in token_to_ix:
            if use_glove:
                pretrained_emb.append(spacy_tool(word).vector)
        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb, max_token


    def ans_stat(self, json_file):
        ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))[:2]

        return ans_to_ix, ix_to_ans


    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_ques_ans(self, idx):

        qid = self.qid_list[idx]
        iid = self.ques_dict[qid]['imageId']

        ques = self.ques_dict[qid]['question']
        ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=self.max_token)
        ans_iter = np.zeros(1)

        if self.__C.RUN_MODE in ['train']:
            # process answers
            ans = self.ques_dict[qid]['answer']
            ans_iter = self.proc_ans(ans, self.ans_to_ix)

        return ques_ix_iter, ans_iter, iid


    def load_img_feats(self, idx, iid):
        frcn_feat = np.load(self.iid_to_frcn_feat_path[iid])
        frcn_feat_iter = self.proc_img_feat(frcn_feat['x'], img_feat_pad_size=self.__C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][0])

        grid_feat = np.load(self.iid_to_grid_feat_path[iid])
        grid_feat_iter = grid_feat['x']

        bbox_feat_iter = self.proc_img_feat(
            self.proc_bbox_feat(
                frcn_feat['bbox'],
                (frcn_feat['height'], frcn_feat['width'])
            ),
            img_feat_pad_size=self.__C.FEAT_SIZE['gqa']['BBOX_FEAT_SIZE'][0]
        )

        return frcn_feat_iter, grid_feat_iter, bbox_feat_iter



    # ------------------------------------
    # ---- Real-Time Processing Utils ----
    # ------------------------------------

    def proc_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return img_feat


    def proc_bbox_feat(self, bbox, img_shape):
        bbox_feat = np.zeros((bbox.shape[0], 4), dtype=np.float32)

        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])

        return bbox_feat


    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques.lower()
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
        ans = prep_ans(ans)
        ans_ix[0] = ans_to_ix[ans]

        return ans_ix
