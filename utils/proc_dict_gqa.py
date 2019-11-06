# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import sys
sys.path.append('../')
from openvqa.utils.ans_punct import prep_ans
from openvqa.core.path_cfgs import PATH
import json, re

path = PATH()


ques_dict_preread = {
    'train': json.load(open(path.RAW_PATH['gqa']['train'], 'r')),
    'val': json.load(open(path.RAW_PATH['gqa']['val'], 'r')),
    'testdev': json.load(open(path.RAW_PATH['gqa']['testdev'], 'r')),
    'test': json.load(open(path.RAW_PATH['gqa']['test'], 'r')),
}

# Loading question word list
stat_ques_dict = {
    **ques_dict_preread['train'],
    **ques_dict_preread['val'],
    **ques_dict_preread['testdev'],
    **ques_dict_preread['test'],
}

stat_ans_dict = {
    **ques_dict_preread['train'],
    **ques_dict_preread['val'],
    **ques_dict_preread['testdev'],
}


def tokenize(stat_ques_dict):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
        'CLS': 2,
    }

    max_token = 0
    for qid in stat_ques_dict:
        ques = stat_ques_dict[qid]['question']
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        if len(words) > max_token:
            max_token = len(words)

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)

    return token_to_ix, max_token


def ans_stat(stat_ans_dict):
    ans_to_ix = {}
    ix_to_ans = {}

    for qid in stat_ans_dict:
        ans = stat_ans_dict[qid]['answer']
        ans = prep_ans(ans)

        if ans not in ans_to_ix:
            ix_to_ans[ans_to_ix.__len__()] = ans
            ans_to_ix[ans] = ans_to_ix.__len__()

    return ans_to_ix, ix_to_ans

token_to_ix, max_token = tokenize(stat_ques_dict)
ans_to_ix, ix_to_ans = ans_stat(stat_ans_dict)
# print(ans_to_ix)
# print(ix_to_ans)
# print(token_to_ix)
# print(token_to_ix.__len__())
# print(max_token)
json.dump([ans_to_ix, ix_to_ans, token_to_ix, max_token], open('../openvqa/datasets/gqa/dicts.json', 'w'))
