# Benchmark and Model Zoo

## Environment

We use the following environment to run all the experiments in this page.

- Python 3.6
- PyTorch 0.4.1
- CUDA 9.0.176
- CUDNN 7.0.4

## VQA-v2

We provide two groups of results (including the accuracies of *Overall*, *Yes/No*, *Number* and *Other*) for each model on VQA-v2 using different training schemas: 1) Model training on the `train` split and evaluated on the `val` split (Train -> Val); 2) Model training on the `train+val+vg` splits and evaluated on the `test-dev` split (Train+val+vg -> Test-dev). We only provide pre-trained models for the latter schema. 

**Note that for one model, the used base learning rate in the two schemas may be different, you should modify this setting in the config file to reimplement the results.**

#### Train -> Val


Model | Base lr | Overall (%) | Yes/No (%) | Number (%) | Other (%)
:-: | :-: | :-: | :-: | :-: | :-: 
[BAN-4](https://github.com/MILVLG/openvqa/blob/576876f284af27281ae0e22a9f4c63b7f61da4da/configs/vqa/ban_4.yml) |2e-3| 65.86 | 83.53 | 46.36 | 57.56 |
[BAN-8](https://github.com/MILVLG/openvqa/blob/576876f284af27281ae0e22a9f4c63b7f61da4da/configs/vqa/ban_8.yml) |2e-3| 66.00 | 83.61 | 47.04 | 57.62 |
[MCAN-small](https://github.com/MILVLG/openvqa/blob/576876f284af27281ae0e22a9f4c63b7f61da4da/configs/vqa/mcan_small.yml) |1e-4| 67.17 | 84.82 | 49.31 | 58.48 | 
[MCAN-large](https://github.com/MILVLG/openvqa/blob/576876f284af27281ae0e22a9f4c63b7f61da4da/configs/vqa/mcan_large.yml) |7e-5| 67.50 | 85.14 | 49.66 | 58.80 | 

#### Train+val+vg -> Test-dev

Model | Base lr | Overall (%) | Yes/No (%) | Number (%) | Other (%) | Download
:-: | :-: | :-: |:-: |:-: |:-: | :-:
[MCAN-small](https://github.com/MILVLG/openvqa/blob/576876f284af27281ae0e22a9f4c63b7f61da4da/configs/vqa/mcan_small.yml) |1e-4| 70.69 | 87.08 | 53.16 | 60.66 |  [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EWSniKgB8Y9PropErzcAedkBKwJCeBP6b5x5oT_I4LiWtg?e=HZiGuf)
[MCAN-large](https://github.com/MILVLG/openvqa/blob/576876f284af27281ae0e22a9f4c63b7f61da4da/configs/vqa/mcan_large.yml) |5e-5| | | |  | 
## GQA


## CLEVR



