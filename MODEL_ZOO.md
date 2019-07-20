# Benchmark and Model Zoo

## Environment

We use the following environment to run all the experiments in this page.

- Python 3.6
- PyTorch 0.4.1
- CUDA 9.0.176
- CUDNN 7.0.4

## VQA-v2

We provide two groups of results (including the accuracies of *Overall*, *Yes/No*, *Number* and *Other*) for each model on VQA-v2 using different training schemas: 1) Model training on the `train` split and evaluated on the `val` split (Train -> Val); 2) Model training on the `train+val+vg` splits and evaluated on the `test-dev` split (Train+val+vg -> Test-dev). We only provide pre-trained models for the latter schema. 

**Note that for one model, the used base learning rate in the two schemas may be different, you should modify this setting in the config file to reimplement the results.**.

#### Train -> Val


Model | Base lr | Overall (%) | Yes/No (%) | Number (%) | Other (%)
:-: | :-: | :-: | :-: | :-: | :-: 
[MCAN-small](https://github.com/MILVLG/openvqa/blob/576876f284af27281ae0e22a9f4c63b7f61da4da/configs/vqa/mcan_small.yml) |1e-4| 67.17 | 84.82 | 49.31 | 58.48 | 

#### Train+val+vg -> Test-dev

Model | Base lr | Overall (%) | Yes/No (%) | Number (%) | Other (%) | Download
:-: | :-: | :-: |:-: |:-: |:-: | :-:
[MCAN-small](https://github.com/MILVLG/openvqa/blob/576876f284af27281ae0e22a9f4c63b7f61da4da/configs/vqa/mcan_small.yml) |1e-4| 70.7  | 86.91 | 53.42 | 60.75 | -

## GQA


## CLEVR



