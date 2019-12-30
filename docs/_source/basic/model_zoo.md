# Benchmark and Model Zoo

## Environment

We use the following environment to run all the experiments in this page.

- Python 3.6
- PyTorch 0.4.1
- CUDA 9.0.176
- CUDNN 7.0.4

## VQA-v2

We provide three groups of results (including the accuracies of *Overall*, *Yes/No*, *Number* and *Other*) for each model on VQA-v2 using different training schemes as follows. We provide pre-trained models for the latter two schemes. 

- **Train -> Val**: trained on the `train` split and evaluated on the `val` split. 
- **Train+val -> Test-dev**: trained on the `train+val` splits and evaluated on the `test-dev` split. 

- **Train+val+vg -> Test-dev**: trained on the `train+val+vg` splits and evaluated on the `test-dev` split.  

**Note that for one model, the used base learning rate in the two schemes may be different, you should modify this setting in the config file to reproduce the results.**



#### Train -> Val

| Model                                                                                  | Base lr | Overall (%) | Yes/No (%) | Number (%) | Other (%) |
|:--------------------------------------------------------------------------------------:|:-------:|:-----------:|:----------:|:----------:|:---------:|
| [BUTD](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/butd.yml)             | 2e-3    | 63.84       | 81.40      | 43.81      | 55.78     |
| [MFB](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/mfb.yml)               | 7e-4    | 65.35       | 83.23      | 45.31      | 57.05     |
| [MFH](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/mfh.yml)               | 7e-4    | 66.18       | 84.07      | 46.55      | 57.78     |
| [BAN-4](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/ban_4.yml)           | 2e-3    | 65.86       | 83.53      | 46.36      | 57.56     |
| [BAN-8](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/ban_8.yml)           | 2e-3    | 66.00       | 83.61      | 47.04      | 57.62     |
| [MCAN-small](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/mcan_small.yml) | 1e-4    | 67.17       | 84.82      | 49.31      | 58.48     |
| [MCAN-large](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/mcan_large.yml) | 7e-5    | 67.50       | 85.14      | 49.66      | 58.80     |

#### Train+val -> Test-dev

| Model                                                                                  | Base lr | Overall (%) | Yes/No (%) | Number (%) | Other (%) | Download                                                                                                                  |
|:--------------------------------------------------------------------------------------:|:-------:|:-----------:|:----------:|:----------:|:---------:|:-------------------------------------------------------------------------------------------------------------------------:|
| [BUTD](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/butd.yml)             | 2e-3    | 66.98       | 83.28      | 46.19      | 57.85     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EWSOkcCVGMpAot9ol0IJP3ABv3cWFRvGFB67980PHiCk3Q?e=OkjDhj) |
| [MFB](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/mfb.yml)               | 7e-4    | 68.29       | 84.64      | 48.29      | 58.89     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ET-B23hG7UNPrQ0hha77V5kBMxAokIr486lB3YwMt-zhow?e=XBk7co) |
| [MFH](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/mfh.yml)               | 7e-4    | 69.11       | 85.56      | 48.81      | 59.69     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EUpvJD3c7NZJvBAbFOXTS0IBk1jCSz46bi7Pfq1kzJ35PA?e=be97so) |
| [BAN-4](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/ban_4.yml)           | 1.4e-3  | 68.9        | 85.0       | 49.5       | 59.56     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EVUabhYppDBImgV6b0DdGr0BrxTdSLm7ux9rN65T_8DZ0Q?e=zSGIYg) |
| [BAN-8](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/ban_8.yml)           | 1.4e-3  | 69.07       | 85.2       | 49.63      | 59.71     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EbJgyL7FPTFAqzMm3HB1xDIBjXpWygOoXrdnDZKEIu34rg?e=kxCVue) |
| [MCAN-small](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/mcan_small.yml) | 1e-4    | 70.33       | 86.77      | 52.14      | 60.40     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EcFeQCi_9MVBn6MeESly8OYBZCeBEuaPQqZjT-oXidgKKg?e=5dGjUt) |
| [MCAN-large](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/mcan_large.yml) | 5e-5    | 70.48       | 86.90      | 52.11      | 60.63     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/Ee6HdFN_FcZAsQEm85WesHgBZBkY8dZ-278dDYG_ty_IwA?e=WK4SX4) |

#### Train+val+vg -> Test-dev

| Model                                                                                  | Base lr | Overall (%) | Yes/No (%) | Number (%) | Other (%) | Download                                                                                                                  |
|:--------------------------------------------------------------------------------------:|:-------:|:-----------:|:----------:|:----------:|:---------:|:-------------------------------------------------------------------------------------------------------------------------:|
| [BUTD](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/butd.yml)             | 2e-3    | 67.54       | 83.48      | 46.97      | 58.62     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EbLMhJsx9AVJi-ipqtkzHckBS5TWo_au3T8wHPEdDKMgPQ?e=kozuxV) |
| [MFB](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/mfb.yml)               | 7e-4    | 68.25       | 84.79      | 48.24      | 58.68     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EfLYkv1XBgNJgOMU5PAo04YBHxAVmpeJtnZecqJztJdNig?e=OVPJSk) |
| [MFH](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/mfh.yml)               | 7e-4    | 68.86       | 85.38      | 49.27      | 59.21     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EXGNuWmba8JOnQkkpfqokqcBzJ6Yw1ID6hl7hj2nyJaNJA?e=3TL5HC) |
| [BAN-4](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/ban_4.yml)           | 1.4e-3  | 69.31       | 85.42      | 50.15      | 59.91     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ERAUbsBJzcNHjXcINxDoWOQByR0jSbdNp8nonuFdbyc8yA?e=B5iGKU) |
| [BAN-8](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/ban_8.yml)           | 1.4e-3  | 69.48       | 85.40      | 50.82      | 60.14     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EW6v-dZOdJhFoKwT3bIx8M8B_U998hE8YD9zUJsUpo0rjQ?e=znhy2f) |
| [MCAN-small](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/mcan_small.yml) | 1e-4    | 70.69       | 87.08      | 53.16      | 60.66     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EWSniKgB8Y9PropErzcAedkBKwJCeBP6b5x5oT_I4LiWtg?e=HZiGuf) |
| [MCAN-large](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/mcan_large.yml) | 5e-5    | 70.82       | 87.19      | 52.56      | 60.98     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EQvT2mjBm4ZGnE-jBgAJCbIBC9RBiHwl-XEDr8T63DS10w?e=HjYsOA) |

## GQA
We provide a group of results (including *Accuracy*, *Binary*, *Open*, *Validity*, *Plausibility*, *Consistency*, *Distribution*) for each model on GQA as follows.  

- **Train+val -> Test-dev**: trained on the `train(balance) + val(balance)` splits and evaluated on the `test-dev(balance)` split. 

**The results shown in the following are obtained from the [online server](https://evalai.cloudcv.org/web/challenges/challenge-page/225/overview). Note that the offline Test-dev result is evaluated by the provided offical script, which results in slight difference compared to the online result due to some unknown reasons.**

#### Train+val -> Test-dev

| Model | Base lr | Accuracy (%) | Binary (%) | Open (%) | Validity (%) | Plausibility (%) | Consistency (%) | Distribution | Download |
|:------:|:-------:|:------------:|:----------:|:--------:|:------------:|:----------------:|:----------------:|:------------:|:--------:|
| [BUTD (frcn+bbox)](https://github.com/MILVLG/openvqa/tree/master/configs/gqa/butd.yml)        | 2e-4    | 53.38       | 67.78      | 40.72      | 96.62     | 84.81     | 77.62     | 1.26     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EaalaQ6VmBJCgeoZiPp45_gBn20g7tpkp-Uq8IVFcun64w?e=WgRMEj) |
| [BAN-4 (frcn+bbox)](https://github.com/MILVLG/openvqa/tree/master/configs/gqa/ban_4.yml)        | 2e-4    | 55.01       | 72.02      | 40.06      | 96.94     | 85.67     | 81.85     | 1.04     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EdRIuVXaJqBJoXg3T7N0xfYBsPl-GlgW2hq2toqm2gOxXg?e=hPng3c) |
| [BAN-8 (frcn+bbox)](https://github.com/MILVLG/openvqa/tree/master/configs/gqa/ban_8.yml)        | 1e-4    | 56.19       | 73.31      | 41.13      | 96.77     | 85.58     | 84.64     | 1.09     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ES8FCQxFsqJBnvdoOcF_724BJgJml6iStYYK9UeUbI8Uyw?e=Pcff9r) |
| [MCAN-small (frcn)](https://github.com/MILVLG/openvqa/tree/master/configs/gqa/mcan_small.yml) | 1e-4    | 53.41       | 70.29      | 38.56      | 96.77     | 85.32     | 82.29     | 1.40     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ER_i5xbPuXNCiC15iVtxBvgBTe7IBRpqpWTmeAY5svv3Ew?e=w8iJpv) |
| [MCAN-small (frcn+grid)](https://github.com/MILVLG/openvqa/tree/master/configs/gqa/mcan_small.yml) | 1e-4    | 54.28       | 71.68      | 38.97      | 96.79     | 85.11     | 84.49     | 1.20     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EbsPhIGkvpNKtqBbFmIFIucBQO_dM6lDgQL-gdd3RnzziQ?e=4uKDlw) |
| [MCAN-small (frcn+bbox)](https://github.com/MILVLG/openvqa/tree/master/configs/gqa/mcan_small.yml) | 1e-4    | 58.20       | 75.87      | 42.66      | 97.01     | 85.41     | 87.99     | 1.25     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EQCUNFPnpC1HliLDFCSDUc4BUdbdq40iPZVi5tLOCrVaQA?e=2aldJS) |
| [MCAN-small (frcn+bbox+grid)](https://github.com/MILVLG/openvqa/tree/master/configs/gqa/mcan_small.yml) | 1e-4    | 58.38       | 76.49      | 42.45      | 96.98     | 84.47     | 87.36     | 1.29     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EcrY2vDlzERLksouT5_cbcIBM1BCPkPdg4MyPmci8xrQig?e=UpPTao) |
| [MCAN-large (frcn+bbox+grid)](https://github.com/MILVLG/openvqa/tree/master/configs/gqa/mcan_large.yml) | 5e-5    | 58.10       | 76.98      | 41.50      | 97.01     | 85.43     | 87.34     | 1.20     | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/Ed6PBjIDEHpDot3vY__T-OIBJGdW51RFo2u_pm-7S5TMPA?e=zTSwPZ) |


## CLEVR

We provide a group of results (including *Overall*, *Count*, *Exist*, *Compare Numbers*, *Query Attribute*, *Compare Attribute*) for each model on CLEVR as follows.  

- **Train -> Val**: trained on the `train` split and evaluated on the `val` split. 

#### Train -> Val

| Model | Base lr | Overall (%) | Count (%) | Exist (%) | Compare Numbers (%) | Query Attribute (%) | Compare Attribute (%) | Download |
|:-----:|:-------:|:-------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [MCAN-small](https://github.com/MILVLG/openvqa/tree/master/configs/clevr/mcan_small.yml) | 4e-5 | 98.74 | 96.81 | 99.27 | 98.89 | 99.53 | 99.19 | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ERtwnuAoeHNKjs0qTkWC3cYBWVuUk7BLk88cnCKNFxYYlQ?e=lTRULt) |
