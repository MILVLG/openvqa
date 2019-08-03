# Benchmark and Model Zoo

## Environment

We use the following environment to run all the experiments in this page.

- Python 3.6
- PyTorch 0.4.1
- CUDA 9.0.176
- CUDNN 7.0.4

## VQA-v2

We provide two groups of results (including the accuracies of *Overall*, *Yes/No*, *Number* and *Other*) for each model on VQA-v2 using different training schemas: 1) Model training on the `train` split and evaluated on the `val` split (Train -> Val); 2) Model training on the `train+val+vg` splits and evaluated on the `test-dev` split (Train+val+vg -> Test-dev). We only provide pre-trained models for the latter schema. 

**Note that for one model, the used base learning rate in the two schemas may be different, you should modify this setting in the config file to reproduce the results.**

#### Train -> Val


Model | Base lr | Overall (%) | Yes/No (%) | Number (%) | Other (%)
:-: | :-: | :-: | :-: | :-: | :-: 
[BUTD](./configs/vqa/butd.yml) |2e-3| 63.84 | 81.40 | 43.81 | 55.78 |
[BAN-4](./configs/vqa/ban_4.yml) |2e-3| 65.86 | 83.53 | 46.36 | 57.56 |
[BAN-8](./configs/vqa/ban_8.yml) |2e-3| 66.00 | 83.61 | 47.04 | 57.62 |
[MFB](./configs/vqa/mfb.yml) |7e-4| 65.35 | 83.23 | 45.31 | 57.05 |
[MFH](./configs/vqa/mfh.yml) |7e-4| 66.18 | 84.07 | 46.55 | 57.78 |
[MCAN-small](./configs/vqa/mcan_small.yml) |1e-4| 67.17 | 84.82 | 49.31 | 58.48 | 
[MCAN-large](./configs/vqa/mcan_large.yml) |7e-5| 67.50 | 85.14 | 49.66 | 58.80 | 

#### Train+val+vg -> Test-dev

Model | Base lr | Overall (%) | Yes/No (%) | Number (%) | Other (%) | Download
:-: | :-: | :-: |:-: |:-: |:-: | :-:
[BUTD](./configs/vqa/butd.yml)             | 2e-3 | 67.54 | 83.48 | 46.97 | 58.62 |  [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EbLMhJsx9AVJi-ipqtkzHckBS5TWo_au3T8wHPEdDKMgPQ?e=kozuxV)
[BAN-4](./configs/vqa/ban_4.yml)           |1.4e-3| 69.31 | 85.42 | 50.15 | 59.91 |  [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ERAUbsBJzcNHjXcINxDoWOQByR0jSbdNp8nonuFdbyc8yA?e=B5iGKU)
[BAN-8](./configs/vqa/ban_8.yml)           |1.4e-3| 69.48 | 85.40 | 50.82 | 60.14 |  [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EW6v-dZOdJhFoKwT3bIx8M8B_U998hE8YD9zUJsUpo0rjQ?e=znhy2f)
[MFB](./configs/vqa/mfb.yml)               |7e-4  | 68.25 | 84.79 | 48.24 | 58.68 |  [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EfLYkv1XBgNJgOMU5PAo04YBHxAVmpeJtnZecqJztJdNig?e=OVPJSk)
[MFH](./configs/vqa/mfh.yml)               |7e-4  | 68.86 | 85.38 | 49.27 | 59.21 |  [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EXGNuWmba8JOnQkkpfqokqcBzJ6Yw1ID6hl7hj2nyJaNJA?e=3TL5HC)
[MCAN-small](./configs/vqa/mcan_small.yml) |1e-4  | 70.69 | 87.08 | 53.16 | 60.66 |  [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EWSniKgB8Y9PropErzcAedkBKwJCeBP6b5x5oT_I4LiWtg?e=HZiGuf)
[MCAN-large](./configs/vqa/mcan_large.yml) |5e-5  | 70.82 | 87.19 | 52.56 | 60.98 |  [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EQvT2mjBm4ZGnE-jBgAJCbIBC9RBiHwl-XEDr8T63DS10w?e=HjYsOA)

## GQA


## CLEVR



