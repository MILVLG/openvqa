# Getting Started

This page provides basic tutorials about the usage of mmdetection.
For installation instructions, please see [INSTALL.md](INSTALL.md).

## Training

The following script will start training with the default hyperparameters:

```bash
$ python3 run.py --RUN='train' --MODEL='mcan_small' --DATASET='vqa'
```
All checkpoint files will be saved to:

```
ckpts/ckpt_<VERSION>/epoch<EPOCH_NUMBER>.pkl
```

and the training log file will be placed at:

```
results/log/log_run_<VERSION>.txt
```

To add：

1. ```--VERSION=str```, e.g.```--VERSION='small_model'``` to assign a name for your this model.

2. ```--GPU=str```, e.g.```--GPU='2'``` to train the model on specified GPU device.

3. ```--NW=int```, e.g.```--NW=8``` to accelerate I/O speed.

4. ```--MODEL={'small', 'large'}```  ( Warning: The large model will consume more GPU memory, maybe [Multi-GPU Training and Gradient Accumulation](#Multi-GPU-Training-and-Gradient-Accumulation) can help if you want to train the model with limited GPU memory.)

5. ```--SPLIT={'train', 'train+val', 'train+val+vg'}``` can combine the training datasets as you want. The default training split is ```'train+val+vg'```.  Setting ```--SPLIT='train'```  will trigger the evaluation script to run the validation score after every epoch automatically.

6. ```--RESUME=True``` to start training with saved checkpoint parameters. In this stage, you should assign the checkpoint version```--CKPT_V=str``` and the resumed epoch number ```CKPT_E=int```.

7. ```--MAX_EPOCH=int``` to stop training at a specified epoch number.

8. ```--PRELOAD=True``` to pre-load all the image features into memory during the initialization stage (Warning: needs extra 25~30GB memory and 30min loading time from an HDD drive).


####  Multi-GPU Training and Gradient Accumulation

We recommend to use the GPU with at least 8 GB memory, but if you don't have such device, don't worry, we provide two ways to deal with it:

1. _Multi-GPU Training_: 

    If you want to accelerate training or train the model on a device with limited GPU memory, you can use more than one GPUs:

	Add ```--GPU='0, 1, 2, 3...'```

    The batch size on each GPU will be adjusted to `BATCH_SIZE`/#GPUs automatically.

2. _Gradient Accumulation_: 

    If you only have one GPU less than 8GB, an alternative strategy is provided to use the gradient accumulation during training:
	
	Add ```--ACCU=n```  
	
    This makes the optimizer accumulate gradients for`n` small batches and update the model weights at once. It is worth noting that  `BATCH_SIZE` must be divided by ```n``` to run this mode correctly. 


## Validation and Testing

**Warning**: If you train the model use ```--MODEL``` args or multi-gpu training, it should be also set in evaluation.


#### Offline Evaluation

Offline evaluation only support the VQA 2.0 *val* split. If you want to evaluate on the VQA 2.0 *test-dev* or *test-std* split, please see [Online Evaluation](#Online-Evaluation).

There are two ways to start:

(Recommend)

```bash
$ python3 run.py --RUN='val' --CKPT_V=str --CKPT_E=int
```

or use the absolute path instead:

```bash
$ python3 run.py --RUN='val' --CKPT_PATH=str
```


#### Online Evaluation

The evaluations of both the VQA 2.0 *test-dev* and *test-std* splits are run as follows:

```bash
$ python3 run.py --RUN='test' --CKPT_V=str --CKPT_E=int
```

Result files are stored in ```results/result_test/result_run_<'PATH+random number' or 'VERSION+EPOCH'>.json```

You can upload the obtained result json file to [Eval AI](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) to evaluate the scores on *test-dev* and *test-std* splits.


## Pretrained models

We provide two pretrained models, namely the `small` model and the `large` model. The small model corrresponds to the one describe in our paper with slightly higher performance (the overall accuracy on the *test-dev* split is 70.63% in our paper) due to different pytorch versions. The large model uses a 2x larger `HIDDEN_SIZE=1024` compared to the small model with `HIDDEN_SIZE=512`. 

The performance of the two models on *test-dev* split is reported as follows:

_Model_ | Overall | Yes/No | Number | Other
:-: | :-: | :-: | :-: | :-:
_Small_ | 70.7 | 86.91 | **53.42** | 60.75| 
_Large_ | **70.93**| **87.39** | 52.78 | **60.98**|


These two models can be downloaded from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EtNU5OG1dNhMq8M1pgeuQZwBgcj2RQCVnzLGDeDfnejPMQ?e=ynYhvk) or [BaiduYun](https://pan.baidu.com/s/1GW_SFErXSIBJ2Ojg2qaRmw#list/path=%2F), and you should unzip and put them to the correct folders as follows:

```angular2html
|-- ckpts
	|-- ckpt_small
	|  |-- epoch13.pkl
	|-- ckpt_large
	|  |-- epoch13.pkl

```

Set ```--CKPT={'small', 'large'} --CKPT_E=13``` to testing or resume training, details can be found in [Training](#Training) and [Validation and Testing](#Validation-and-Testing). 
 
