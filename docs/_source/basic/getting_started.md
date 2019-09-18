# Getting Started

This page provides basic tutorials about the usage of mmdetection.
For installation instructions, please see [Installation](install).

## Training

The following script will start training a `mcan_small` model on the `VQA-v2` dataset:

```bash
$ python3 run.py --RUN='train' --MODEL='mcan_small' --DATASET='vqa'
```

- ```--RUN={'train','val','test'}``` to set the mode to be executed.

- ```--MODEL=str```, e.g., to assign the model to be executed.

- ```--DATASET={'vqa','gqa','clevr'}``` to choose the dataset to be executed.

All checkpoint files will be saved to:

```
ckpts/ckpt_<VERSION>/epoch<EPOCH_NUMBER>.pkl
```

and the training log file will be placed at:

```
results/log/log_run_<VERSION>.txt
```

To add：

- ```--VERSION=str```, e.g., ```--VERSION='v1'``` to assign a name for your this model.

- ```--GPU=str```, e.g., ```--GPU='2'``` to train the model on specified GPU device.

- ```--SEED=int```, e.g., ```--SEED=123``` to use a fixed seed to initialize the model, which obtains exactly the same model. Unset it results in random seeds.

- ```--NW=int```, e.g., ```--NW=8``` to accelerate I/O speed.

- ```--SPLIT=str``` to set the training sets as you want.  Setting ```--SPLIT='train'```  will trigger the evaluation script to run the validation score after every epoch automatically.

- ```--RESUME=True``` to start training with saved checkpoint parameters. In this stage, you should assign the checkpoint version```--CKPT_V=str``` and the resumed epoch number ```CKPT_E=int```.

- ```--MAX_EPOCH=int``` to stop training at a specified epoch number.

If you want to resume training from an existing checkpoint, you can use the following script:

```bash
$ python3 run.py --RUN='train' --MODEL='mcan_small' --DATASET='vqa' --CKPT_V=str --CKPT_E=int
```

where the args `CKPT_V` and `CKPT_E` must be specified, corresponding to the version and epoch number of the loaded model.


####  Multi-GPU Training and Gradient Accumulation

We recommend to use the GPU with at least 8 GB memory, but if you don't have such device,  we provide two solutions to deal with it:

- _Multi-GPU Training_: 

    If you want to accelerate training or train the model on a device with limited GPU memory, you can use more than one GPUs:

	Add ```--GPU='0, 1, 2, 3...'```

    The batch size on each GPU will be adjusted to `BATCH_SIZE`/#GPUs automatically.

- _Gradient Accumulation_: 

    If you only have one GPU less than 8GB, an alternative strategy is provided to use the gradient accumulation during training:
	
	Add ```--ACCU=n```  
	
    This makes the optimizer accumulate gradients for`n` small batches and update the model weights at once. It is worth noting that  `BATCH_SIZE` must be divided by ```n``` to run this mode correctly. 


## Validation and Testing

**Warning**:  The args ```--MODEL``` and `--DATASET` should be set to the same values as those in the training stage.


### Validation on Local Machine

Offline evaluation on local machine only support the evaluations on the *val* split. If you want to evaluate the *test* split, please see [Evaluation on online server](#Evaluation on online server).

There are two ways to start:

(Recommend)

```bash
$ python3 run.py --RUN='val' --MODEL=str --DATASET='{vqa,gqa,clevr}' --CKPT_V=str --CKPT_E=int
```

or use the absolute path instead:

```bash
$ python3 run.py --RUN='val' --MODEL=str --DATASET='{vqa,gqa,clevr}' --CKPT_PATH=str
```

- For VQA-v2, the results on *val* split

### Testing on Online Server

All the evaluations on the test split of VQA-v2, GQA and CLEVR benchmarks can be achieved by using 

```bash
$ python3 run.py --RUN='test' --MODEL=str --DATASET='{vqa,gqa,clevr}' --CKPT_V=str --CKPT_E=int
```

Result file are saved at: ```results/result_test/result_run_<CKPT_V>_<CKPT_E>.json```

- For VQA-v2, the result file is uploaded the [VQA challenge website](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) to evaluate the scores on *test-dev* or *test-std* split.

- For GQA,  the result file is uploaded to the [GQA Challenge website](<https://evalai.cloudcv.org/web/challenges/challenge-page/225/overview>) to evaluate the scores on *test* or *test-dev* split. 
- For CLEVR, the result file can be evaluated via sending an email to the author [Justin Johnson](<https://cs.stanford.edu/people/jcjohns/>) with attaching this file, and he will reply the scores via email too.   