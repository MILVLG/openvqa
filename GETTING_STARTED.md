# Getting Started

This page provides basic tutorials about the usage of mmdetection.
For installation instructions, please see [INSTALL.md](INSTALL.md).

## Training

The following script will start training with the default hyperparameters:

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

**Warning**: If you train the model use ```--MODEL``` args or multi-gpu training, it should be also set in evaluation.


### Offline Evaluation

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


### Online Evaluation

VQA-v2 and GQA support online evaluation.

For VQA-v2, *test-dev* and *test-std* splits are run as follows:

```bash
$ python3 run.py --RUN='test' --CKPT_V=str --CKPT_E=int
```

Result files are stored in ```results/result_test/result_run_<'PATH+random number' or 'VERSION+EPOCH'>.json```

You can upload the obtained result json file to [Eval AI](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) to evaluate the scores on *test-dev* and *test-std* splits.

For GQA, 
