# Adding a custom VQA model

This is a tutorial on how to add a custom VQA model into OpenVQA. Follow the steps below, you will obtain a model that can run across VQA/GQA/CLEVR datasets.

## 1. Preliminary

All implemented models are placed at ```<openvqa>/openvqa/models/```, so the first thing to do is to create a  folder there for your VQA model named by `<YOU_MODEL_NAME>`. After that, all your model related files will be placed in the folder  ```<openvqa>/openvqa/models/<YOU_MODEL_NAME>/```.

## 2. Dataset Adapter

Create a python file `<openvqa>/openvqa/models/<YOU_MODEL_NAME>/adapter.py` to bridge your model and different datasets. Different datasets have different input features, thus resulting in different operators to handle the features. 

#### Input

Input features (packed as `feat_dict`) for different datasets.

#### Output

Customized pre-processed features to be fed into the model.

#### Adapter Template

```
from openvqa.core.base_dataset import BaseAdapter
class Adapter(BaseAdapter):
    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C

    def vqa_init(self, __C):
    	# Your Implementation

    def gqa_init(self, __C):
    	# Your Implementation

    def clevr_init(self, __C):
    	# Your Implementation

    def vqa_forward(self, feat_dict):
    	# Your Implementation
       
    def gqa_forward(self, feat_dict):
    	# Your Implementation
        
    def clevr_forward(self, feat_dict):
    	# Your Implementation
    
```

Each dataset-specific initiation function `def <dataset>_init(self, __C)` corresponds to one feed-forward function `def <dataset>_forward(self, feat_dict)`, your implementations should follow the principles ```torch.nn.Module.__init__()``` and ```torch.nn.Module.forward()```, respectively.

The variable ` feat_dict`  consists of the input feature names for the datasets, which corresponds to the definitions in `<openvqa>/openvqa/core/base_cfg.py` 

```
vqa:{
	'FRCN_FEAT': buttom-up features -> [batchsize, num_bbox, 2048],
	'BBOX_FEAT': bbox coordinates -> [batchsize, num_bbox, 5],
}
gqa:{
	'FRCN_FEAT': official buttom-up features -> [batchsize, num_bbox, 2048],
	'BBOX_FEAT': official bbox coordinates -> [batchsize, num_bbox, 5],
	'GRID_FEAT': official resnet grid features -> [batchsize, num_grid, 2048],
}
clevr:{
	'GRID_FEAT': resnet grid features -> [batchsize, num_grid, 1024],
}
```

More detailed examples can be referred to the adapter for the [MCAN](https://github.com/MILVLG/openvqa/tree/master/openvqa/models/mcan/adapter.py) model.



## 3. Definition of model hyper-parameters

Create a python file named ```<openvqa>/openvqa/models/<YOUR MODEL NAME>/model_cfgs.py```

#### Configuration Template

```
from openvqa.core.base_cfgs import BaseCfgs
class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()
        # Your Implementation
```

Only the variable you defined here can be used in the network. The variable value can be override in the running configuration file described later. 

#### Example

```
# model_cfgs.py
from openvqa.core.base_cfgs import BaseCfgs
class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()
        self.LAYER = 6
```

```
# net.py
class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C
        
        print(__C.LAYER)
```

```
Output: 6
```

## 4. Main body

Create a python file for the main body of the model as ```<openvqa>/openvqa/models/<YOUR MODEL NAME>/net.py```. Note that the filename must be `net.py` since this filename will be invoked by the running script. Except the file, other auxiliary model files invoked by `net.py`  can be named arbitrarily.

When implementation, you should pay attention to the following restrictions:

- The main module should be named `Net`, i.e., `class Net(nn.Module):`
- The `init` function has three input variables: *pretrained_emb* corresponds to the  GloVe embedding features for the question; *token\_size* corresponds to the number of all dataset words; *answer_size* corresponds to the number of classes for prediction.
- The `forward` function has four input variables: *frcn_feat*, *grid_feat*, *bbox_feat*, *ques_ix*. 
- In the `init` function, you should initialize the `Adapter` which you've already defined above. In the `forward` function, you should feed *frcn_feat*, *grid_feat*, *bbox_feat* into the `Adapter` to obtain the processed image features.
- Return a prediction tensor of size [batch\_size, answer_size]. Note that no activation function like ```sigmoid``` or ```softmax``` is appended on the prediction. The activation has been designed for the prediction in the loss function outside.   

#### Model Template 

```
import torch.nn as nn
from openvqa.models.mcan.adapter import Adapter
class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
    	super(Net, self).__init__()
    	self.__C = __C
		self.adapter = Adapter(__C)
   
   	def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
   		img_feat = self.adapter(frcn_feat, grid_feat, bbox_feat)
   		# model implementation
   	    ...
   	    
		return pred
```

## 5. Declaration of running configurations

Create a  `yml` file at```<openvqa>/configs/<dataset>/<YOUR_CONFIG_NAME>.yml``` and define your hyper-parameters here. We suggest that `<YOUR_CONFIG_NAME>`= `<YOUR_MODEL_NAME>`. If you have the requirement to have one base model support the running scripts for different variants. (e.g., MFB and MFH), you can have different yml files (e.g., `mfb.yml` and `mfh.yml`) and use the `MODEL_USE` param in the yml file to specify the actual used model (i.e., mfb).  

### Example:
```
MODEL_USE: <YOUR MODEL NAME>  # Must be defined
LAYER: 6
LOSS_FUNC: bce
LOSS_REDUCTION: sum
```

Finally, to register the added model to the running script,  you can modify `<openvqa/run.py>` by adding your `<YOUR_CONFIG_NAME>` into the arguments for models [here](https://github.com/MILVLG/openvqa/tree/master/run.py#L22). 


By doing all the steps above, you are able to use ```--MODEL=<YOUR_CONFIG_NAME>```  to train/val/test your model like other provided models. For more information about the usage of the running script, please refer to the [Getting Started](https://openvqa.readthedocs.io/en/latest/basic/getting_started.html) page.   
