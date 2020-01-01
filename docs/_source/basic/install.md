# Installation

This page provides basic prerequisites to run OpenVQA, including the setups of hardware, software, and datasets.

## Hardware & Software Setup

A machine with at least **1 GPU (>= 8GB)**, **20GB memory** and **50GB free disk space** is required.  We strongly recommend to use a SSD drive to guarantee high-speed I/O.

The following packages are required to build the project correctly.

- [Python](https://www.python.org/downloads/) >= 3.5
- [Cuda](https://developer.nvidia.com/cuda-toolkit) >= 9.0 and [cuDNN](https://developer.nvidia.com/cudnn)
- [PyTorch](http://pytorch.org/) >= 0.4.1 with CUDA (**PyTorch 1.x is also supported**).
- [SpaCy](https://spacy.io/) and initialize the [GloVe](https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz) as follows:

```bash
$ pip install -r requirements.txt
$ wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
$ pip install en_vectors_web_lg-2.1.0.tar.gz
```

## Dataset Setup

The following datasets should be prepared before running the experiments. 

**Note that if you only want to run experiments on one specific dataset, you can focus on the setup for that and skip the rest.** 

### VQA-v2

- Image Features

The image features are extracted using the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) strategy, with each image being represented as an dynamic number (from 10 to 100) of 2048-D features. We store the features for each image in a `.npz` file. You can prepare the visual features by yourself or download the extracted features from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EsfBlbmK1QZFhCOFpr4c5HUBzUV0aH2h1McnPG1jWAxytQ?e=2BZl8O) or [BaiduYun](https://pan.baidu.com/s/1C7jIWgM3hFPv-YXJexItgw#list/path=%2F). The downloaded files contains three files: **train2014.tar.gz, val2014.tar.gz, and test2015.tar.gz**, corresponding to the features of the train/val/test images for *VQA-v2*, respectively. 

All the image feature files are unzipped and placed in the `data/vqa/feats` folder to form the following tree structure:

```
|-- data
	|-- vqa
	|  |-- feats
	|  |  |-- train2014
	|  |  |  |-- COCO_train2014_...jpg.npz
	|  |  |  |-- ...
	|  |  |-- val2014
	|  |  |  |-- COCO_val2014_...jpg.npz
	|  |  |  |-- ...
	|  |  |-- test2015
	|  |  |  |-- COCO_test2015_...jpg.npz
	|  |  |  |-- ...
```

- QA Annotations

Download all the annotation `json` files for VQA-v2, including the [train questions](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip), [val questions](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip), [test questions](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip), [train answers](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip), and [val answers](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip). 

In addition, we use the VQA samples from the Visual Genome to augment the training samples. We pre-processed these samples by two rules: 

1. Select the QA pairs with the corresponding images appear in the MS-COCO *train* and *val* splits; 
2. Select the QA pairs with the answer appear in the processed answer list (occurs more than 8 times in whole *VQA-v2* answers).

We provide our processed vg questions and annotations files, you can download them from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EmVHVeGdck1IifPczGmXoaMBFiSvsegA6tf_PqxL3HXclw) or [BaiduYun](https://pan.baidu.com/s/1QCOtSxJGQA01DnhUg7FFtQ#list/path=%2F).

All the QA annotation files are unzipped and placed in the `data/vqa/raw` folder to form the following tree structure:

```
|-- data
	|-- vqa
	|  |-- raw
	|  |  |-- v2_OpenEnded_mscoco_train2014_questions.json
	|  |  |-- v2_OpenEnded_mscoco_val2014_questions.json
	|  |  |-- v2_OpenEnded_mscoco_test2015_questions.json
	|  |  |-- v2_OpenEnded_mscoco_test-dev2015_questions.json
	|  |  |-- v2_mscoco_train2014_annotations.json
	|  |  |-- v2_mscoco_val2014_annotations.json
	|  |  |-- VG_questions.json
	|  |  |-- VG_annotations.json

```

### GQA

- Image Features
  
Download the [spatial features](https://nlp.stanford.edu/data/gqa/spatialFeatures.zip) and [object features](https://nlp.stanford.edu/data/gqa/objectFeatures.zip) for GQA from its official website. **Spatial Features Files** include `gqa_spatial_*.h5` and `gqa_spatial_info.json`. **Object Features Files** include `gqa_objects_*.h5` and `gqa_objects_info.json`.  
To make the input features consistent with those for VQA-v2, we provide a [script](https://github.com/MILVLG/openvqa/tree/master/data/gqa/gqa_feat_preproc.py) to transform `.h5` feature files into multiple `.npz` files, with each file corresponding to one image. 

```bash
$ cd data/gqa

$ unzip spatialFeatures.zip
$ python gqa_feat_preproc.py --mode=spatial --spatial_dir=./spatialFeatures --out_dir=./feats/gqa-grid
$ rm -r spatialFeatures.zip ./spatialFeatures

$ unzip objectFeatures.zip
$ python gqa_feat_preproc.py --mode=object --object_dir=./objectFeatures --out_dir=./feats/gqa-frcn
$ rm -r objectFeatures.zip ./objectFeatures
```

All the processed feature files are placed in the `data/gqa/feats` folder to form the following tree structure:

```
|-- data
	|-- gqa
	|  |-- feats
	|  |  |-- gqa-frcn
	|  |  |  |-- 1.npz
	|  |  |  |-- ...
	|  |  |-- gqa-grid
	|  |  |  |-- 1.npz
	|  |  |  |-- ...
```

- Questions and Scene Graphs

Download all the GQA [QA files](https://nlp.stanford.edu/data/gqa/questions1.2.zip) from the official site, including all the splits needed for training, validation and testing. Download the [scene graphs files](https://nlp.stanford.edu/data/gqa/sceneGraphs.zip) for `train` and `val` splits from the official site. Download  the [supporting files](https://nlp.stanford.edu/data/gqa/eval.zip) from the official site, including the `train` and `val` choices supporting files for the evaluation.  

All the question files and scene graph files are unzipped and placed in the `data/gqa/raw` folder to form the following tree structure:

```
|-- data
	|-- gqa
	|  |-- raw
	|  |  |-- questions1.2
	|  |  |  |-- train_all_questions
	|  |  |  |  |-- train_all_questions_0.json
	|  |  |  |  |-- ...
	|  |  |  |  |-- train_all_questions_9.json
	|  |  |  |-- train_balanced_questions.json
	|  |  |  |-- val_all_questions.json
	|  |  |  |-- val_balanced_questions.json
	|  |  |  |-- testdev_all_questions.json
	|  |  |  |-- testdev_balanced_questions.json
	|  |  |  |-- test_all_questions.json
	|  |  |  |-- test_balanced_questions.json
	|  |  |  |-- challenge_all_questions.json
	|  |  |  |-- challenge_balanced_questions.json
	|  |  |  |-- submission_all_questions.json
	|  |  |-- eval
	|  |  |  |-- train_choices
	|  |  |  |  |-- train_all_questions_0.json
	|  |  |  |  |-- ...
	|  |  |  |  |-- train_all_questions_9.json
	|  |  |  |-- val_choices.json
	|  |  |-- sceneGraphs
	|  |  |  |-- train_sceneGraphs.json
	|  |  |  |-- val_sceneGraphs.json
```

### CLEVR

- Images, Questions and Scene Graphs

Download all the [CLEVR v1.0](https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip) from the official site, including all the splits needed for training, validation and testing.  

All the image files, question files and scene graph files are unzipped and placed in the `data/clevr/raw` folder to form the following tree structure:

```
|-- data
	|-- clevr
	|  |-- raw
	|  |  |-- images
	|  |  |  |-- train
	|  |  |  |  |-- CLEVR_train_000000.json
	|  |  |  |  |-- ...
	|  |  |  |  |-- CLEVR_train_069999.json
	|  |  |  |-- val
	|  |  |  |  |-- CLEVR_val_000000.json
	|  |  |  |  |-- ...
	|  |  |  |  |-- CLEVR_val_014999.json
	|  |  |  |-- test
	|  |  |  |  |-- CLEVR_test_000000.json
	|  |  |  |  |-- ...
	|  |  |  |  |-- CLEVR_test_014999.json
	|  |  |-- questions
	|  |  |  |-- CLEVR_train_questions.json
	|  |  |  |-- CLEVR_val_questions.json
	|  |  |  |-- CLEVR_test_questions.json
	|  |  |-- scenes
	|  |  |  |-- CLEVR_train_scenes.json
	|  |  |  |-- CLEVR_val_scenes.json
```

- Image Features
  
To make the input features consistent with those for VQA-v2, we provide a [script](https://github.com/MILVLG/openvqa/tree/master/data/clevr/clevr_extract_feat.py) to extract image features using a pre-trained ResNet-101 model like most previous works did and generate `.h5` files, with each file corresponding to one image. 

```bash
$ cd data/clevr

$ python clevr_extract_feat.py --mode=all --gpu=0
```

All the processed feature files are placed in the `data/clevr/feats` folder to form the following tree structure:

```
|-- data
	|-- clevr
	|  |-- feats
	|  |  |-- train
	|  |  |  |-- 1.npz
	|  |  |  |-- ...
	|  |  |-- val
	|  |  |  |-- 1.npz
	|  |  |  |-- ...
	|  |  |-- test
	|  |  |  |-- 1.npz
	|  |  |  |-- ...
```