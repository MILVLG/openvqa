# --------------------------------------------------------
# OpenVQA
# CLEVR images feature extraction script
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

'''
Command line example:
python clevr_extract_feat.py --mode=all --gpu=0

python clevr_extract_feat.py --mode=train --gpu=0 --model=resnet101 --model_stage=3 --batch_size=128 --image_height=224 --image_width=224
'''

import argparse, os, json
import h5py
import numpy as np
from scipy.misc import imread, imresize

import torch
import torchvision
torch.set_num_threads(5)

def build_model(args):
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if not 'resnet' in args.model:
        raise ValueError('Feature extraction only supports ResNets')

    cnn = getattr(torchvision.models, args.model)(pretrained=True)
    layers = [cnn.conv1, 
              cnn.bn1,
              cnn.relu,
              cnn.maxpool]
    for i in range(args.model_stage):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(cnn, name))

    model = torch.nn.Sequential(*layers)
    model.cuda()
    model.eval()
    return model


def batch_feat(cur_batch, model):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    # image_batch = torch.FloatTensor(image_batch)
    image_batch = torch.FloatTensor(image_batch).cuda()
    image_batch = torch.autograd.Variable(image_batch, volatile=True)

    feats = model(image_batch)
    feats = feats.data.cpu().clone().numpy()

    return feats


def extract_feature(args, images_path, h5_path):
    input_paths = []
    idx_set = set()
    for file in os.listdir(images_path):
        if not file.endswith('.png'):
            continue
        idx = int(os.path.splitext(file)[0].split('_')[-1])
        input_paths.append((os.path.join(images_path, file), idx))
        idx_set.add(idx)
    input_paths.sort(key=lambda x: x[1])
    assert len(idx_set) == len(input_paths)
    assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1
    print('Image number:', len(input_paths))

    model = build_model(args)

    img_size = (args.image_height, args.image_width)
    with h5py.File(h5_path, 'w') as fp:
        feat_dset = None
        i0 = 0
        cur_batch = []
        for i, (path, idx) in enumerate(input_paths):
            img = imread(path, mode='RGB')
            img = imresize(img, img_size, interp='bicubic')
            img = img.transpose(2, 0, 1)[None]
            cur_batch.append(img)
            if len(cur_batch) == args.batch_size:
                feats = batch_feat(cur_batch, model)
                if feat_dset is None:
                    N = len(input_paths)
                    _, C, H, W = feats.shape
                    feat_dset = fp.create_dataset('features', (N, C, H, W), dtype=np.float32)
                i1 = i0 + len(cur_batch)
                feat_dset[i0:i1] = feats
                i0 = i1
                print('Processed %d / %d images' % (i1, len(input_paths)), end='\r')
                cur_batch = []
        if len(cur_batch) > 0:
            feats = batch_feat(cur_batch, model)
            i1 = i0 + len(cur_batch)
            feat_dset[i0:i1] = feats
            print('Processed %d / %d images' % (i1, len(input_paths)), end='\r')
    
    print('Extract image features to generate h5 files sucessfully!')


def h5_2_npz(h5_path, npz_path):
    if not os.path.exists(npz_path):
        os.mkdir(npz_path)
        print('Create dir:', npz_path)

    imgfeat_dict = h5py.File(h5_path, 'r')
    x = imgfeat_dict['features']
    print(x.shape)
    for ix in range(x.shape[0]):
        print('%d/%d' % (ix, x.shape[0]), end='\r')
        np.savez(npz_path + str(ix) + '.npz', x=x[ix].reshape(1024, 196).transpose(1, 0))
    print('h5 files transform into npz files successfully!')


parser = argparse.ArgumentParser(description='clevr_extract_feat')
parser.add_argument('--mode', '-mode',  choices=['all', 'train', 'val', 'test'], default='all', help='mode', type=str)
parser.add_argument('--gpu', '-gpu', default='0', type=str)
parser.add_argument('--model', '-model', default='resnet101')
parser.add_argument('--model_stage', '-model_stage', default=3, type=int)
parser.add_argument('--batch_size', '-batch_size', default=128, type=int)

parser.add_argument('--image_height', '-image_height', default=224, type=int)
parser.add_argument('--image_width', '-image_width', default=224, type=int)


if __name__ == '__main__':
    train_images_path = './raws/images/train/'
    val_images_path = './raws/images/val/'
    test_images_path = './raws/images/test/'
    train_feat_h5_path = './raw/images/train.h5'
    val_feat_h5_path = './raw/images/val.h5'
    test_feat_h5_path = './raw/images/test.h5'
    train_npz_path = './feats/train/'
    val_npz_path = './feats/val/'
    test_npz_path = './feats/test/'

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print('mode:', args.mode)
    print('gpu:', args.gpu)
    print('model:', args.model)
    print('model_stage:', args.model_stage)
    print('batch_size:', args.batch_size)
    print('image_height:', args.image_height)
    print('image_width:', args.image_width)
    
    # process train images features
    if args.mode in ['train', 'all']:
        print('\nProcess [train] images features:')
        extract_feature(args, train_images_path, train_feat_h5_path)
        h5_2_npz(train_feat_h5_path, train_npz_path)

    # process val images features
    if args.mode in ['val', 'all']:
        print('\nProcess [val] images features:')
        extract_feature(args, val_images_path, val_feat_h5_path)
        h5_2_npz(val_feat_h5_path, val_npz_path)

    # processs test images features
    if args.mode in ['test', 'all']:
        print('\nProcess [test] images features:')
        extract_feature(args, test_images_path, test_feat_h5_path)
        h5_2_npz(test_feat_h5_path, test_npz_path)
    