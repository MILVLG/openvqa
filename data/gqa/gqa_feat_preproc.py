# --------------------------------------------------------
# OpenVQA
# GQA spatial features & object features .h5 files to .npz files transform script
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

'''
Command line example:
(1) Process spatial features
python gqa_feat_preproc.py --mode=spatial --spatial_dir=./spatialFeatures --out_dir=./feats/gqa-grid

(2) Process object features
python gqa_feat_preproc.py --mode=object --object_dir=./objectFeatures --out_dir=./feats/gqa-frcn
'''

import h5py, glob, json, cv2, argparse
import numpy as np

# spatial features
def process_spatial_features(feat_path, out_path):
    info_file = feat_path + '/gqa_spatial_info.json'
    try:
        info = json.load(open(info_file, 'r'))
    except:
        print('Failed to open info file:', info_file)
        return
    print('Total grid features', len(info))
 
    print('Making the <h5 index> to <image id> dict...')
    h5idx_to_imgid = {}
    for img_id in info:
        h5idx_to_imgid[str(info[img_id]['file']) + '_' + str(info[img_id]['idx'])] = img_id

    for ix in range(16):
        feat_file = feat_path + '/gqa_spatial_' + str(ix) + '.h5'
        print('Processing', feat_file)
        try:
            feat_dict = h5py.File(feat_file, 'r')
        except:
            print('Failed to open feat file:', feat_file)
            return

        features = feat_dict['features']

        for iy in range(features.shape[0]):
            img_id = h5idx_to_imgid[str(ix) + '_' + str(iy)]
            feature = features[iy]
            # save to .npz file ['x']
            np.savez(
                out_path + '/' + img_id + '.npz',
                x=feature.reshape(2048, 49).transpose(1, 0),    # (49, 2048)
            )

    print('Process spatial features successfully!')


# object features
def process_object_features(feat_path, out_path):
    info_file = feat_path + '/gqa_objects_info.json'
    try:
        info = json.load(open(info_file, 'r'))
    except:
        print('Failed to open info file:', info_file)
        return
    print('Total frcn features', len(info))

    print('Making the <h5 index> to <image id> dict...')
    h5idx_to_imgid = {}
    for img_id in info:
        h5idx_to_imgid[str(info[img_id]['file']) + '_' + str(info[img_id]['idx'])] = img_id

    for ix in range(16):
        feat_file = feat_path + '/gqa_objects_' + str(ix) + '.h5'
        print('Processing', feat_file)

        try:
            feat_dict = h5py.File(feat_file, 'r')
        except:
            print('Failed to open feat file:', feat_file)
            return

        bboxes = feat_dict['bboxes']
        features = feat_dict['features']

        for iy in range(features.shape[0]):
            img_id = h5idx_to_imgid[str(ix) + '_' + str(iy)]
            img_info = info[img_id]
            objects_num = img_info['objectsNum']
            # save to .npz file ['x', 'bbox', 'width', 'height']
            np.savez(
                out_path + '/' + img_id + '.npz',
                x=features[iy, :objects_num],
                bbox=bboxes[iy, :objects_num],
                width=img_info['width'],
                height=img_info['height'],
            )

    print('Process object features successfully!')


parser = argparse.ArgumentParser(description='gqa_h52npz')
parser.add_argument('--mode', '-mode', choices=['object', 'spatial', 'frcn', 'grid'], help='mode', type=str)
parser.add_argument('--object_dir', '-object_dir', help='object features dir', type=str)
parser.add_argument('--spatial_dir', '-spatial_dir', help='spatial features dir', type=str)
parser.add_argument('--out_dir', '-out_dir', help='output dir', type=str)

args = parser.parse_args()   

mode = args.mode
object_path = args.object_dir
spatial_path = args.spatial_dir
out_path = args.out_dir

print('mode:', mode)
print('object_path:', object_path)
print('spatial_path:', spatial_path)
print('out_path:', out_path)

# process spatial features
if mode in ['spatial', 'grid']:
    process_spatial_features(spatial_path, out_path)

# process object features
if mode in ['object', 'frcn']:
    process_object_features(object_path, out_path)

