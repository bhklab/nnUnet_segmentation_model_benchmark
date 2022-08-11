import nibabel as nib
import numpy as np
import glob
import os
import time
import itertools
import pandas as pd
import copy
from monai.metrics import compute_hausdorff_distance
from pathlib import Path

#ground truth label
label_dir = Path('/cluster/projects/radiomics/Gyn_Autosegmentation/Task80_gyn')
# label_dir1 = Path('/cluster/projects/radiomics/Gyn_Autosegmentation/Task80_benchmark')
#inference from model
data_dir =  '/cluster/projects/radiomics/Gyn_Autosegmentation/Task80_benchmark_inferNew/'
test_labels = [str(path) for path in sorted(label_dir.glob('labelsTs/*.nii.gz'))]
# test_labels = [str(path) for path in sorted(label_dir.glob('labelsTs/*.nii.gz'))]
# test_labels = sorted(glob.glob(os.path.join(label_dir,  '*.nii.gz')))
# test_images = sorted(glob.glob(os.path.join(data_dir, '*' ,'image', 'benchmark_*.nii.gz')))
# print(test_images)

def dice_coef_singlelabel(y_true, y_pred, label):
    y_true_i = copy.deepcopy(y_true)
    y_pred_i = copy.deepcopy(y_pred)
    y_true_i[y_true_i != label] = 0.
    y_pred_i[y_pred_i != label] = 0.
    y_true_i[y_true_i == label] = 1
    y_pred_i[y_pred_i == label] = 1
    return dice_coef(y_true_i[:,:,:38], y_pred_i[:,:,:38])

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 1e-6
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice=0
    for index in range(numLabels):
        idx = index + 1
        y_true_i = copy.deepcopy(y_true)
        y_pred_i = copy.deepcopy(y_pred)
        # print(y_true.shape)
        # print(y_pred.shape)
        # # print(y_true_i.shape)
        # # print(y_pred_i.shape)
        y_true_i[y_true_i != idx] = 0.
        y_pred_i[y_pred_i != idx] = 0.
        y_true_i[y_true_i == idx] = 1
        y_pred_i[y_pred_i == idx] = 1
        dice += dice_coef(y_true_i, y_pred_i)
    return dice/numLabels # taking average

def hausdorff_distance(y_true, y_pred, label):
    y_true_i = copy.deepcopy(y_true)
    y_pred_i = copy.deepcopy(y_pred)
    y_true_i[y_true_i != label] = 0.
    y_pred_i[y_pred_i != label] = 0.
    y_true_i[y_true_i == label] = 1
    y_pred_i[y_pred_i == label] = 1
    return compute_hausdorff_distance( y_pred_i, y_true_i, True)


# transforms = ['Affine', 
transforms = ['Affine', 'AnisoDownsample',
'BiasField',
'ContrastCompression',
'ContrastExpansion',
'ElasticDeformation',
'Ghosting',
'RandomMotion',
'RicianNoise',
'Smoothing']
severity_lvls = [1,2,3,4,5]
# severity_lvls = [1]


cols = [
    'model',
    'transform',
    'severity',
    'label',
    'image_id',
    'dice',
    # 'hd',
]
df = list()





model_name = 'nn_Unet'
task_id = 'Task_80_gyn'
out_dir = '/cluster/home/t114639uhn/segmentation_robustness/results'

## For calculating average dice across all labels on multilabelsegmentation images
# for i in list(itertools.product(transforms, severity_lvls)):
#     tr = i[0]
#     sev_lvl = i[1]
#     tr_lvl = tr + '_' + str(sev_lvl)
#     test_images = sorted(glob.glob(os.path.join(data_dir, tr_lvl ,'benchmark_*.nii.gz')))
#     for i in range(len(test_labels)-3):
#         path = test_labels[i].split('.')[-3]
#         img_num = path.split('/')[-1]
#         print(img_num)
#         pred = nib.load(test_images[i]).get_fdata()
#         label = nib.load(test_labels[i]).get_fdata()
#         dice = dice_coef_multilabel(label, pred, 4)
#         row_entry = [
#             model_name,
#             tr,
#             sev_lvl,
#             img_num,
#             dice,
#         ]
#         df.append(dict(zip(cols, row_entry)))

##for file globbig mismatch problem
# img_num_mapping = {
#     'gyn_0120':'00',
#     'gyn_0121':'01',
#     'gyn_0122':'02',
#     'gyn_0123':'03',
#     'gyn_0124':'04',
#     'gyn_0125':'05',
#     'gyn_0126':'06',
#     'gyn_0127':'07',
#     'gyn_0128':'08',
#     'gyn_0129':'09',
#     'gyn_0130':'010',
#     'gyn_0131':'011'}

# For calculating single label dice for each of the labels
for i in list(itertools.product(transforms, severity_lvls)):
    tr = i[0]
    sev_lvl = i[1]
    tr_lvl = tr + '_' + str(sev_lvl)
    # test_images = [str(path) for path in sorted(data_dir.glob(tr_lvl + '/image/benchmark_*.nii.gz'))]
    for i in range(len(test_labels)):
        path = test_labels[i].split('.')[-3]
        img_num = path.split('/')[-1]
        print(img_num + tr_lvl)
        prefix = '000'
        length =  len(str(i))
        prefix = prefix[:(-1)*length]
        image_path = data_dir + tr_lvl + '/image/benchmark_' + prefix + str(i) + '.nii.gz'
        pred = nib.load(image_path).get_fdata()
        label = nib.load(test_labels[i]).get_fdata()
        print(pred.shape)
        print(label.shape)
        for i in range(4):
            label_num = i + 1
            dice = dice_coef_singlelabel(label, pred, label_num)
            # hd = hausdorff_distance(label, pred, label_num)
            # print(hd)
            row_entry = [
            model_name,
            tr,
            sev_lvl,
            label_num,
            img_num,
            dice,
            # hd,
            ]
            df.append(dict(zip(cols, row_entry)))



df = pd.DataFrame(df)
csv_name = 'nnunet_task_80_dice_multilabel.csv'
df.to_csv(os.path.join(out_dir, csv_name))
