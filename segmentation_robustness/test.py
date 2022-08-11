# import argparse
import glob
import os
import time

from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance, compute_meandice
from monai.networks.layers import Norm
from monai.networks.nets import UNETR
import monai.transforms as tf
from monai.utils import get_torch_version_tuple, set_determinism
import numpy as np
import pandas as pd
import torch
import pickle
import joblib

print_config()
#this testing module is adjusted to be used with nn_Unet with saved inference images.
# set AMP
amp = True

if amp and get_torch_version_tuple() < (1, 6):
    raise RuntimeError('AMP feature only exists in PyTorch version greater than v1.6.')


out_dir = '/cluster/home/t114639uhn/segmentation_robustness/results'
print(f'Files will be saved to: {out_dir}')

set_determinism(seed=0)

# get data files
#directory for clean image
orig_dir = '/cluster/projects/radiomics/Gyn_Autosegmentation/Task80_gyn'
#inference from nn_Unet
data_dir =  '/cluster/projects/radiomics/Gyn_Autosegmentation/Task80_benchmark_infer'
#label from benchmarking
# label_dir = '/cluster/projects/radiomics/Gyn_Autosegmentation/Task80_benchmark'
label_dir = '/cluster/projects/radiomics/Gyn_Autosegmentation/Task80_gyn/labelsTs'
#load clean image and ground truth
test_images_clean = sorted(glob.glob(os.path.join(orig_dir,  'imagesTs', '*.nii.gz')))
test_labels_clean = sorted(glob.glob(os.path.join(orig_dir,  'labelsTs', '*.nii.gz')))

#load inference
test_images = sorted(glob.glob(os.path.join(data_dir, '*', 'benchmark_*.nii.gz')))
# test_labels = sorted(glob.glob(os.path.join(label_dir, '*',  'bencmark_*.nii.gz')))
test_labels = sorted(glob.glob(os.path.join(label_dir, '*.nii.gz')))

# test_images = [x for x in test_images if 'Clean_0' not in x]

test_images = test_images_clean + test_images
test_labels = test_labels_clean + test_labels

test_files = [
    {'image': image_name, 'label': label_name}
    for image_name, label_name in zip(test_images, test_labels)
]

print(f'Loaded {len(test_files)} for testing...')



base_transforms = [
    tf.LoadImaged(keys=['image', 'label']),
    tf.EnsureChannelFirstd(keys=['image', 'label']),
    tf.Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), diagonal=True),
    tf.ToTensord(keys=['image', 'label']),
    tf.AsDiscreted(keys=['label'], threshold_values=True),
    tf.ToNumpyd(keys=['image', 'label']),
    tf.NormalizeIntensityd(keys=['image'], channel_wise=True)
]



# model_strs = ['resunet_0.000']
# for prob in probs[1:]:
#     model_strs.append(f'resunet_{prob:.3f}')

# models = dict()
# for model_str in model_strs:
#     models[model_str] = dict()
#     models[model_str]['patch_based'] = True
#     models[model_str]['patch_size'] = 96
#     models[model_str]['post_transforms'] = [
#         tf.ToTensord(keys=['image', 'label'])
#     ]
#     models[model_str]['channels'] = (16, 32, 64, 128, 256)



# model_dir = '/cluster/projects/radiomics/Gyn_Autosegmentation/Task_95_model/Mytrainer__nnUNetPlansv2.1/fold_4/model_best.model.pkl'
# device = torch.device("cuda:0")
device = torch.device("cpu")
post_pred = tf.AsDiscrete(argmax=True, to_onehot=True, n_classes=4)
post_label = tf.AsDiscrete(to_onehot=True, n_classes=4)

df = list()
cols = [
    'model',
    'transform',
    'severity',
    'subject',
    'time',
    'dice',
    'hd95',
    'tp',
    'fp',
    'fn',
    'tn'
]

#adjust difference in label and inference
def adjust_label(label):
    label[label == 5] = 0
    return label

# model setup
model_name = 'nnUnet'
settings = {
    'patch_based': False,
    'patch_size': 96,
    'post_transforms': [
        tf.ToTensord(keys=['image', 'label'])
    ]
}
#loading model for testing
# print('Loading model...')
# model = UNETR(in_channels = 3, out_channels = 4, img_size = (320,320,50))

# model.load_state_dict(torch.load(model_dir))
# model.eval(in_channels = 3, out_channels = 4, img_size = (320,320,50))
# model = joblib.load(model_dir)
# model = pickle.load(open(model_dir, "rb"))
# with open(model_dir, 'rb') as f:
#     obj = f.read()
# model = pickle.loads(obj, encoding='latin1')
# model = torch.load(model_dir).to(device)
# model.eval()
# dataset and dataloader
transforms = tf.Compose(base_transforms + settings['post_transforms'])
test_ds = Dataset(data=test_files, transform=transforms)
test_loader = DataLoader(test_ds, batch_size=1)
# eval loop
a = 0
with torch.no_grad():
    step = 0
    step_start = time.time()
    for test_data in test_loader:
        subject_id = test_data['label_meta_dict']['filename_or_obj'][0].split('/')[-2]
        transform_name, sv_lvl = test_data['label_meta_dict']['filename_or_obj'][0].split('/')[-3].split('_')
        step += 1
        test_inputs, test_labels = (
            test_data["image"].to(device),
            test_data["label"].to(device),
        )
        # if amp:
        #     with torch.cuda.amp.autocast():
        #         if not settings['patch_based']:
        #             test_outputs = model(test_inputs)
        #         else:
        #             roi_size = (settings['patch_size'],)*3
        #             sw_batch_size = 4
        #             test_outputs = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
        # if a == 0:
        #     if not settings['patch_based']:
        #         test_outputs = model.predict(test_inputs)
        #     else:
        #         roi_size = (settings['patch_size'],)*3
        #         sw_batch_size = 4
        #         test_outputs = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
        # test_outputs = post_pred(test_outputs)
        # test_labels = post_label(test_labels)
        # dice = compute_meandice(
        #     y_pred=test_outputs,
        #     y=test_labels,
        #     include_background=False,
        # )
        # dice = dice.sum().item()
        # hd95 = compute_hausdorff_distance(
        #     y_pred=test_outputs,
        #     y=test_labels,
        #     include_background=False,
        #     percentile=95.0
        # )
        # hd95 = hd95.sum().item()
        print(torch.unique(test_inputs))
        test_labels = adjust_label(test_labels)
        print(test_labels.shape)
        dice = compute_meandice(
            y_pred = test_inputs,
            y = test_labels,
            include_background=False
        ).item()
        hd95 = compute_hausdorff_distance(
            y_pred =test_inputs,
            y=test_labels,
            include_background=False,
            percentile=95
        ).item()
        test_inputs = post_pred(test_inputs[0]).unsqueeze(0)
        test_labels = post_label(test_labels[0]).unsqueeze(0)
        test_inputs = test_inputs[0]
        test_labels = test_labels[0]
        # print(torch.mul(test_inputs, test_labels).sum(dim=(2, 3)))
        # tn, tp = torch.mul(test_inputs, test_labels).sum(dim=(2, 3))
        # tn = tn.item()
        # tp = tp.item()
        tn = 0
        tp = 0
        fp = torch.nn.functional.relu(test_inputs[1] - test_labels[1]).sum().item()
        fn = torch.nn.functional.relu(test_inputs[0] - test_labels[0]).sum().item()
        step_time = time.time() - step_start
        row_entry = [
            model_name,
            transform_name,
            sv_lvl,
            subject_id,
            step_time,
            dice,
            hd95,
            tp,
            fp,
            fn,
            tn
        ]
        df.append(dict(zip(cols, row_entry)))
        print(f"{step}/{len(test_ds)}, Dice: {dice:.4f}, HD95: {hd95:.4f}"
        f" step time: {(step_time):.4f}")
        step_start = time.time()

df = pd.DataFrame(df)
print('Saving dataframe to csv in output directory...')
# if len(args.models) != 0:
#     csv_name = f"{args.models[0]}.csv"
# else:
#     csv_name = "eval.csv"
csv_name = 'unetnn_task_80_results'
df.to_csv(os.path.join(out_dir, csv_name))
