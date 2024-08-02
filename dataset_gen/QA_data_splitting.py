"""
Check for duplication and overlapping in the train, val, and test data
"""


import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm

# load data
train_df = pd.read_csv('csv_detection_info_clean/train_actual_data.csv')
val_df = pd.read_csv('csv_detection_info_clean/val_data.csv')
test_df = pd.read_csv('csv_detection_info_clean/test_data.csv')

# test duplicated items
train_df[train_df.duplicated(subset='filename', keep=False)].sort_values('pid')
# Empty DataFrame
# Columns: [pid, base_path, filename, ax0_min, ax0_max, ax1_min, ax1_max, confidence, diagnosis, diagnosis_raw, year]
# Index: []
val_df[val_df.duplicated(subset='filename', keep=False)].sort_values('pid')
# Empty DataFrame
# Columns: [pid, base_path, filename, ax0_min, ax0_max, ax1_min, ax1_max, confidence, diagnosis, diagnosis_raw, year]
# Index: []
test_df[test_df.duplicated(subset='filename', keep=False)].sort_values('pid')
# Empty DataFrame
# Columns: [pid, base_path, filename, ax0_min, ax0_max, ax1_min, ax1_max, confidence, diagnosis, diagnosis_raw, year]
# Index: []


# concatenate train and val and test
df = pd.concat([train_df, val_df, test_df])

###################################################
# checking for duplication and overlapping
###################################################


# check using filename

df[df.duplicated(subset='filename', keep=False)].sort_values('pid')
# Empty DataFrame
# Columns: [pid, base_path, filename, ax0_min, ax0_max, ax1_min, ax1_max, confidence, diagnosis, diagnosis_raw, year]
# Index: []

df.shape
# (7229, 11)
df.filename.nunique()
# 7229


# check using images
h5_filename = 'P:/CubiAI/preprocess_data/datasets/elbow_normal_abnormal_800.h5'

# fold 0-3 training
# fold 4 validation
# fold 5 test


def train_datagen():
    for fold in range(3):
        print(f'fold_{fold}')
        with h5py.File(h5_filename, 'r') as f:
            images = f[f'fold_{fold}']['image'][:]
        for i, img in enumerate(images):
            print('train', i)
            yield img[..., 0]


with h5py.File(h5_filename, 'r') as f:
    val_images = f[f'fold_4']['image'][:]


def val_datagen():
    for i, img in enumerate(val_images):
        # print('val', i)
        yield img[..., 0]


with h5py.File(h5_filename, 'r') as f:
    test_images = f[f'fold_5']['image'][:]


def test_datagen():
    for i, img in enumerate(test_images):
        # print('test', i)
        yield img[..., 0]


for img in train_datagen():
    for img2 in tqdm(val_datagen()):
        if np.all(img == img2):
            print('train and val overlap')
            break
        if np.allclose(img, img2):
            print('train and val overlap')
            break

    for img2 in tqdm(test_datagen()):
        if np.all(img == img2):
            print('train and test overlap')
            break
        if np.allclose(img, img2):
            print('train and test overlap')
            break
# no overlap found
