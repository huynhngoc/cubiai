import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import StratifiedKFold
import os
import random
import tensorflow as tf

# update these settings
n_splits = 4
resize_shape = 800

# update these filenames
cropped_folder = 'P:/CubiAI/preprocess_data/cropped'
# csv_folder = 'P:/CubiAI/preprocess_data/csv_detection_info_clean'
# filenames = os.listdir('P:/CubiAI/preprocess_data/csv_detection_info_clean')

h5_filename = 'P:/CubiAI/preprocess_data/datasets/elbow_normal_abnormal_800.h5'
# concat all df, remember to reset index
train_df = pd.read_csv('csv_detection_info_clean/train_actual_data.csv')
val_df = pd.read_csv('csv_detection_info_clean/val_data.csv')
test_df = pd.read_csv('csv_detection_info_clean/test_data.csv')


# choose and adjust target data
# in some cases, diagnosis 2 should become 1
# In some other cases, for ex, multiclass problem with normal 0, level 1 - 3 then diagnosis can be kept as is, the same as this case
# Or if we want to separate level 1-3 then we need to change them into 0-2
# Similarly, if we want to separate normal, level 1 attrose & sklerose, level 2 attrose and primary lesion, level 3 MCD & OCD & UAP,
# we should transform them into correct category
diagnosis = train_df['diagnosis'].values.copy()
random_state = 42
random.seed(random_state)
folds = []
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
for train_indice, test_indice in skf.split(train_df.index, diagnosis):
    additional_UAP = random.sample(
        list(train_df[train_df.diagnosis_raw == '3, UAP'].index), 14)
    additional_OCD = random.sample(
        list(train_df[train_df.diagnosis_raw == '3, OCD'].index), 10)
    indice = np.concatenate([test_indice, additional_OCD, additional_UAP])
    np.random.shuffle(indice)
    folds.append(indice)

for indice in folds:
    print(train_df[train_df.index.isin(indice)].diagnosis_raw.value_counts())
    print(len(indice))
    # selected = train_df[train_df.index.isin(indice)]
    # print(selected[selected.diagnosis_raw.isin(['3, UAP', '3, OCD'])][['pid', 'diagnosis_raw']])

# print out to check folds
print('the target values in each fold')
for i, fold in enumerate(folds):
    print(i, diagnosis[fold])

# create the dataset
with h5py.File(h5_filename, 'w') as f:
    for i in range(len(folds)):
        f.create_group(f'fold_{i}')

train_pid = train_df.pid.values.copy()
for i, fold in enumerate(folds):
    images = []
    selected_df = train_df.iloc[fold]
    real_diagnosis = diagnosis[fold].copy()
    target = real_diagnosis.copy()
    target[target > 0] = 1
    for _, item in selected_df.iterrows():
        year = int(item['year'])
        diagnosis_raw = item['diagnosis_raw']
        filename = item['filename']
        cropped_fn = f'{cropped_folder}/{year}/{diagnosis_raw}/{filename}.npy'
        # add an additional dimension
        img = np.load(cropped_fn)[np.newaxis, ..., np.newaxis]
        # resize with bilinear (default)
        img = tf.image.resize_with_pad(img, resize_shape, resize_shape)
        images.append(img)
    images = np.concatenate(images)
    with h5py.File(h5_filename, 'a') as f:
        f[f'fold_{i}'].create_dataset('image', data=images, dtype='f4')
        f[f'fold_{i}'].create_dataset('target', data=target, dtype='f4')
        f[f'fold_{i}'].create_dataset(
            'diagnosis', data=real_diagnosis, dtype='f4')
        f[f'fold_{i}'].create_dataset(
            'patient_idx', data=train_pid[fold], dtype='i4')  # meta data for mapping

with h5py.File(h5_filename, 'r') as f:
    for k in f.keys():
        print(k)
        for ds in f[k].keys():
            print('--', f[k][ds])


images = []
real_diagnosis = val_df.diagnosis.values.copy()
target = real_diagnosis.copy()
target[target > 0] = 1
for _, item in val_df.iterrows():
    year = int(item['year'])
    diagnosis_raw = item['diagnosis_raw']
    filename = item['filename']
    cropped_fn = f'{cropped_folder}/{year}/{diagnosis_raw}/{filename}.npy'
    # add an additional dimension
    img = np.load(cropped_fn)[np.newaxis, ..., np.newaxis]
    # resize with bilinear (default)
    img = tf.image.resize_with_pad(img, resize_shape, resize_shape)
    images.append(img)
images = np.concatenate(images)
with h5py.File(h5_filename, 'a') as f:
    f.create_group('fold_4')
    f['fold_4'].create_dataset('image', data=images, dtype='f4')
    f['fold_4'].create_dataset('target', data=target, dtype='f4')
    f['fold_4'].create_dataset('diagnosis', data=real_diagnosis, dtype='f4')
with h5py.File(h5_filename, 'a') as f:
    f['fold_4'].create_dataset(
        'patient_idx', data=val_df.pid.values, dtype='i4')  # meta data for mapping

images = []
real_diagnosis = test_df.diagnosis.values.copy()
target = real_diagnosis.copy()
target[target > 0] = 1
for _, item in test_df.iterrows():
    year = int(item['year'])
    diagnosis_raw = item['diagnosis_raw']
    filename = item['filename']
    cropped_fn = f'{cropped_folder}/{year}/{diagnosis_raw}/{filename}.npy'
    # add an additional dimension
    img = np.load(cropped_fn)[np.newaxis, ..., np.newaxis]
    # resize with bilinear (default)
    img = tf.image.resize_with_pad(img, resize_shape, resize_shape)
    images.append(img)
images = np.concatenate(images)
with h5py.File(h5_filename, 'a') as f:
    del f['fold_5']
    f.create_group('fold_5')
    f['fold_5'].create_dataset('image', data=images, dtype='f4')
    f['fold_5'].create_dataset('target', data=target, dtype='f4')
    f['fold_5'].create_dataset('diagnosis', data=real_diagnosis, dtype='f4')
    f['fold_5'].create_dataset(
        'patient_idx', data=test_df.pid.values, dtype='i4')  # meta data for mapping
