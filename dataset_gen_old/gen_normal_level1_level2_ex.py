import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

# update these settings
n_splits = 4
resize_shape = 32*7

# update these filenames
cropped_folder = '//nmbu.no/LargeFile/Project/CubiAI/preprocess/cropped'
filenames = [
    'csv_detection_info_clean/20_0.csv',
    'csv_detection_info_clean/20_1, artrose.csv',
    'csv_detection_info_clean/20_2, primaerlesjon.csv',
]
h5_filename = '//nmbu.no/LargeFile/Project/CubiAI/preprocess/datasets/test_normal_level1_level2.h5'
# concat all df, remember to reset index
df = pd.concat([pd.read_csv(fn) for fn in filenames]).reset_index()

# choose and adjust target data
# in some cases, diagnosis 2 should become 1
# In some other cases, for ex, multiclass problem with normal 0, level 1 - 3 then diagnosis can be kept as is, the same as this case
# Or if we want to separate level 1-3 then we need to change them into 0-2
# Similarly, if we want to separate normal, level 1 attrose & sklerose, level 2 attrose and primary lesion, level 3 MCD & OCD & UAP,
# we should transform them into correct category
diagnosis = df['diagnosis'].values.copy()

folds = []
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
for train_indice, test_indice in skf.split(df.index, diagnosis):
    np.random.shuffle(test_indice)
    folds.append(test_indice)

# print out to check folds
print('the target values in each fold')
for i, fold in enumerate(folds):
    print(i, diagnosis[fold])

# create the dataset
with h5py.File(h5_filename, 'w') as f:
    for i in range(len(folds)):
        f.create_group(f'fold_{i}')


for i, fold in enumerate(folds):
    images = []
    selected_df = df.iloc[fold]
    target = diagnosis[fold]
    for _, item in selected_df.iterrows():
        year = item['year']
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
        f[f'fold_{i}'].create_dataset('x', data=images, dtype='f4')
        f[f'fold_{i}'].create_dataset('y', data=target, dtype='f4')
        f[f'fold_{i}'].create_dataset(
            'patient_idx', data=fold, dtype='i4')  # meta data for mapping

with h5py.File(h5_filename, 'r') as f:
    for k in f.keys():
        print(k)
        for ds in f[k].keys():
            print('--', f[k][ds])
