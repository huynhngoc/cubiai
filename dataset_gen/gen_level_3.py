import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf


# update these settings
resize_shape = 800  # 224 - 320 - 640 - 800 - 1280

# update these filenames
cropped_folder = '//nmbu.no/LargeFile/Project/CubiAI/preprocess/cropped'
filenames = [
    'csv_detection_info_clean/0.csv',
    'csv_detection_info_clean/1, artrose og-eller sklerose.csv',
    'csv_detection_info_clean/2, artrose.csv',
    'csv_detection_info_clean/2, mistanke MCD.csv',
    'csv_detection_info_clean/3, artrose.csv',
    'csv_detection_info_clean/3, MCD.csv',
    'csv_detection_info_clean/3, OCD.csv',
    'csv_detection_info_clean/3, UAP.csv'
]
# REMEMBER TO UPDATE THE DATASET NAME
h5_filename = '//nmbu.no/LargeFile/Project/CubiAI/preprocess/datasets/800_level_3.h5'

# concat all df, remember to reset index
ds = pd.concat([pd.read_csv(fn) for fn in filenames]).reset_index()
# TEST TO GET ONLY 25% OF THE TOTAL SAMPLES IN THE DATASET

diagnoses=['1, artrose og-eller sklerose', '2, artrose', '2, mistanke MCD',
           '3, artrose', '3, MCD', '3, OCD', '3, UAP'] # Don't have to add 0, since it is already registered as an integer in the diagnosis_raw column

for i,d in enumerate(diagnoses): # Change all diagnoses to numbers from 1 through 7 (Not normal samples)
    ds.diagnosis.iloc[np.where(ds['diagnosis_raw']==d)] = i+1

df = ds[ds['diagnosis']==6].sample(n=8, random_state=12, axis = 0, replace=False) # 25% of OCD is only 4, which is too little
for d in range(7):
    if d+1 == 6: 
        print(d+1, sum(df['diagnosis']==d+1))
        continue
    else:
        new = ds[ds['diagnosis']==d+1].sample(frac=0.35, random_state=12, axis = 0, replace=False)
        df = pd.concat([df, new], ignore_index=True)
    print(d+1, sum(df['diagnosis']==d+1)) # print number of samples in each class

# Run the code above to see if the dataset will be balanced at first

# choose and adjust target data
# in this case, diagnosis 2 should become 1
# In some other cases, for ex, multiclass problem with normal 0, level 1 - 3 then diagnosis can be kept as is
# Or if we want to separate level 1-3 then we need to change them into 0-2
# Similarly, if we want to separate normal, level 1 artrose & sklerose, level 2 artrose and primary lesion, level 3 MCD & OCD & UAP,
# we should transform them into correct category
diagnosis = df['diagnosis'].values.copy()
d_raw = diagnosis.copy() # will be the diagnosis_raw in the .h5-file

# CHANGE TARGET IN FOLLOWING LINES
# important to change in increasing order, so that we dont mix targets.
diagnosis[diagnosis == 1] = 0
diagnosis[diagnosis == 2] = 1
diagnosis[diagnosis == 3] = 1
diagnosis[diagnosis == 4] = 2
diagnosis[diagnosis == 5] = 2
diagnosis[diagnosis == 6] = 2
diagnosis[diagnosis == 7] = 2

sum(diagnosis==1)
sum(diagnosis==0)
sum(diagnosis==2)

n_splits = 4
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
        diagnosis_raw = item['diagnosis_raw']
        filename = item['filename']
        cropped_fn = f'{cropped_folder}/{diagnosis_raw}/{filename}.npy'
        # add an additional dimension
        img = np.load(cropped_fn)[np.newaxis, ..., np.newaxis]
        # resize with bilinear (default)
        img = tf.image.resize_with_pad(img, resize_shape, resize_shape)
        images.append(img)
    images = np.concatenate(images)
    with h5py.File(h5_filename, 'a') as f:
        f[f'fold_{i}'].create_dataset('x', data=images, dtype='f4')
        f[f'fold_{i}'].create_dataset('y', data=target, dtype='f4')
        f[f'fold_{i}'].create_dataset('diagnosis', data=d_raw[fold], dtype='f4') # Want the number corresponding to diagnosis in the dataset
        f[f'fold_{i}'].create_dataset(
            'patient_idx', data=fold, dtype='i4')  # meta data for mapping

with h5py.File(h5_filename, 'r') as f:
    for k in f.keys():
        print(k)
        for ds in f[k].keys():
            print('--', f[k][ds])
