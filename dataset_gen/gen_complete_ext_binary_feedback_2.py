import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf


# update these settings
resize_shape = 800  # 224 - 320 - 640 - 800 - 1280

# update these filenames
cropped_folder = '//nmbu.no/Research/Project/CubiAI/preprocess/cropped'
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
h5_filename = '//nmbu.no/Research/Project/CubiAI/preprocess/datasets/800_complete_ext_binary_feedback_2.h5'
rs=12
# For at indeksene fortsatt skal stemme med normal_abnormal_2 må 0 nye lastes inn i eget datasett og legges til etterpå.

ds_new = pd.read_csv('csv_detection_info_clean/0 nye.csv')
#ds_new.diagnosis_raw = '0 nye'

# concat all df, remember to reset index
ds = pd.concat([pd.read_csv(fn) for fn in filenames]).reset_index(drop=True)


diagnoses=['1, artrose og-eller sklerose', '2, artrose', '2, mistanke MCD',
           '3, artrose', '3, MCD', '3, OCD', '3, UAP'] # Don't have to add 0, since it is already registered as an integer in the diagnosis_raw column

for i,d in enumerate(diagnoses): # Change all diagnoses to numbers from 1 through 7 (Not normal samples)
    ds.diagnosis.iloc[np.where(ds['diagnosis_raw']==d)] = i+1

np.unique(ds['diagnosis']) # Want all diagnoses from 0 through 7

df = ds[ds['diagnosis']==0].sample(n=500, random_state=124, axis = 0, replace=False, ignore_index=False)
print(0, sum(df['diagnosis']==0))
for d in range(7):
    if d+1 == 6: # 25% of OCD is only 4, which is too little
        new = ds[ds['diagnosis']==d+1].sample(n=8, random_state=12, axis = 0, replace=False)
    else:
        new = ds[ds['diagnosis']==d+1].sample(frac=0.25, random_state=12, axis = 0, replace=False)
    df = pd.concat([df, new], ignore_index=False)
    print(d+1, sum(df['diagnosis']==d+1)) # print number of samples in each class


# Run the code above to see if the dataset will be balanced at first
df = ds.drop(df.index).reset_index(drop=True)
df = pd.concat([df,ds_new],ignore_index=True)

#Dropping the indeces that were used in the feedback training
df=df.drop([ 699, 895, 3143, 1759, 695, 646, 2097, 2834, 804, 1249, 2628, 2916, 1229, 1769, 1825, 3034, 2142, 1804, 1830, 1021, 2613, 1159, 496, 1072, 1794, 2594, 1878, 13, 1746, 3089, 2282, 575, 3042, 2274, 649, 1078, 1807, 1253, 1512, 1871, 294, 1821, 925, 3431, 1329, 1354, 2267, 2739, 2832, 1805, 682, 2941, 12, 820, 717, 3337, 1400, 1444, 3502, 3242, 2725, 1275, 267, 3018, 1073, 3393, 1839, 738, 1813, 3513, 2147, 1118, 2581, 2172, 618, 826, 1773, 227, 1897, 1819, 707, 2843, 1197, 1778, 2961, 1744, 548, 3239, 2135, 1779, 794, 1170, 1460, 1259, 617, 1629, 1820, 704, 3455, 700, 597, 2885, 3135, 1711, 5, 2568, 3471, 3362, 1614, 3375, 615, 1864, 1860, 2502, 2003, 534, 3411, 2245, 1838, 2902, 3327, 2938, 206, 185, 3380, 3142, 683, 2470, 2598, 1750, 948, 2999, 2919, 1826, 709, 1720, 2979, 3054, 1077, 3247, 1001, 1840, 3328, 841]).reset_index(drop=True)

print('number of samples from each diagnosis')
for d in range(8):
    print(d, sum(df['diagnosis']==d))


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
diagnosis[diagnosis == 1] = 1
diagnosis[diagnosis == 2] = 1
diagnosis[diagnosis == 3] = 1
diagnosis[diagnosis == 4] = 1
diagnosis[diagnosis == 5] = 1
diagnosis[diagnosis == 6] = 1
diagnosis[diagnosis == 7] = 1
diagnosis[diagnosis == 8] = 1

print('abnormals',sum(diagnosis==1))
print('normals',sum(diagnosis==0))

rng = np.random.default_rng(seed=rs)


n_splits = 4
folds = []
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
for train_indice, test_indice in skf.split(df.index, diagnosis):
    rng.shuffle(test_indice)
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
