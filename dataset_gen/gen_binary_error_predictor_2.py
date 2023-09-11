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
h5_filename = '//nmbu.no/Research/Project/CubiAI/preprocess/datasets/800_binary_error_predictor_2.h5'
rs=12
# concat all df, remember to reset index
ds = pd.concat([pd.read_csv(fn) for fn in filenames]).reset_index(drop=True)
# TEST TO GET ONLY 25% OF THE TOTAL SAMPLES IN THE DATASET

diagnoses=['1, artrose og-eller sklerose', '2, artrose', '2, mistanke MCD',
           '3, artrose', '3, MCD', '3, OCD', '3, UAP'] # Don't have to add 0, since it is already registered as an integer in the diagnosis_raw column

for i,d in enumerate(diagnoses): # Change all diagnoses to numbers from 1 through 7 (Not normal samples)
    ds.diagnosis.iloc[np.where(ds['diagnosis_raw']==d)] = i+1
df = ds[ds['diagnosis']==0].sample(n=500, random_state=124, axis = 0, replace=False, ignore_index=False)

print(0, sum(df['diagnosis']==0))
for d in range(7):
    if d+1 == 6: # 25% of OCD is only 4, which is too little
        new = ds[ds['diagnosis']==d+1].sample(n=8, random_state=12, axis = 0, replace=False)
    else:
        new = ds[ds['diagnosis']==d+1].sample(frac=0.25, random_state=12, axis = 0, replace=False)
    df = pd.concat([df, new], ignore_index=False)
    print(d+1, sum(df['diagnosis']==d+1)) # print number of samples in each class
df = ds.drop(df.index).reset_index(drop=True)
print('number of samples from each diagnosis')
for d in range(8):
    print(d, sum(df['diagnosis']==d))

# From here diagnosis is 1 if the sample was wrongly predicted
# indeces are collected from check_results_binary notebook
df.diagnosis=0
df.loc[[  2,   12,   17,   50,   65,   75,   76,   78,   86,   89,  128,
        163,  178,  185,  189,  190,  213,  225,  238,  253,  254,  267,
        279,  280,  309,  317,  318,  352,  362,  393,  398,  419,  424,
        429,  458,  460,  463,  467,  468,  486,  534,  535,  538,  548,
        549,  559,  600,  601,  607,  615,  618,  629,  638,  639,  643,
        645,  646,  651,  658,  664,  666,  667,  681,  682,  683,  684,
        685,  686,  689,  690,  695,  697,  699,  700,  707,  717,  721,
        723,  738,  745,  747,  749,  756,  767,  772,  788,  796,  805,
        811,  813,  820,  826,  834,  841,  867,  868,  871,  876,  888,
        894,  895,  925,  930,  936,  948,  955,  970, 1021, 1025, 1038,
       1040, 1073, 1096, 1103, 1114, 1170, 1172, 1223, 1229, 1237, 1249,
       1254, 1259, 1267, 1286, 1294, 1308, 1316, 1319, 1323, 1329, 1330,
       1338, 1348, 1350, 1351, 1353, 1354, 1364, 1370, 1396, 1400, 1439,
       1444, 1460, 1461, 1510, 1542, 1583, 1584, 1597, 1600, 1614, 1629,
       1638, 1650, 1744, 1746, 1747, 1748, 1749, 1750, 1759, 1761, 1771,
       1772, 1776, 1782, 1783, 1786, 1788, 1791, 1792, 1794, 1796, 1805,
       1806, 1811, 1820, 1821, 1829, 1833, 1836, 1837, 1839, 1840, 1847,
       1858, 1860, 1864, 1872, 1876, 1888, 2070, 2077, 2078, 2135, 2147,
       2204, 2213, 2214, 2245, 2255, 2274, 2282],'diagnosis'] = 1

n_drop=len(df)-2*len(df[df.diagnosis==1])
df = df.drop(df[df.diagnosis == 0].sample(n=n_drop, random_state=rs, axis = 0, replace=False).index).reset_index()

# choose and adjust target data
# in this case, diagnosis 2 should become 1
# In some other cases, for ex, multiclass problem with normal 0, level 1 - 3 then diagnosis can be kept as is
# Or if we want to separate level 1-3 then we need to change them into 0-2
# Similarly, if we want to separate normal, level 1 artrose & sklerose, level 2 artrose and primary lesion, level 3 MCD & OCD & UAP,
# we should transform them into correct category
diagnosis = df['diagnosis'].values.copy()
d_raw = diagnosis.copy() # will be the diagnosis_raw in the .h5-file

sum(diagnosis==1)
sum(diagnosis==0)

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
