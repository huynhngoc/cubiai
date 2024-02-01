import h5py
import numpy as np
import gc

h5_filename = 'P:/CubiAI/preprocess_data/datasets/elbow_normal_abnormal_800.h5'

# check number of patients in each level
with h5py.File(h5_filename, 'r') as f:
    for k in f.keys():
        print(k)
        for ds in f[k].keys():
            print('--', f[k][ds])
            if ds == 'target':
                unique, counts = np.unique(f[k][ds], return_counts=True)
                print(np.asarray((unique, counts)).T)

with h5py.File(h5_filename, 'r') as f:
    for k in f.keys():
        print(k)
        for ds in f[k].keys():
            print('--', f[k][ds])
            if ds == 'diagnosis':
                unique, counts = np.unique(f[k][ds], return_counts=True)
                print(np.asarray((unique, counts)).T)


new_h5_filename = 'P:/CubiAI/preprocess_data/datasets/elbow_abnormal_800.h5'
with h5py.File(new_h5_filename, 'w') as f:
    for i in range(6):
        f.create_group(f'fold_{i}')

for i in range(6):
    with h5py.File(h5_filename, 'r') as f:
        print('reading image')
        image = f[f'fold_{i}']['image'][:]
    with h5py.File(h5_filename, 'r') as f:
        print('reading target')
        target = f[f'fold_{i}']['target'][:]
        print('reading patient_idx')
        pid = f[f'fold_{i}']['patient_idx'][:]
        print('reading diagnosis')
        diagnosis = f[f'fold_{i}']['diagnosis'][:]
        print('done reading')
    selected_index = target > 0
    print('selected', selected_index.sum())
    new_image = image[selected_index].copy()
    new_pid = pid[selected_index].copy()
    new_diagnosis = diagnosis[selected_index].copy()
    level = diagnosis[selected_index].copy() - 1
    # save to new h5file
    with h5py.File(new_h5_filename, 'a') as f:
        f[f'fold_{i}'].create_dataset('image', data=new_image, dtype='f4')
        f[f'fold_{i}'].create_dataset('patient_idx', data=new_pid, dtype='f4')
        f[f'fold_{i}'].create_dataset(
            'diagnosis', data=new_diagnosis, dtype='f4')
        f[f'fold_{i}'].create_dataset('level', data=level, dtype='f4')


with h5py.File(new_h5_filename, 'r') as f:
    for k in f.keys():
        print(k)
        for ds in f[k].keys():
            print('--', f[k][ds])
            if ds == 'diagnosis' or ds == 'level':
                unique, counts = np.unique(f[k][ds], return_counts=True)
                print(np.asarray((unique, counts)).T)
