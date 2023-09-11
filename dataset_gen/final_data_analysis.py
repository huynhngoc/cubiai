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
cropped_folder = '//nmbu.no/LargeFile/Project/CubiAI/preprocess_data/cropped'
csv_folder = 'P:/CubiAI/preprocess_data/csv_detection_info_clean'
filenames = os.listdir('P:/CubiAI/preprocess_data/csv_detection_info_clean')

h5_filename = '//nmbu.no/LargeFile/Project/CubiAI/preprocess_data/datasets/normal_abnormal_800.h5'
# concat all df, remember to reset index
df = pd.concat([pd.read_csv(csv_folder + '/' + fn)
               for fn in filenames]).reset_index(drop=True)
# fix year for samples before 2022
year = df.year.values
year[df.year.isnull()] = 21
df.year = year

# analysis of data
df.value_counts(df.year)  # 4615 vs 2614
df.value_counts(df.diagnosis)  # 4199 - 1336 - 809 - 808
df[df.year == 21].diagnosis.value_counts()  # 2273 - 1027 - 706 - 609
df[df.year == 22].diagnosis.value_counts()  # 1926 - 309 - 199 - 180
df.value_counts(df.diagnosis_raw)
# 0                               3027
# 0 nye (lagt til 15.03.23)       1172
# 1, artrose og-eller sklerose    1027
# 2, artrose                       624
# 3, MCD                           456
# 1, artrose                       284
# 2, mistanke MCD                  263
# 3, artrose                       263
# 3, UAP                            68
# 1, sklerose                       25
# 3, OCD                            22

df[df.diagnosis_raw == '3, OCD'].year.value_counts()  # 15 vs 6
df[df.diagnosis_raw == '3, UAP'].year.value_counts() # 48 vs 20

# save df
# df.to_csv('csv_detection_info_clean/full_data.csv', index_label='pid')

df = pd.read_csv('csv_detection_info_clean/full_data.csv')

selected_21 = df.year == 21
selected_22 = df.year == 22

# total 7231 rows --> 2200 hold out test
# 5 + 2 = 7 OCD
# 11 + 4 = 15 UAP
# 26 MCD (level 3)
# 23 artrose (level 3)
# 15 MCD (level 2)
# 60 level 2
# 100 level 1

# selecting hold-out test set
random.seed(42)
# number of ocd
ocd_21_num = 5
ocd_22_num = 2
selected_ocd = df.diagnosis_raw == '3, OCD'
ocd_list = random.sample(list(df[selected_ocd & selected_21].index), ocd_21_num) + \
    random.sample(list(df[selected_ocd & selected_21].index), ocd_22_num)
# number of UAP
uap_21_num = 11
uap_22_num = 4
selected_uap = df.diagnosis_raw == '3, UAP'
uap_list = random.sample(list(df[selected_uap & selected_21].index), uap_21_num) + \
    random.sample(list(df[selected_uap & selected_22].index), uap_22_num)
# number of MCD (level 3)
mcd_21_num = 18
mcd_22_num = 8
selected_mcd = df.diagnosis_raw == '3, MCD'
mcd_list = random.sample(list(df[selected_mcd & selected_21].index), mcd_21_num) + \
    random.sample(list(df[selected_mcd & selected_22].index), mcd_22_num)

# number of artrose (level 3)
artrose_21_num = 16
artrose_22_num = 7
selected_artrose = df.diagnosis_raw == '3, artrose'
artrose_list = random.sample(list(df[selected_artrose & selected_21].index), artrose_21_num) + \
    random.sample(list(df[selected_artrose & selected_22].index), artrose_22_num)
# all level 3
level_3 = ocd_list + uap_list + mcd_list + artrose_list

# number of MCD (level 2)
lvl_2_mcd_21_num = 10
lvl_2_mcd_22_num = 5
selected_mcd_lvl_2 = df.diagnosis_raw == '2, mistanke MCD'
lvl_2_mcd_list = random.sample(list(df[selected_mcd_lvl_2 & selected_21].index), lvl_2_mcd_21_num) + \
    random.sample(list(df[selected_mcd_lvl_2 & selected_22].index), lvl_2_mcd_22_num)

# number of artrose (level 2)
lvl_2_artrose_21_num = 45
lvl_2_artrose_22_num = 15
selected_artrose_lvl_2 = df.diagnosis_raw == '2, artrose'
lvl_2_artrose_list = random.sample(list(df[selected_artrose_lvl_2 & selected_21].index), lvl_2_artrose_21_num) + \
    random.sample(list(df[selected_artrose_lvl_2 & selected_22].index), lvl_2_artrose_22_num)

# all level 2
level_2 = lvl_2_mcd_list + lvl_2_artrose_list


# level 1
lvl_1_21_num = 80
lvl_1_22_num = 20
selected_lvl_1 = df.diagnosis == 1
level_1 = random.sample(list(df[selected_lvl_1 & selected_21].index), lvl_1_21_num) + \
    random.sample(list(df[selected_lvl_1 & selected_22].index), lvl_1_22_num)

# normal
lvl_0_21_num = 1383
lvl_0_22_num = 600
selected_lvl_0 = df.diagnosis == 0
level_0 = random.sample(list(df[selected_lvl_0 & selected_21].index), lvl_0_21_num) + \
    random.sample(list(df[selected_lvl_0 & selected_22].index), lvl_0_22_num)

assert len(level_0 + level_1 + level_2 + level_3) == 2229
assert np.all(df[df.pid.isin(level_0)].diagnosis == 0)
assert np.all(df[df.pid.isin(level_1)].diagnosis == 1)
assert np.all(df[df.pid.isin(level_2)].diagnosis == 2)
assert np.all(df[df.pid.isin(level_3)].diagnosis == 3)

holdout_index = level_0 + level_1 + level_2 + level_3
# df[df.pid.isin(holdout_index)].to_csv('csv_detection_info_clean/test_data.csv', index=False)

test_df = pd.read_csv('csv_detection_info_clean/test_data.csv')

test_df.diagnosis_raw.value_counts()
# 0                               1270
# 0 nye (lagt til 15.03.23)        713
# 1, artrose og-eller sklerose      80
# 2, artrose                        60
# 3, MCD                            26
# 3, artrose                        23
# 1, artrose                        20
# 2, mistanke MCD                   15
# 3, UAP                            15
# 3, OCD                             7

train_df = df[~df.pid.isin(test_df.pid)]

train_df.diagnosis_raw.value_counts()
# 0                               1757
# 1, artrose og-eller sklerose     947
# 2, artrose                       563
# 0 nye (lagt til 15.03.23)        459
# 3, MCD                           430
# 1, artrose                       264
# 2, mistanke MCD                  248
# 3, artrose                       240
# 3, UAP                            53
# 1, sklerose                       25
# 3, OCD                            14

train_df.diagnosis.value_counts() # 2216 - 1236 - 811 - 737


# val set
# level 3 OCD 3
# level 3 UAP 10
# level 3 MCD 90
# level 3 artrose 50
# level 2 MCD 50
# level 2 others 100
# level 1 remaining

random.seed(42)
val_selected_ocd = list(train_df[train_df.diagnosis_raw == '3, OCD'].pid)
val_selected_uap = list(train_df[train_df.diagnosis_raw == '3, UAP'].pid)
val_selected_lvl3_mcd = list(train_df[train_df.diagnosis_raw == '3, MCD'].pid)
val_selected_lvl3_artrose = list(train_df[train_df.diagnosis_raw == '3, artrose'].pid)
val_selected_lvl2_mcd = list(train_df[train_df.diagnosis_raw == '2, mistanke MCD'].pid)
val_selected_lvl2_artrose = list(train_df[train_df.diagnosis_raw == '2, artrose'].pid)
val_selected_lvl1 = list(train_df[train_df.diagnosis == 1].pid)
val_selected_lvl0 = list(train_df[train_df.diagnosis == 0].pid)

val_list = random.sample(val_selected_ocd, 3) + \
           random.sample(val_selected_uap, 10) + \
           random.sample(val_selected_lvl3_mcd, 90) + \
           random.sample(val_selected_lvl3_artrose, 50) + \
           random.sample(val_selected_lvl2_mcd, 50) + \
           random.sample(val_selected_lvl2_artrose, 100) + \
           random.sample(val_selected_lvl1, 197) + \
           random.sample(val_selected_lvl0, 500)

val_df = train_df[train_df.pid.isin(val_list)]
# val_df.to_csv('csv_detection_info_clean/val_data.csv', index=False)
val_df = pd.read_csv('csv_detection_info_clean/val_data.csv')
val_df.diagnosis_raw.value_counts()


actual_train_df = train_df[~train_df.pid.isin(val_df.pid)]
#actual_train_df.to_csv('csv_detection_info_clean/train_actual_data.csv', index=False)
actual_train_df = pd.read_csv('csv_detection_info_clean/train_actual_data.csv')

actual_train_df.diagnosis_raw.value_counts()
