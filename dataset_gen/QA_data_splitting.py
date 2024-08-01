"""
Check for duplication and overlapping in the train, val, and test data
"""


import pandas as pd

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

df[df.duplicated(subset='filename', keep=False)].sort_values('pid')
# Empty DataFrame
# Columns: [pid, base_path, filename, ax0_min, ax0_max, ax1_min, ax1_max, confidence, diagnosis, diagnosis_raw, year]
# Index: []

df.shape
# (7229, 11)
df.filename.nunique()
# 7229
