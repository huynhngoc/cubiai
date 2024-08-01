import numpy as np
import h5py
import pandas as pd
import tensorflow as tf
import gc
from matplotlib import pyplot as plt
from deoxys.model import load_model, model_from_full_config
from deoxys.utils import load_json_config
import sys
import os
from pathlib import Path

# Current file's directory
current_path = Path(__file__).resolve().parent
print(f"Current Path: {current_path}")  # Debugging: Print current path

# Parent directory (one level up)
parent_path = current_path.parent
print(f"Parent Path: {parent_path}")  # Debugging: Print parent path

# Append parent directory to sys.path if not already included
if str(parent_path) not in sys.path:
    sys.path.append(str(parent_path))
    print(f"Updated sys.path: {sys.path}")  # Debugging: Print updated sys.path

# Attempt to import customize_obj
import customize_obj


test_res = pd.read_csv('S:/Master/data/vargrad/prediction_test.csv')
selected = pd.read_csv('S:/Master/data/vargrad/prediction_test.csv')

df = pd.merge(selected, test_res, 'inner', 'patient_idx')

images = []
with h5py.File('S:/Master/data/elbow_abnormal_800.h5', 'r') as f:
    pids = f['fold_5']['patient_idx'][:]
    for pid in df.patient_idx:
        indice = np.argwhere(pids == pid)[0]
        images.append(f['fold_5']['image'][indice])

images = np.concatenate(images)
#np.save('../selected.npy', images)

images = np.load('../selected.npy')
results_location = 'S:/Master/data/vargrad/'
best_epochs = []
model_name = 'b3_0005_categorical_onehot_level'
with open(results_location + f'{model_name}/info.txt', 'r') as f:
    best_epochs.append(int(f.readline()[-25:-22]))
#for i in range(1, 5):
#    model_name = f'b{i}_0005_categorical_onehot_level'
#    with open(results_location + f'{model_name}/info.txt', 'r') as f:
#        best_epochs.append(int(f.readline()[-25:-22]))

all_var_grads = []
all_smooth_square_grads = []
config = load_json_config(f'config/pretrain/b3_0005_categorical_onehot_level.json')
config['dataset_params']['config']['batch_size'] = 2
config['dataset_params']['config']['batch_cache'] = 1
config['dataset_params']['config']['filename'] = 'S:/Master/data/elbow_abnormal_800.h5'

best_model = model_from_full_config(
    config, results_location + f'b3_0005_categorical_onehot_level/model/model.{best_epochs[0]:03d}.h5')

#for i in range(1, 5):
#    print('B', i)
#    config = load_json_config(f'config/pretrain/b{i}_0005_categorical_onehot_level.json')
#    config['dataset_params']['config']['batch_size'] = 2
#    config['dataset_params']['config']['batch_cache'] = 1
#    config['dataset_params']['config']['filename'] = 'S:/Master/data/elbow_abnormal_800.h5'

#    best_model = model_from_full_config(
#        config, results_location + f'b{i}_0005_categorical_onehot_level/model/model.{best_epochs[i-1]:03d}.h5')

model = best_model.model
    # print(model.summary())
dr = best_model.data_reader

tf_dtype = model.inputs[0].dtype
print('TF dtype', tf_dtype)

#final_var_grads = []
#final_smooth_grads = []
#final_smooth_square_grads = []
#for batch_x in images.copy():
#    for pp in dr.preprocessors:
#        x = pp.transform(np.array([batch_x]), np.array([0]))[0]
#    np_random_gen = np.random.default_rng(1123)
#    new_shape = list(x.shape) + [20]
#    var_grad = np.zeros(new_shape)
#    for trial in range(20):
#        print(f'Trial {trial+1}/20')
#        noise = (np_random_gen.normal(
#            loc=0.0, scale=.05, size=x.shape[:-1]) * 255)
#        x_noised = x + np.stack([noise]*3, axis=-1)
#        x_noised = tf.Variable(x_noised, dtype=tf_dtype)
#        with tf.GradientTape() as tape:
#            tape.watch(x_noised)
#            pred = model(x_noised)
#        grads = tape.gradient(pred, x_noised).numpy()
#        var_grad[..., trial] = grads

#   final_var_grad = var_grad.std(axis=-1)**2
#    final_smooth_grad = var_grad.mean(axis=-1)
#    final_smooth_square_grad = (var_grad ** 2).mean(axis=-1)
#    gc.collect()
#    final_var_grads.append(final_var_grad)
    # final_smooth_grads.append(final_smooth_grad)
#    final_smooth_square_grads.append(final_smooth_square_grad)
#all_var_grads.append(final_var_grads)
# all_smooth_grads.append(final_smooth_grads)
#all_smooth_square_grads.append(final_smooth_square_grads)

#all_var_grads = np.array([np.concatenate(g) for g in all_var_grads])
#all_smooth_square_grads = np.array(
#    [np.concatenate(g) for g in all_smooth_square_grads])
#np.save('../selected_var_grad_v2.npy', all_var_grads)
#np.save('../selected_smooth_grad_square_v2.npy', all_smooth_square_grads)


all_var_grads = np.load('../selected_var_grad_v2.npy')



preds = df[[f'b{i}' for i in range(1, 5)]]
entropies = (-np.log(preds) * preds).values
true_diagnoses = df.diagnosis.values
raw_diagnoses = df.diagnosis_raw.values
sum_preds = (preds.values > 0.5).astype(float).sum(axis=1)
comment = df.comment
pid = df.pid
option = df['type']
# selected_images = images[selected_indice]
selected_vargrad = all_var_grads
# selected_smooth = all_smooth_square_grads

for i in range(len(images)):
    img = images[i]
    pred = preds.values[i]
    avg_pred = pred.mean()
    entropy = entropies[i]
    avg_entropy = entropy.mean()
    true_diagnosis = true_diagnoses[i]
    raw_diagnosis = raw_diagnoses[i]
    sum_pred = sum_preds[i]
    all_agree = (sum_pred == 0) or (sum_pred == 4)
    pred_text = 'Abnormal' if avg_pred > 0.5 else 'Normal'
    pred_is_correct = int(avg_pred > 0.5) == int(true_diagnosis > 0)
    high_entropy = avg_entropy > 0.08

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(2, 3, 1)
    ax.imshow(img, 'gray')
    ax.axis('off')
    ax.set_title(f'Patient: {pid[i]}, {option[i]}')

    ax = plt.subplot(2, 3, 4)
    ax.axis([0, 5, 0, 10])
    pos_y = 9.5
    ax.text(0, pos_y, 'True diagnosis:', fontsize=13)
    ax.text(2, pos_y, raw_diagnosis, fontsize=10)

    pos_y -= 0.9
    ax.text(0, pos_y, 'Ensemble predicted:', fontsize=13)
    if pred_is_correct:
        ax.text(3, pos_y, pred_text, color='green', fontsize=13)
    else:
        ax.text(3, pos_y, pred_text, color='red', fontsize=13)

    pos_y -= 0.9
    ax.text(0, pos_y, 'High avg entropy:', fontsize=13)
    if pred_is_correct and high_entropy:
        ax.text(3, pos_y, 'Yes (Suspected)', color='orange', fontsize=13)
    elif pred_is_correct and not high_entropy:
        ax.text(3, pos_y, 'No', color='green', fontsize=13)
    elif not pred_is_correct and not high_entropy:
        ax.text(3, pos_y, 'No (QA failed)', color='red', fontsize=13)
    else:
        ax.text(3, pos_y, 'Yes (Suspected)', color='green', fontsize=13)

    pos_y -= 0.9
    ax.text(0, pos_y, 'All agree:', fontsize=13)
    if pred_is_correct and all_agree:
        ax.text(3, pos_y, 'Yes', color='green', fontsize=13)
    elif pred_is_correct and not all_agree:
        ax.text(3, pos_y, 'No (Suspected)', color='orange', fontsize=13)
    elif not pred_is_correct and all_agree:
        ax.text(3, pos_y, 'Yes (QA failed)', color='red', fontsize=13)
    else:
        ax.text(3, pos_y, 'No (Suspected)', color='green', fontsize=13)

    pos_y -= 0.9
    ax.text(0, pos_y, 'Results by models:', fontsize=13)
    pos_y -= 0.9
    ax.text(0, pos_y, 'Model', fontsize=13)
    ax.text(1, pos_y, 'Predicted', fontsize=13)
    ax.text(3, pos_y, 'High entropy', fontsize=13)
    for j in range(4):
        pos_y -= 0.9
        ax.text(0, pos_y, f'B{j+1}', fontsize=13)
        pred_correct = int(pred[j] > 0.5) == int(true_diagnosis > 0)
        ax.text(1, pos_y, 'Abnormal' if pred[j] > 0.5 else 'Normal',
                color='green' if pred_correct else 'red', fontsize=13)
        if entropy[j] > 0.08:
            ax.text(3, pos_y, 'Yes (Suspected)',
                    color='orange' if pred_correct else 'green', fontsize=13)
        else:
            if pred_correct:
                ax.text(3, pos_y, 'No', color='green', fontsize=13)
            else:
                ax.text(3, pos_y, 'No (QA failed)', color='red', fontsize=13)
    pos_y -= 0.9
    ax.text(0, pos_y, f'Comment {comment[i]}', fontsize=8)
    ax.axis('off')
    for j in range(4):
        if j < 2:
            ax = plt.subplot(2, 3, j+2)
        else:
            ax = plt.subplot(2, 3, j+3)
        explain_map = selected_vargrad[j][i].mean(axis=-1).copy()
        vmax = np.quantile(explain_map, 0.99)
        vmin = np.quantile(explain_map, 0.)  # explain_map.min()
        thres = np.quantile(explain_map, 0.7)
        explain_map[explain_map < thres] = np.nan
        ax.axis('off')
        ax.imshow(img[..., 0], 'gray')
        ax.imshow(explain_map, 'Reds', alpha=0.5, vmin=vmin, vmax=vmax)
        ax.set_title(f'B{j+1}')

    plt.tight_layout()
    plt.show()
    print('Finish', i)


for i in range(len(images)):
    if pid[i] != 3847:
        continue
    img = images[i]
    pred = preds.values[i]
    avg_pred = pred.mean()
    entropy = entropies[i]
    avg_entropy = entropy.mean()
    true_diagnosis = true_diagnoses[i]
    raw_diagnosis = raw_diagnoses[i]
    sum_pred = sum_preds[i]
    all_agree = (sum_pred == 0) or (sum_pred == 4)
    pred_text = 'Abnormal' if avg_pred > 0.5 else 'Normal'
    pred_is_correct = int(avg_pred > 0.5) == int(true_diagnosis > 0)
    high_entropy = avg_entropy > 0.08
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.imshow(img, 'gray')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    for j in range(4):
        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()
        explain_map = selected_vargrad[j][i].mean(axis=-1).copy()
        vmax = np.quantile(explain_map, 0.99)
        vmin = np.quantile(explain_map, 0.)  # explain_map.min()
        thres = np.quantile(explain_map, 0.7)
        explain_map[explain_map < thres] = np.nan
        ax.axis('off')
        ax.imshow(img[..., 0], 'gray')
        ax.imshow(explain_map, 'Reds', alpha=0.5, vmin=vmin, vmax=vmax)
        # ax.set_title(f'B{j+1}')
        plt.tight_layout()
        plt.show()


# for i in range(len(images)):
#     # if pid[i] != 3847:
#     #     continue
#     img = images[i]
#     pred = preds.values[i]
#     avg_pred = pred.mean()
#     entropy = entropies[i]
#     avg_entropy = entropy.mean()
#     true_diagnosis = true_diagnoses[i]
#     raw_diagnosis = raw_diagnoses[i]
#     sum_pred = sum_preds[i]
#     all_agree = (sum_pred == 0) or (sum_pred == 4)
#     pred_text = 'Abnormal' if avg_pred > 0.5 else 'Normal'
#     pred_is_correct = int(avg_pred > 0.5) == int(true_diagnosis > 0)
#     high_entropy = avg_entropy > 0.08

#     for j in range(4):
#         explain_map = selected_vargrad[j][i].mean(axis=-1).copy()
#         vmax = np.quantile(explain_map, 0.99)
#         vmin = np.quantile(explain_map, 0.)  # explain_map.min()
#         thres = np.quantile(explain_map, 0.7)

#         print(f'B{j+1}:', (thres - vmin)/(vmax-vmin), vmin, vmax, thres)

#     print('Finish', i)


plt.colorbar()
