import os
import json
import re
import zipfile
import numpy as np
import pandas as pd
import SimpleITK as sitk

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt


val_idx = [4,  14,  26,  44,  45,  48,  54,  60,  73,  85,  88,  93,  95,
           98, 102, 108, 125, 148, 154, 157, 167, 172, 208, 216]
train_idx = [i for i in range(225) if i not in val_idx]
base_folder = '//nmbu.no/LargeFile/Project/CubiAI/sortering/dicom sortert level 2/21/0 god kvalitet'
df = pd.read_csv('info.csv')


def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.
    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0,
            boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


def read_image(filename):
    min_size = 2**5
    image = sitk.ReadImage(filename)
    img = sitk.GetArrayFromImage(image).astype('float32')
    imin, imax = img.min(), img.max()
    img -= imin
    img /= (imax - imin)
    # img *= 255
    img = img[0]
    w, h = img.shape
    w_pad = w % min_size
    if w_pad > 0:
        w_pad = min_size - w_pad
    h_pad = h % min_size
    if h_pad > 0:
        h_pad = min_size - h_pad
    img = np.pad(img, [(0, w_pad), (0, h_pad)], constant_values=0.0)
    return img[..., np.newaxis]


def train_generator():
    while True:
        selected_idx = np.random.choice(train_idx)
        data = df.loc[selected_idx, :]
        img = read_image(base_folder + '/' + data.filename)
        if img.shape[0] > 2500 or img.shape[1] > 2500:
            continue
        bboxes = np.array([data.ax0_min, data.ax1_min,
                           data.ax0_max, data.ax1_max], dtype='float32')
        print(img.shape, bboxes)
        if np.random.random() > 0.5:
            print('flip left')
            img = np.flip(img, axis=0)
            bboxes[0], bboxes[2] = (img.shape[0] - bboxes[2] - 1,
                                    img.shape[0] - bboxes[0] - 1)
        if np.random.random() > 0.5:
            print('flip down')
            img = np.flip(img, axis=1)
            bboxes[1], bboxes[3] = (img.shape[1] - bboxes[3] - 1,
                                    img.shape[1] - bboxes[1] - 1)
        if np.random.random() < 0.2:
            print('transpose')
            img = np.transpose(img, [1, 0, 2])
            bboxes[0], bboxes[1] = bboxes[1], bboxes[0]
            bboxes[2], bboxes[3] = bboxes[3], bboxes[2]
        # ax1 is x, ax0 is y
        bboxes[0], bboxes[1] = bboxes[1], bboxes[0]
        bboxes[2], bboxes[3] = bboxes[3], bboxes[2]
        bboxes = bboxes[np.newaxis, np.newaxis, ...]
        print(img.shape, bboxes)
        bboxes = convert_to_xywh(bboxes)
        yield img, bboxes


gen = train_generator()
img, bbox = next(gen)
bbox = convert_to_corners(bbox)
plt.imshow(img, 'gray')
ax = plt.gca()
x1, y1, x2, y2 = bbox.numpy()[0, 0]
w, h = x2 - x1, y2 - y1
patch = plt.Rectangle(
    [x1, y1], w, h, fill=False, edgecolor='red', linewidth=1.
)
# patch = plt.Rectangle(
#     [y1, x1], h, w, fill=False, edgecolor='yellow', linewidth=1.
# )
ax.add_patch(patch)
plt.show()
