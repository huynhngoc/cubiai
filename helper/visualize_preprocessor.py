import h5py
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.filters import unsharp_mask
from matplotlib import pyplot as plt


class ChannelRepeater:
    def __init__(self, channel=0, num_repeat=3):
        self.channel = channel
        if '__iter__' not in dir(self.channel):
            self.channel = [channel]
        self.num_repeat = num_repeat

    def transform(self, images, targets):
        new_images = []
        for _ in range(self.num_repeat):
            new_images.append(images[..., self.channel])
        new_images = np.concatenate(new_images, axis=-1)

        return new_images, targets


class EqualizeAdaHist:
    def __init__(self, channel=0):
        self.channel = channel

    def transform(self, images, targets):
        transformed_images = []
        for img in images[..., self.channel]:
            transformed_images.append(equalize_adapthist(img))
        new_images = np.array(images)
        new_images[..., self.channel] = np.array(transformed_images)

        return new_images, targets

class Unsharp:
    def __init__(self, channel=0):
        self.channel = channel

    def transform(self, images, targets):
        transformed_images = []
        for img in images[..., self.channel]:
            transformed_images.append(unsharp_mask(img, radius=5, amount=2))
        new_images = np.array(images)
        new_images[..., self.channel] = np.array(transformed_images)

        return new_images, targets



data_filename = 'P:/CubiAI/preprocess_data/datasets/elbow_abnormal_800.h5'


with h5py.File(data_filename, 'r') as f:
    images = f['fold_0']['image'][:4]
    targets = f['fold_0']['diagnosis'][:4]


cr = ChannelRepeater()
eh = EqualizeAdaHist(1)
unsharp = Unsharp(2)

images, _ = cr.transform(images, targets)
images, _ = eh.transform(images, targets)
images, _ = unsharp.transform(images, targets)

for i in range(4):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i][..., 0], 'gray')
    plt.axis('off')
    plt.subplot(3, 4, 4 + i + 1)
    plt.imshow(images[i][..., 1], 'gray')
    plt.axis('off')
    plt.subplot(3, 4, 8 + i + 1)
    plt.imshow(images[i][..., 2], 'gray')
    plt.axis('off')
plt.show()
