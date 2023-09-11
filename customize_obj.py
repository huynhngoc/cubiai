from deoxys.customize import custom_architecture, custom_preprocessor
from deoxys.loaders.architecture import BaseModelLoader
from deoxys.data.preprocessor import BasePreprocessor
from deoxys.utils import deep_copy


from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model

import numpy as np


@custom_architecture
class EfficientNetModelLoader(BaseModelLoader):
    """
    Create a sequential network from list of layers
    """
    map_name = {
        'B0': efficientnet.EfficientNetB0,
        'B1': efficientnet.EfficientNetB1,
        'B2': efficientnet.EfficientNetB2,
        'B3': efficientnet.EfficientNetB3,
        'B4': efficientnet.EfficientNetB4,
        'B5': efficientnet.EfficientNetB5,
        'B6': efficientnet.EfficientNetB6,
        'B7': efficientnet.EfficientNetB7
    }

    def __init__(self, architecture, input_params):
        self._input_params = deep_copy(input_params)
        self.options = architecture

    def load(self):
        """

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network of sequential layers
            from the configured layer list.
        """
        num_class = self.options['num_class']
        pretrained = self.options['pretrained']
        shape = self._input_params['shape']
        efficientNet = self.map_name[self.options['class_name']]

        if num_class <= 2:
            num_class = 1
            activation = 'sigmoid'
        else:
            activation = 'softmax'

        if pretrained:
            model = efficientNet(include_top=False, classes=num_class,
                                 classifier_activation=activation, input_shape=shape, pooling='avg')
            dropout_out = Dropout(0.3)(model.output)
            pred = Dense(num_class, activation=activation)(dropout_out)
            model = Model(model.inputs, pred)
        else:
            model = efficientNet(weights=None, include_top=True, classes=num_class,
                                 classifier_activation=activation, input_shape=shape)

        return model


@custom_preprocessor
class PretrainedEfficientNet(BasePreprocessor):
    def transform(self, images, targets):
        # efficientNet requires input between [0-255]
        images = images * 255
        # pretrain require 3 channel
        new_images = np.concatenate([images, images, images], axis=-1)

        return new_images, targets


@custom_preprocessor
class OneHot(BasePreprocessor):
    def __init__(self, num_class=2):
        if num_class <= 2:
            num_class = 1
        self.num_class = num_class

    def transform(self, images, targets):
        # labels to one-hot encode
        new_targets = np.zeros((len(targets), self.num_class))
        if self.num_class==1:
            new_targets[..., 0] = targets
        else:
            for i in range(self.num_class):
                new_targets[..., i][targets == i] = 1

        return images, new_targets
