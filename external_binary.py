"""
Run the experiments pipeline
"""

import customize_obj
# import h5py
# from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from deoxys.experiment import DefaultExperimentPipeline
# from deoxys.model.callbacks import PredictionCheckpoint
# from deoxys.utils import read_file
import argparse
import os
import shutil
from deoxys.utils import read_csv
# import numpy as np
# from pathlib import Path
# from comet_ml import Experiment as CometEx
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef


class Matthews_corrcoef_scorer:
    def __call__(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)

    def _score_func(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)


metrics.SCORERS['mcc'] = Matthews_corrcoef_scorer()


def metric_avg_score(res_df, postprocessor):
    res_df['avg_score'] = res_df[['AUC', 'roc_auc', 'f1', 'f1_0',
                                  'BinaryAccuracy', 'mcc']].mean(axis=1)

    return res_df


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("log_folder")
    parser.add_argument("--best_epoch", default=0, type=int)
    parser.add_argument("--temp_folder", default='', type=str)
    parser.add_argument("--meta", default='patient_idx', type=str)
    parser.add_argument(
        "--monitor", default='avg_score', type=str)
    parser.add_argument(
        "--monitor_mode", default='max', type=str)
    parser.add_argument("--memory_limit", default=0, type=int)

    args, unknown = parser.parse_known_args()

    if args.memory_limit:
        # Restrict TensorFlow to only allocate X-GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=1024 * args.memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    if '2d' in args.log_folder:
        meta = args.meta
    else:
        meta = args.meta.split(',')[0]

    def binarize(targets, predictions):
        return targets, (predictions > 0.5).astype(targets.dtype)

    def flip(targets, predictions):
        return 1 - targets, 1 - (predictions > 0.5).astype(targets.dtype)

    log_folder = args.log_folder + '_' + args.dataset_file[:-5].split('/')[-1]

    if not os.path.exists(log_folder):
        shutil.copytree(args.log_folder, log_folder)

    if not os.path.exists(log_folder + '/model'):
        shutil.copytree(args.log_folder + '/model', log_folder + '/model')

    ex = DefaultExperimentPipeline(
        log_base_path=log_folder,
        temp_base_path=args.temp_folder + '_' +
        args.dataset_file[:-5].split('/')[-1]
    )
    if args.best_epoch == 0:
        try:
            ex = ex.load_best_model(
                monitor=args.monitor,
                use_raw_log=False,
                mode=args.monitor_mode,
                custom_modifier_fn=metric_avg_score
            )
        except Exception as e:
            print("Error while loading best model", e)
            print(e)
    else:
        print(f'Loading model from epoch {args.best_epoch}')
        ex.from_file(args.log_folder +
                     f'/model/model.{args.best_epoch:03d}.h5')
    ex.run_external(
        args.dataset_file
    ).apply_post_processors(
        map_meta_data=meta, run_test=True,
        metrics=['AUC', 'roc_auc', 'BinaryCrossentropy',
                 'BinaryAccuracy', 'mcc', 'f1', 'f1'],
        metrics_sources=['tf', 'sklearn',
                         'tf', 'tf', 'sklearn', 'sklearn', 'sklearn'],
        process_functions=[None, None, None, None, binarize, binarize, flip],
        metrics_kwargs=[{}, {}, {}, {}, {}, {}, {'metric_name': 'f1_0'}]
    )
