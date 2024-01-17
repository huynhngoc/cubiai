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
# import os
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


try:
    metrics.SCORERS['mcc'] = Matthews_corrcoef_scorer()
except:
    pass
try:
    metrics._scorer._SCORERS['mcc'] = Matthews_corrcoef_scorer()
except:
    pass


def metric_avg_score(res_df, postprocessor):
    res_df['avg_score'] = res_df[['AUC', 'accuracy', 'mcc']].mean(axis=1)
    return res_df


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("log_folder")
    parser.add_argument("--temp_folder", default='', type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--model_checkpoint_period", default=1, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=1, type=int)
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
    print('training from configuration', args.config_file,
          'and saving log files to', args.log_folder)
    print('Unprocesssed prediction are saved to', args.temp_folder)

    def binarize(targets, predictions):
        return targets, (predictions > 0.5).astype(targets.dtype)

    def flip(targets, predictions):
        return 1 - targets, 1 - (predictions > 0.5).astype(targets.dtype)

    def decode(targets, predictions):
        return targets.argmax(axis=1), predictions.argmax(axis=1)
    exp = DefaultExperimentPipeline(
        log_base_path=args.log_folder,
        temp_base_path=args.temp_folder
    ).from_full_config(
        args.config_file
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=20,
        prediction_checkpoint_period=20,
        epochs=20,
        save_val_inputs=False,
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=args.model_checkpoint_period,
        prediction_checkpoint_period=args.prediction_checkpoint_period,
        epochs=args.epochs,
        initial_epoch=20,
        save_val_inputs=False,
    ).apply_post_processors(
        map_meta_data=meta, run_test=False,
        metrics=['AUC', 'roc_auc', 'roc_auc', 'CategoricalCrossentropy',
                 'BinaryAccuracy', 'mcc', 'accuracy'],
        metrics_sources=['tf', 'sklearn', 'sklearn',
                         'tf', 'tf', 'sklearn', 'sklearn'],
        process_functions=[None, None, None, None, None, decode, decode],
        metrics_kwargs=[{}, {'metric_name': 'roc_auc_ovr', 'multi_class': 'ovr'},
                        {}, {}, {}, {}, {}]
    ).plot_performance().load_best_model(
        monitor=args.monitor,
        use_raw_log=False,
        mode=args.monitor_mode,
        custom_modifier_fn=metric_avg_score
    ).run_test(
    ).apply_post_processors(
        map_meta_data=meta, run_test=True,
        metrics=['AUC', 'roc_auc', 'roc_auc', 'CategoricalCrossentropy',
                 'BinaryAccuracy', 'mcc', 'accuracy'],
        metrics_sources=['tf', 'sklearn', 'sklearn',
                         'tf', 'tf', 'sklearn', 'sklearn'],
        process_functions=[None, None, None, None, None, decode, decode],
        metrics_kwargs=[{}, {'metric_name': 'roc_auc_ovr', 'multi_class': 'ovr'},
                        {}, {}, {}, {}, {}]
    )
