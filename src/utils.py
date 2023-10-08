import tensorflow as tf
import warnings
import yaml
import shutil
import os
from pathlib import Path
tfk = tf.keras
K = tfk.backend

from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
import numpy as np
from tensorflow.keras.metrics import MeanIoU

def create_callbacks(config):

    sm_dir = config.run_dir
    if config.training.reset:
        if os.path.exists(sm_dir):
            shutil.rmtree(sm_dir)
    log_dir = sm_dir+'logs/'
    checkpoints_dir = sm_dir+'checkpoints/'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    history_output_path = sm_dir+'history.csv'

    to_track = config.training.export_metric
    checkpoint_path = str(checkpoints_dir) + "/sm-{epoch:04d}"
    checkpoint_path = checkpoint_path + "-{" + to_track + ":4.5f}.hdf5"
    check1 = ModelCheckpoint(checkpoint_path,
                            #  monitor = config.training.export_metric,
                             verbose = 1, 
                             save_weights_only = True, 
                            #  mode = config.training.export_mode
                             )

    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode="min", min_lr=1e-8)
    history_logger = CSVLogger(history_output_path, separator=",", append=True)
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)
    return [check1, lr_reduction, history_logger, tensor_board]

def get_checkpoints_info(checkpoints_dir: Path):
    """Returns info about checkpoints.

    Returns:
        A list of dictionaries related to each checkpoint:
            {'epoch': int, 'path': pathlib.Path, 'value': float}
    """

    checkpoints = checkpoints_dir.glob('*.hdf5')
    ckpt_info = list()
    for cp in checkpoints:
        splits = str(cp.name).split('.hdf5')[0].split('-')
        epoch = int(splits[1])
        metric_value = float(splits[2])
        ckpt_info.append({'path': cp, 'epoch': epoch, 'value': metric_value})
    return ckpt_info

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_keras = true_positives / (possible_positives + K.epsilon())
        return recall_keras

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_keras = true_positives / (predicted_positives + K.epsilon())
        return precision_keras
    
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def dice_coef(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    # y_true_f = tf.y_true_f
    # y_pred_f = tf.y_pred_f
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    score = (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
  union = K.sum(y_true, [1, 2, 3])+K.sum(y_pred, [1, 2, 3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

class m_iou():
    def __init__(self, classes: int) -> None:
        self.classes = classes
    def mean_iou(self,y_true, y_pred):
        y_pred = np.argmax(y_pred, axis = 3)
        miou_keras = MeanIoU(num_classes= self.classes)
        miou_keras.update_state(y_true, y_pred)
        return miou_keras.result().numpy()
    def miou_class(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis = 3)
        miou_keras = MeanIoU(num_classes= self.classes)
        miou_keras.update_state(y_true, y_pred)        
        values = np.array(miou_keras.get_weights()).reshape(self.classes, self.classes)
        for i in  range(self.classes):
            class_iou = values[i,i] / (sum(values[i,:]) + sum(values[:,i]) - values[i,i])
            print(f'IoU for class{str(i + 1)} is: {class_iou}')

def dice_loss(y_true, y_pred):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.math.sigmoid(y_pred)
  numerator = 2 * tf.reduce_sum(y_true * y_pred)
  denominator = tf.reduce_sum(y_true + y_pred)

  return 1 - numerator / denominator

def load_config_file(path):
    """
    loads the yaml config file and returns a dictionary

    :param path: path to yaml config file
    :return: a dictionary of {config_name: config_value}
    """
    with open(path) as f:
        data_map = yaml.safe_load(f)

    config_obj = Struct(**data_map)
    return config_obj

class Struct:
    def __init__(self, **entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Struct(**v)
            else:
                self.__dict__[k] = v
