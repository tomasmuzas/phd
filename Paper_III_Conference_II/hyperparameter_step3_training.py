import os
import warnings
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.get_logger().setLevel("ERROR")
warnings.simplefilter("ignore")

from optuna.storages import RetryFailedTrialCallback
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, multiply, Conv2D, Dense, BatchNormalization, LayerNormalization, Input, MaxPool2D, Dropout, Flatten
import numpy as np
import keras
from phd.Augmentations.common_code.augmentation_functions import *
from phd.Augmentations.common_code.get_dataset import *
from phd.Augmentations.common_code.models import *
from phd.Augmentations.common_code.training_loop import *
from phd.Augmentations.common_code.utils import *
import optuna

AUTO = tf.data.AUTOTUNE


import wandb

wandb.login()

bin_folder_name = input("Enter bin folder name:")
experiment_name = input("Enter experiment name:")
train_size = int(input("Train size:"))
test_size = int(input("Test size:"))


training_config = {
"ENABLE_DETERMINISM": False,
"SEED": 17,
"AUTO": tf.data.AUTOTUNE,
"TRAIN_BATCH_SIZE": 128,
"TEST_BATCH_SIZE": 128,
"NUMBER_OF_EPOCHS": 500,
"EARLY_STOPPING_TOLERANCE": 10,
"LEARNING_RATE": 1e-4,
"IMAGE_SIZE": 128,
"SHUFFLE_BUFFER": train_size,
"TPU": False,
"AUGMENTATIONS_ROTATE": False,
"AUGMENTATIONS_FLIP_HORIZONTALLY": False,
"AUGMENTATIONS_FLIP_VERTICALLY": False,
"AUGMENTATIONS_ZOOM": False,
"AUGMENTATIONS_RANDOM_NOISE": False,
"AUGMENTATIONS_CENTER_NOISE": False,
"AUGMENTATIONS_OUTSIDE_NOISE": False,
"AUGMENTATIONS_PERLIN_CENTER_NOISE": False,
"AUGMENTATIONS_SALT_AND_PEPPER": False,
"AUGMENTATIONS_CUTOUT": False,
"USE_ADABELIEF_OPTIMIZER": False,
"GALAXY_DATASET": "SDSS",
"TRAIN_DATASET_SIZE": train_size,
"TEST_DATASET_SIZE": test_size,
"STRETCH": False,
"STEPS_PER_EXECUTION": 1,
"FOLDS": 10,
"NUMBER_OF_CLASSES": 2,
"CACHE": False,
"METRIC": "F1",
"LOAD_WEIGHTS": False
}
training_config["WANDB_PROJECT_NAME"] = f"Paper3, Step 3"
training_config["EXPERIMENT_DESCRIPTION"] = f"paper3_step3_{experiment_name}"
training_config["LOCAL_GCP_PATH_BASE"] = f"."
training_config["REMOTE_GCP_PATH_BASE"] = f"."
training_config["DATASET_PATH"] = f"TFDataset/{bin_folder_name}/training"


@keras.saving.register_keras_serializable()
class FocalConv(keras.layers.Layer):
    def __init__(self, filters, kernel_size, factor, activation):
        super().__init__()
        self.filters = filters
        self.kernel = kernel_size
        self.activation = activation
        self.factor = factor
        self.conv1 = Conv2D(filters=self.filters, kernel_size=self.kernel, activation=self.activation, padding="same")

    def build(self, input_shape):
        self.gauss = self.make_gaussian(size=input_shape[1], depth=self.filters, factor=self.factor)

    def call(self, x, training=False):
      x = self.conv1(x, training=training)
      batch_size = tf.shape(x)[0]
      xx = tf.repeat(self.gauss, repeats=batch_size, axis=0)
      x = multiply([x, xx])

      return x
    
    def get_config(self):
       return {
          "filters": self.filters,
          "kernel_size": self.kernel,
          "activation": self.activation,
          "factor": self.factor
       }

    def make_gaussian(self, size: int, depth: int, factor: float):
        """ Make a square gaussian kernel.

        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        fwhm = size // 2
        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]

        x0 = y0 = size // 2

        gauss = np.exp(-1 * factor * np.log(2) * ((x-x0) ** 2 + (y-y0) ** 2) / fwhm **2)

        return tf.expand_dims(np.stack([gauss for _ in range(depth)], -1), axis=0)


def create_model(training_config, depth, filters, kernel_size, focal_factor):
  model = Sequential(name="Cavanagh")
  model.add(Input(shape=(training_config["IMAGE_SIZE"], training_config["IMAGE_SIZE"], 3)))

  for _ in range(depth):
      model.add(FocalConv(filters=filters, kernel_size= kernel_size, factor = focal_factor, activation='relu'))
      model.add(BatchNormalization())
      model.add(MaxPool2D(2))


  model.add(Flatten())
  model.add(Dropout(0.7))

  for _ in range(3):
      model.add(Dense(128, activation='relu'))

  model.add(Dense(training_config["NUMBER_OF_CLASSES"], activation='softmax'))

  return model

def best_sdss(training_config):
  return create_model(training_config, 4, 128, 5, 10)

def best_des(training_config):
  return create_model(training_config, 4, 256, 5, 9)

def best_h_1(training_config):
  return create_model(training_config, 4, 128, 7, 2)

def best_h_2(training_config):
  return create_model(training_config, 5, 256, 7, 2)

def best_h_3(training_config):
  return create_model(training_config, 3, 64, 7, 4)


models = [
    {'name': 'best_sdss', 'func': best_sdss, 'starting_fold': 1},
    {'name': 'best_des', 'func': best_des, 'starting_fold': 1},
    {'name': 'best_h_1', 'func': best_h_1, 'starting_fold': 1},
    {'name': 'best_h_2', 'func': best_h_2, 'starting_fold': 1},
    {'name': 'best_h_3', 'func': best_h_3, 'starting_fold': 1},
]

perform_training(models, training_config)
