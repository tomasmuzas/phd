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
"SHUFFLE_BUFFER": 9000,
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
"FOLDS": 1,
"NUMBER_OF_CLASSES": 2
}
training_config["WANDB_PROJECT_NAME"] = f"TrainTestValidation, {training_config['NUMBER_OF_CLASSES']} classes"
training_config["EXPERIMENT_DESCRIPTION"] = f"asd"
training_config["LOCAL_GCP_PATH_BASE"] = f"."
training_config["REMOTE_GCP_PATH_BASE"] = f"."
training_config["DATASET_PATH"] = f"TFDataset/{bin_folder_name}/tuning"
 
 
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
 
 
class SparseF1Score(tf.keras.metrics.F1Score):
 
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
 
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(tf.reshape(tf.one_hot(y_true, training_config["NUMBER_OF_CLASSES"]), [-1, training_config["NUMBER_OF_CLASSES"]]), y_pred, sample_weight)
 
    def result(self):
        return super().result()
 
def model_builder(trial_number):
  params = {}
  with open(f'paper3_models_step2/{experiment_name}/{trial_number}/model.json', 'r') as fp:
      params = json.load(fp)
 
  num_layers = params['depth']
  filters = params['filters']
  kernel_size = params['kernel']
  dropout_rate = params['dropout']
  num_dense = params['dense_depth']
  dense_output = params['dense']
  focal = params['focal']
  focal_factor = params['focal_factor']
 
 
  model = Sequential(name="Cavanagh")
  model.add(Input(shape=(training_config["IMAGE_SIZE"], training_config["IMAGE_SIZE"], 3)))
 
  for i in range(num_layers):
 
    if focal:
      model.add(FocalConv(filters=filters, kernel_size=kernel_size, activation='relu', factor = focal_factor))
    else:
      model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2))
 
  
  model.add(Flatten())
  model.add(Dropout(dropout_rate))
 
  for j in range(num_dense):
    model.add(Dense(dense_output, activation='relu'))
  model.add(Dense(training_config["NUMBER_OF_CLASSES"], activation='softmax'))
 
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), SparseF1Score(
                average='macro',
            )])
  return model
 
 
test_dataset = get_intial_fold_dataset(
    training_config,
    f"{training_config['REMOTE_GCP_PATH_BASE']}/{training_config['DATASET_PATH']}/test",
    seed = training_config["SEED"],
    shuffle = False,
    cache = True).batch(training_config["TEST_BATCH_SIZE"]).cache()
 
cached_train_dataset = get_intial_fold_dataset(
    training_config,
    f"{training_config['REMOTE_GCP_PATH_BASE']}/{training_config['DATASET_PATH']}/train",
    training_config["SEED"],
    shuffle = True,
    cache = True)
 
for i in range(297):
    if os.path.exists(f'paper3_models_step2/{experiment_name}/{i}/result.json'):
       print(f"Experiment {i} already finished, skipping.")
       continue
 
    print(f"Trial {i}")
 
    best_epoch = 0
    max_f1 = 0
 
    try:
        with tf.device('/GPU:0'):
            model = model_builder(i)
 
            for epoch in range(500):
                train_dataset = shuffle_dataset(
                cached_train_dataset,
                training_config,
                training_config["TRAIN_BATCH_SIZE"],
                seed = training_config["SEED"] + epoch,
                augment = False,
                drop_remainder = False
                )
 
                history = model.fit(
                x= train_dataset,
                validation_data = test_dataset,
                epochs = 1,
                verbose = 1,
                shuffle = False
                )
 
                val_f1 = history.history['val_f1_score'][-1]
 
                if(val_f1 > max_f1):
                    max_f1 = val_f1
                    best_epoch = epoch
 
                if(epoch - best_epoch > 10):
                    with open(f'paper3_models_step2/{experiment_name}/{i}/result.json', 'w') as fp:
                        json.dump({
                            "total_epochs": epoch,
                            "best_f1": max_f1
                            }, fp)
                    del train_dataset
 
                    gc.collect()
                    break
    except tf.errors.ResourceExhaustedError as e:
       print("OOM error, continuing")
       pass