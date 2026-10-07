import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter("ignore")


import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, multiply, Conv2D, Dense, BatchNormalization, Input, MaxPool2D, Dropout, Flatten
import numpy as np
from sklearn.preprocessing import normalize
from keras import layers
from keras import ops
import keras


tf.get_logger().setLevel('ERROR')
tf.get_logger().setLevel("ERROR")



bin_folder_name = input("Enter bin folder name:")
experiment_name = input("Enter experiment name:")
train_size = int(input("Train size:"))
test_size = int(input("Test size:"))


AUTO = tf.data.AUTOTUNE

def get_intial_fold_dataset(training_config, path, seed, shuffle, cache = False):
  if not training_config["ENABLE_DETERMINISM"]:
    seed = None

  dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(path + "/*.tfrec"), num_parallel_reads=AUTO) # if TPU else 20)

  if(shuffle):
    dataset = dataset.shuffle(training_config["SHUFFLE_BUFFER"], seed = seed)

  dataset = dataset.map(lambda records: tf.io.parse_single_example(
      records,
      {
          "image": tf.io.FixedLenFeature([], dtype=tf.string),
          "class": tf.io.FixedLenFeature([], dtype=tf.int64),

          # "label": tf.io.FixedLenFeature([], dtype=tf.string),
          "objid": tf.io.FixedLenFeature([], dtype=tf.string),
          # "one_hot_class": tf.io.VarLenFeature(tf.float32)
      }),
      num_parallel_calls=AUTO)
  dataset = dataset.map(lambda item: (tf.reshape(tf.image.decode_jpeg(item['image'], channels=3), [training_config["IMAGE_SIZE"], training_config["IMAGE_SIZE"], 3]), item['class']), num_parallel_calls=AUTO)
  if(cache):
    print(f"caching jpeg dataset into memory")
    dataset = dataset.cache()

  dataset = dataset.map(lambda x,y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTO)
  dataset = dataset.map(lambda x,y: (tf.keras.layers.Rescaling(scale=1./255)(x), y), num_parallel_calls=AUTO)

  return dataset

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
"SHUFFLE_BUFFER": 4096,
"TPU": False,
"AUGMENTATIONS_ROTATE": False,
"AUGMENTATIONS_FLIP_HORIZONTALLY": False,
"AUGMENTATIONS_FLIP_VERTICALLY": False,
"AUGMENTATIONS_ZOOM": False,
"AUGMENTATIONS_RANDOM_NOISE": False,
"AUGMENTATIONS_CENTER_NOISE": False,
"AUGMENTATIONS_OUTSIDE_NOISE": False,
"AUGMENTATIONS_PERLIN_CENTER_NOISE": False,
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
training_config["LOCAL_GCP_PATH_BASE"] = f"TFDataset/{bin_folder_name}"
training_config["REMOTE_GCP_PATH_BASE"] = f"TFDataset/{bin_folder_name}"
training_config["DATASET_PATH"] = f"tuning"


@keras.saving.register_keras_serializable()
class FocalConv(keras.layers.Layer):
    def __init__(self, filters, kernel_size, factor, activation):
        super().__init__()
        self.filters = filters
        self.kernel = kernel_size
        self.activation = activation
        self.factor = factor
        self.conv1 = Conv2D(filters=self.filters, kernel_size=self.kernel, activation=self.activation, padding='same')

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

def model_builder(hp):
    
  model = Sequential(name="Custom")
  model.add(Input(shape=(128, 128, 3)))

  num_layers = hp.Int('num_layers', min_value=3, max_value=5, step=1)

  model = Sequential(name="Cavanagh")
  model.add(Input(shape=(training_config["IMAGE_SIZE"], training_config["IMAGE_SIZE"], 3)))

  for i in range(num_layers):

    layer_type = hp.Choice(f"type_{i}", ["SimpleConv", "FocalConv"])
    focal_factor = hp.Int(f'focal_factor_{i}', parent_name=f'type_{i}', parent_values=["FocalConv"], min_value=1, max_value=10, step=1)
    kernel_size = hp.Int(f'kernel_size_{i}', min_value=3, max_value=7, step=2)
    filters = hp.Int(f'filters_{i}', min_value=16, max_value=64, step=16)

    if (layer_type == "SimpleConv"):
      model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
    elif(layer_type == "FocalConv"):
      model.add(FocalConv(filters=filters, kernel_size=kernel_size, activation='relu', factor= focal_factor))

    model.add(BatchNormalization())
    model.add(MaxPool2D(2))

  model.add(Flatten())
  model.add(Dropout(0.5))

  num_dense = hp.Int('num_dense', min_value=1, max_value=3, step=1)
  for j in range(num_dense):
    dense_output = hp.Int(f'dense_output_{j}', min_value=32, max_value=256, step=32)
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
    cache = True).batch(training_config["TEST_BATCH_SIZE"])

train_dataset = get_intial_fold_dataset(
    training_config,
    f"{training_config['REMOTE_GCP_PATH_BASE']}/{training_config['DATASET_PATH']}/train",
    training_config["SEED"],
    shuffle = True,
    cache = True).batch(training_config["TEST_BATCH_SIZE"])

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', mode="max", patience=10)

with tf.device('/GPU:0'):

    tuner = kt.Hyperband(
        model_builder,
        objective=kt.Objective("val_f1_score", direction="max"),
        max_epochs=100,
        overwrite=False,
        hyperband_iterations = 3,
        directory=f'paper3_keras_tuning_focal/{experiment_name}',
        project_name='best_f1',
        max_consecutive_failed_trials=1)
    
    tuner.search(x= train_dataset,
                validation_data = test_dataset,
                epochs = 500,
                callbacks=[stop_early],
                verbose = 1,
                shuffle = False)

    top_hyperparams = tuner.get_best_hyperparameters(1)
    
    for hyperparams in top_hyperparams:
        for prop, val in hyperparams.values.items():
            if "tuner" in prop:
                continue
    
            print(f'{prop}: {val}')