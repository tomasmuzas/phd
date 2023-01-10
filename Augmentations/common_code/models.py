import os
import numpy as np
import random
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, Input, MaxPool2D

def create_ResNet50(training_config):
  model = Sequential()
  model.add(ResNet50(
    input_shape = (training_config["IMAGE_SIZE"], training_config["IMAGE_SIZE"], 3),
    include_top = False,
    weights = None,
    pooling = 'avg'))
  model.add(Dense(1, activation='sigmoid'))
  return model

def create_EffNetB7(training_config):
  model = Sequential()
  model.add(tf.keras.applications.efficientnet.EfficientNetB7(
    input_shape = (training_config["IMAGE_SIZE"], training_config["IMAGE_SIZE"], 3),
    include_top = False,
    weights = None,
    pooling = 'avg'))
  model.add(Dense(1, activation='sigmoid'))
  return model

def Dieleman(training_config):
  model = Sequential(name="Dieleman")
  model.add(Input(shape=(training_config["IMAGE_SIZE"], training_config["IMAGE_SIZE"], 3)))
  model.add(Conv2D(filters=32, kernel_size=6, activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPool2D(2))
  model.add(Conv2D(filters=64, kernel_size=5, activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPool2D(2))
  model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
  model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPool2D(2))
  model.add(Flatten())
  if training_config["ENABLE_DETERMINISM"]:
    model.add(Dropout(0.5, seed = training_config["SEED"]))
  else:
    model.add(Dropout(0.5))
  model.add(Dense(256,activation='relu'))
  model.add(Dense(256,activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  return model

def SimpleModel(training_config):
  model = Sequential(name="SimpleModel")
  model.add(Input(shape=(training_config["IMAGE_SIZE"], training_config["IMAGE_SIZE"], 3)))
  model.add(Conv2D(filters=256, kernel_size=6, activation='relu'))
  model.add(Conv2D(filters=128, kernel_size=5, activation='relu'))
  model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  return model

def SimplerModel(training_config):
  model = Sequential(name="SimplerModel")
  model.add(Input(shape=(training_config["IMAGE_SIZE"], training_config["IMAGE_SIZE"], 3)))
  model.add(Conv2D(filters=128, kernel_size=6, activation='relu'))
  model.add(MaxPool2D(2))
  model.add(Conv2D(filters=32, kernel_size=5, activation='relu'))
  model.add(MaxPool2D(2))
  model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  return model

def Cavanagh(training_config):
  model = Sequential(name="Cavanagh")
  model.add(Input(shape=(training_config["IMAGE_SIZE"], training_config["IMAGE_SIZE"], 3)))
  model.add(Conv2D(filters=32, kernel_size=7, activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPool2D(2))
  model.add(Conv2D(filters=64, kernel_size=5, activation='relu'))
  model.add(Conv2D(filters=64, kernel_size=5, activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPool2D(2))
  model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPool2D(2))
  model.add(Flatten())
  if training_config["ENABLE_DETERMINISM"]:
    model.add(Dropout(0.5, seed = training_config["SEED"]))
  else:
    model.add(Dropout(0.5))
  model.add(Dense(256,activation='relu'))
  model.add(Dense(256,activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  return model