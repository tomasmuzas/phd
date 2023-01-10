import tensorflow as tf
from augmentation_functions import *

AUTO = tf.data.AUTOTUNE

def get_dataset(training_config, path, batch_size, seed, augment, shuffle, drop_remainder):
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

          "label": tf.io.FixedLenFeature([], dtype=tf.string),
          "objid": tf.io.FixedLenFeature([], dtype=tf.string),
          "one_hot_class": tf.io.VarLenFeature(tf.float32)
      }),
      num_parallel_calls=AUTO)
  dataset = dataset.map(lambda item: (tf.reshape(tf.image.decode_jpeg(item['image'], channels=3), [training_config["IMAGE_SIZE"], training_config["IMAGE_SIZE"], 3]), item['class']), num_parallel_calls=AUTO)
  dataset = dataset.map(lambda x,y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTO)
  dataset = dataset.map(lambda x,y: (tf.keras.layers.Rescaling(scale=1./255)(x), y), num_parallel_calls=AUTO)

  if(augment):
      if training_config["AUGMENTATIONS_ZOOM"]:
        dataset = dataset.map(lambda x,y : (random_zoom(x, training_config["IMAGE_SIZE"], seed), y), num_parallel_calls=AUTO)
      if training_config["AUGMENTATIONS_FLIP_HORIZONTALLY"]:
        dataset = dataset.map(lambda x,y : (random_invert_horizontally(x, seed), y), num_parallel_calls=AUTO)
      if training_config["AUGMENTATIONS_FLIP_VERTICALLY"]:
        dataset = dataset.map(lambda x,y : (random_invert_vertically(x, seed), y), num_parallel_calls=AUTO)
      if training_config["AUGMENTATIONS_ROTATE"]:
        dataset = dataset.map(lambda x,y : (random_rotate(x, seed), y), num_parallel_calls=AUTO)
      if training_config["AUGMENTATIONS_RANDOM_NOISE"]:
        dataset = dataset.map(lambda x,y : (random_noise(x, seed), y), num_parallel_calls=AUTO)

  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(200 if training_config["TPU"] else 3)

  return dataset