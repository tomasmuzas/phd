import tensorflow as tf

@tf.function
def random_invert_horizontally(x, seed):
  return tf.cond(
    tf.random.uniform([], seed = seed) < 0.5,
    lambda: tf.image.flip_left_right(x),
    lambda: x)

@tf.function
def random_invert_vertically(x, seed):
  return tf.cond(
    tf.random.uniform([], seed = seed) < 0.5,
    lambda: tf.image.flip_up_down(x),
    lambda: x)

@tf.function
def random_rotate(x, seed):
  return tf.cond(
      tf.random.uniform([], seed = seed) < 0.5,
      lambda: tf.image.rot90(x, k = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32, seed = seed)),
      lambda: x)

@tf.function
def random_zoom(x, image_size, seed):
  if  tf.random.uniform([]) < 0.5:
    x = tf.image.crop_to_bounding_box(x, 10, 10, image_size - 20, image_size - 20)
    x = tf.image.resize(x, (image_size, image_size))
  else:
    x
  return x

@tf.function
def random_noise(x, seed):
  if  tf.random.uniform([], seed = seed) < 0.5:
    noise = tf.random.uniform(shape=tf.shape(x), minval= -0.2, maxval= 0.2, dtype=tf.float32)
    x = tf.add(x, noise)
    x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
  else:
    x
  return x