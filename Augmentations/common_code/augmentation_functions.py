import tensorflow as tf
import numpy as np
from astropy.visualization import PercentileInterval
from astropy.modeling.models import Sersic2D
from perlin_numpy import generate_perlin_noise_2d

x,y = np.meshgrid(np.arange(128), np.arange(128))

mod = Sersic2D(amplitude = 3, r_eff = 5, n=1, x_0=64, y_0=64,
               ellip=0, theta=1)
img = mod(x, y)
log_img = np.log10(img)
center_noise_profile_2d = PercentileInterval(90)(log_img)
center_noise_profile = np.repeat(center_noise_profile_2d[:,:,None], 3, axis=2)
outside_noise_profile = 1 - center_noise_profile

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


@tf.function
def targeted_center_noise(x, seed):
  if  tf.random.uniform([], seed = seed) < 0.5:
    noise = tf.random.uniform(shape=tf.shape(x), minval= -0.2, maxval= 0.2, dtype=tf.float32)
    center_noise = tf.math.multiply(noise, center_noise_profile)
    x = tf.add(x, center_noise)
    x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
  else:
    x
  return x

@tf.function
def targeted_outside_noise(x, seed):
  if  tf.random.uniform([], seed = seed) < 0.5:
    noise = tf.random.uniform(shape=tf.shape(x), minval= -0.2, maxval= 0.2, dtype=tf.float32)
    outside_noise = tf.math.multiply(noise, outside_noise_profile)
    x = tf.add(x, outside_noise)
    x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
  else:
    x
  return x

@tf.function
def perlin_center_noise(x, seed):
  if  tf.random.uniform([], seed = seed) < 0.5:
    perlin = generate_perlin_noise_2d((128, 128), (16, 16))
    perlin_3d = np.repeat(perlin[:,:,None], 3, axis=2)
    perlin_with_noise = perlin_3d * center_noise_profile * 0.1
    x = tf.add(x, perlin_with_noise)
    x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
  else:
    x
  return x