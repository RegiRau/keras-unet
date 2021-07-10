import ensurepip

import numpy as np
from keras_unet import TF
if TF:
    from tensorflow.keras import backend as K
else:
    from keras import backend as K

import tensorflow as tf


def convert_to_tensor(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg