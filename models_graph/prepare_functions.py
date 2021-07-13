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


def create_adj_matrix(adj_vector, size = 35):
    adj_matrix = np.zeros((size, size))

    adj_matrix[np.triu_indices(size, k = 1)]  = adj_vector[0:np.shape(np.triu_indices(size, k = 1))[1]]
    adj_matrix[np.tril_indices(size, k = -1)] = adj_vector[0:np.shape(np.triu_indices(size, k = 1))[1]]
    return(adj_matrix)


def create_position_matrix(position_vector, size = 35):
    length = int(len(position_vector)/2)
    position_matrix = position_vector.reshape(length,2)
    position_matrix = position_matrix[0:size,:]
    return position_matrix

