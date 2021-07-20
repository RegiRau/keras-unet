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

#
# y_label_positions = y_label_positions.reshape((y_label_positions.shape[0] * 2))
# adjacency_label_indices = np.triu_indices(y_label_adjacency.shape[1], k=1)
# y_label_adjacency = y_label_adjacency[adjacency_label_indices[0], adjacency_label_indices[1]]
# if y_label_positions.shape[0] >= network_dim * 2:
#     print(
#         'the number of labeld nodes/frame is too high for network dimension - decrease nodes in training data or consider to adapt the network size')
#     a[index, 0:network_dim * 2] = y_label_positions[0:network_dim * 2]
#     b[index, 0:adj_flatten_dim] = y_label_adjacency[0:adj_flatten_dim]
# else:


def create_adj_matrix(adj_vector,networksize , cut_off_size =None):
    if cut_off_size == None:
        cut_off_size = networksize
    adj_matrix = np.zeros((networksize, networksize))

    adj_matrix[np.triu_indices(networksize, k = 1)] = adj_vector[0:np.shape(np.triu_indices(networksize, k = 1))[1]]
    adj_matrix = adj_matrix+np.transpose(adj_matrix)

    adj_matrix = adj_matrix[0:cut_off_size,0:cut_off_size]
    return(adj_matrix)


def create_position_matrix(position_vector, cut_off_size = None):
    length = int(len(position_vector)/2)
    position_matrix = position_vector.reshape(length,2)
    if cut_off_size == None:
        cut_off_size = length
    position_matrix = position_matrix[0:cut_off_size,:]
    return position_matrix

