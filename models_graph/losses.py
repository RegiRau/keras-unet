import ensurepip

import numpy as np
from keras_unet import TF
if TF:
    from tensorflow.keras import backend as K
else:
    from keras import backend as K
import keras

def loss_node_positions(y_true, y_pred):

    loss_positions = K.sum(K.square(y_true - y_pred), axis=-1)
    return loss_positions

def loss_adjacency(y_true, y_pred):
    loss_adjacency = keras.losses.BinaryCrossentropy()
    return loss_adjacency(y_true, y_pred)

