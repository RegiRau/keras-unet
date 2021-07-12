import ensurepip

import numpy as np
from keras_unet import TF
if TF:
    from tensorflow.keras import backend as K
else:
<<<<<<< HEAD
    from keras import backend as K
import keras
=======
    from tensorflow.keras import backend as K
import tensorflow.keras as keras
>>>>>>> 34c5323c8ffbff520fec42ed430bf9a9dbba4f78

def loss_node_positions(y_true, y_pred):

    loss_positions = K.sum(K.square(y_true - y_pred), axis=-1)
    return loss_positions

def loss_adjacency(y_true, y_pred):
    loss_adjacency = keras.losses.BinaryCrossentropy()
    return loss_adjacency(y_true, y_pred)

