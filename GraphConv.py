'''
This file implements the graph convolution layer.
The weight update is achieved through the conv1 layer.
The multiplication between the weight matrix and the feature matrix is realized by the conv1 + reshape method
'''

import numpy as np
import tensorflow as tf
import keras
from keras import layers, Input, Sequential


# GCN layer
def g_conv(X, D, A, num_point):
    # H(l+1)= Delta(D^-1/2 dot A dot D^-1/2 dot H(l) dot W(l))
    D_ = np.sqrt(np.linalg.inv(D))
    G = (D_.dot(A)).dot(D_)
    graph = tf.constant(G, "float32")

    # gcn layer
    g_cov_tensor = tf.matmul(graph, X)
    g_cov_tensor = tf.transpose(g_cov_tensor, [0, 2, 1])
    g_cov_tensor = layers.Conv1D(filters=num_point, kernel_size=1, strides=1,
                                 padding='valid', kernel_initializer="he_normal",
                                 input_shape=g_cov_tensor.shape)(g_cov_tensor)
    g_cov_tensor = keras.layers.BatchNormalization()(g_cov_tensor)
    g_cov_tensor = layers.Activation('relu')(g_cov_tensor)
    opt = tf.transpose(g_cov_tensor, [0, 2, 1])

    return opt

