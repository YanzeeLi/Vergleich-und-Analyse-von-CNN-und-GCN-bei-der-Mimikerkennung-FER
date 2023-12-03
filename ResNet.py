'''
Resnet network composed of cnn layer.
'''

import tensorflow as tf
import keras
from keras import layers, Input, Sequential

# res_block
def res_block(inp, fn, stride):

    # first layer in res_block
    conv_a = layers.Conv2D(filters=fn, kernel_size=(3, 3), strides=(stride, stride),
                           padding='same', input_shape=inp.shape,
                           kernel_initializer="he_normal")(inp)
    
    conv_a = keras.layers.BatchNormalization()(conv_a)
    conv_a = layers.Activation('relu')(conv_a)

    # second layer in res_block
    conv_b = layers.Conv2D(filters=fn, kernel_size=(3, 3), strides=(1, 1),
                           padding='same', input_shape=conv_a.shape,
                           kernel_initializer="he_normal")(conv_a)
    
    conv_b = keras.layers.BatchNormalization()(conv_b)
    conv_b = layers.Activation('relu')(conv_b)

    # perform plus operation
    if stride == 1:
        otp = keras.layers.add([inp, conv_b])
    else:
        conv_c = layers.Conv2D(filters=fn, kernel_size=(1, 1), strides=(stride, stride),
                               padding='valid', input_shape=inp.shape,
                               kernel_initializer="he_normal")(inp)
        
        conv_c = keras.layers.BatchNormalization()(conv_c)
        conv_c = layers.Activation('relu')(conv_c)
        otp = keras.layers.add([conv_c, conv_b])

    otp = layers.Activation('relu')(otp)

    return otp

# res_net
def res_net(input_tensor):
    # layer 1
    conv_0 = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                           padding='same', input_shape=(224, 224, 1,),
                           kernel_initializer="he_normal")(input_tensor)
    
    conv_0 = keras.layers.BatchNormalization()(conv_0)
    conv_0 = layers.Activation('relu')(conv_0)
    conv_0 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_0)

    # res_block
    res_block_1 = res_block(conv_0, 64, 1)
    res_block_2 = res_block(res_block_1, 64, 1)
    res_block_3 = res_block(res_block_2, 128, 2)
    res_block_4 = res_block(res_block_3, 128, 1)
    res_block_5 = res_block(res_block_4, 256, 2)
    res_block_6 = res_block(res_block_5, 256, 1)
    res_block_7 = res_block(res_block_6, 512, 2)
    res_block_8 = res_block(res_block_7, 512, 1)

    # pooling
    opt = keras.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(res_block_8)

    return opt
