'''
Model_8 is implemented entirely based on the gcn layer.
Feature extraction as four parallel processes without face data.
'''

import tensorflow as tf
import keras
from keras import layers, Input, Sequential
import numpy as np
import matplotlib.pyplot as plt
import ReadDatei
import GraphConv


# define the gcn network
def gcn_net(X_, D_, A_, num_point, num_layer):

    count = 0

    while count < num_layer:
        X_ = layers.Flatten()(X_)
        X_ = layers.Reshape((-1, 1))(X_)
        X_ = GraphConv.g_conv(X_, D_, A_, num_point)
        count += 1

    return X_

# draw statistical graph
def plot_hist(train, val, title):
    plt.plot(train)
    plt.plot(val)
    plt.title("model_1" + title)
    plt.ylabel(title)
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


# adjacency matrix and degree matrix
A = np.load("A.npy")
D = np.load("D.npy")

# define hyperparameters
epochs = 100
learn_rate = 0.00001
depth_network = 2

# define network
# define input
input_tensor_eyebrow = layers.Input(shape=(224, 224, 1, ))
input_tensor_eye = layers.Input(shape=(224, 224, 1, ))
input_tensor_nose = layers.Input(shape=(224, 224, 1, ))
input_tensor_mouth = layers.Input(shape=(224, 224, 1, ))

input_tensor_eyebrow_mp = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor_eyebrow)
input_tensor_eye_mp = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor_eye)
input_tensor_nose_mp = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor_nose)
input_tensor_mouth_mp = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor_mouth)

# define gcn layer
eyebrow = gcn_net(input_tensor_eyebrow_mp, D, A, 12544, depth_network)
eye = gcn_net(input_tensor_eye_mp, D, A, 12544, depth_network)
nose = gcn_net(input_tensor_nose_mp, D, A, 12544, depth_network)
mouth = gcn_net(input_tensor_mouth_mp, D, A, 12544, depth_network)

# stitching matrix
new_tensor = keras.layers.concatenate((eyebrow, eye, nose, mouth), axis=-1)
new_tensor = layers.Flatten()(new_tensor)

# dense layer
dense_layer = layers.Dense(units=1000, kernel_initializer="he_normal")(new_tensor)
dense_layer = keras.layers.BatchNormalization()(dense_layer)
dense_layer = layers.Activation('relu')(dense_layer)


# output
opt_tensor = layers.Dense(units=7, activation=keras.activations.softmax)(dense_layer)

network = keras.Model([input_tensor_eyebrow, input_tensor_eye, input_tensor_nose, input_tensor_mouth], opt_tensor)

# model
network.compile(
    optimizer=tf.optimizers.Adam(learning_rate=learn_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy']
)

# training
history = network.fit_generator(ReadDatei.read_datasets(50, "train_datas", 4), steps_per_epoch=630, epochs=epochs,
                                validation_data=ReadDatei.read_datasets(50, "val_datas", 4), validation_steps=70)

# result
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plot_hist(acc, val_acc, "accuracy")
plot_hist(loss, val_loss, "loss")