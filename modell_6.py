'''
Model_6 is a variant of model_1.
But only face data are considered in the feature extraction process.
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
learn_rate = 0.000001
depth_network = 2


# define network
# define input
input_tensor_face = layers.Input(shape=(224, 224, 1, ))

# define gcn layer
input_tensor_face_new = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor_face)

face = gcn_net(input_tensor_face_new, D, A, 12544, depth_network)

# stitching matrix

new_tensor = layers.Flatten()(face)

# dense layer
dense_layer = layers.Dense(units=1000, kernel_initializer="he_normal")(new_tensor)
dense_layer = keras.layers.BatchNormalization()(dense_layer)
dense_layer = layers.Activation('relu')(dense_layer)

# output
opt_tensor = layers.Dense(units=7, activation=keras.activations.softmax)(dense_layer)

network = keras.Model([input_tensor_face], opt_tensor)

# model
network.compile(
    optimizer=tf.optimizers.Adam(learning_rate=learn_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy']
)


# training
history = network.fit_generator(ReadDatei.read_datasets(50, "train_datas", 1), steps_per_epoch=630, epochs=epochs, 
                                validation_data=ReadDatei.read_datasets(50, "val_datas", 1), validation_steps=70)

# result
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plot_hist(acc, val_acc, "accuracy")
plot_hist(loss, val_loss, "loss")