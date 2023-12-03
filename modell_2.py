'''
Model_2 is implemented based on the cnn layer + gcn layer.
Feature extraction as five parallel processes.
'''

import tensorflow as tf
import keras
from keras import layers, Input, Sequential
import numpy as np
import matplotlib.pyplot as plt
import ResNet
import ReadDatei
import GraphConv


# initialization matrix
A = np.ones((5, 5))
D = np.zeros((5, 5))

A[1][3], A[1][4], A[2][4], A[3][1], A[4][1], A[4][2] = 0, 0, 0, 0, 0, 0
D[0][0], D[1][1], D[2][2], D[3][3], D[4][4] = 5, 3, 4, 4, 3

# draw statistical graph
def plot_hist(train, val, title):
    plt.plot(train)
    plt.plot(val)
    plt.title("model_1" + title)
    plt.ylabel(title)
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


# define hyperparameters
epochs = 100
learn_rate = 0.000005

# define network
# define input
input_tensor_face = layers.Input(shape=(224, 224, 1,))
input_tensor_eyebrow = layers.Input(shape=(224, 224, 1,))
input_tensor_eye = layers.Input(shape=(224, 224, 1,))
input_tensor_nose = layers.Input(shape=(224, 224, 1,))
input_tensor_mouth = layers.Input(shape=(224, 224, 1,))

# define cnn layer
face = ResNet.res_net(input_tensor_face)
eyebrow = ResNet.res_net(input_tensor_eyebrow)
eye = ResNet.res_net(input_tensor_eye)
nose = ResNet.res_net(input_tensor_nose)
mouth = ResNet.res_net(input_tensor_mouth)

# stitching feature_matrix
new_tensor = keras.layers.concatenate((face, eyebrow, eye, nose, mouth), axis=-1)
new_tensor = layers.Flatten()(new_tensor)
new_tensor = layers.Reshape((5, 512))(new_tensor)

# gcn layer
gcn_layer = GraphConv.g_conv(new_tensor, D, A, 5)
gcn_layer = layers.Flatten()(gcn_layer)

# output
opt_tensor = layers.Dense(units=7, activation=keras.activations.softmax)(gcn_layer)

network = keras.Model([input_tensor_face, input_tensor_eyebrow, input_tensor_eye,
                       input_tensor_nose, input_tensor_mouth], opt_tensor)

# model
network.compile(
    optimizer=tf.optimizers.Adam(learning_rate=learn_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy']
)

# training
history = network.fit_generator(ReadDatei.read_datasets(50, "train_datas", 5), steps_per_epoch=630, epochs=epochs,
                                validation_data=ReadDatei.read_datasets(50, "val_datas", 5), validation_steps=70)

# result
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plot_hist(acc, val_acc, "accuracy")
plot_hist(loss, val_loss, "loss")


