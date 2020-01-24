# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:11:43 2020

@author: Christian Zwinkels-Valero

This code is practically copied from the tensorflow tutorials and is mostly for
me to get familiar with tesorflow
"""

import tensorflow as tf
import numpy as np

# Load the data to numpy arrays
data_train_in = np.load("X.npy")
data_train_out = np.load("labels.npy")

# Convert numpy array to tensors and normalize features
x_train = tf.convert_to_tensor(data_train_in, dtype=tf.float32)
x_train = tf.keras.utils.normalize(x_train)
y_train = tf.convert_to_tensor(data_train_out, dtype=tf.float32)

# Create the model and add the neural network layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # Turns the inputs into rank 1 tensor
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # First NN layer with 128 neurons using the relu activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # Last layer using softmax

# Model configuration
model.compile(optimizer="SGD", loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, epochs=7)

# Save the model
model.save("yes.model")
