# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:11:10 2020

@author: Christian Zwinkels-Valero

Prints a prediction to the console and shows what the prediction should be as 
a pyplot.
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# Load the data and the model
DevSet = np.load("MNIST_DevSet.npy")
DevSet = tf.convert_to_tensor(DevSet, dtype=tf.float32)
model = tf.keras.models.load_model("yes.model")

# Make the predictions
preds = model.predict(DevSet)

# See how right the prediction was
rint = np.random.randint(0, high=len(DevSet))
print(np.argmax(preds[rint]))  # Prints a random prediction
plt.imshow(DevSet[rint])  # Plots what the prediction should be
plt.show()
