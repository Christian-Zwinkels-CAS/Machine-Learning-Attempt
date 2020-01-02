# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 16:02:11 2019

@author: Christian Zwinkels-Valero
"""

import pandas as pd
import numpy as np

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(z))

def relu(z, d=False):
    if d == False:
        f = np.maximum(0.001*z, z)
    else:
        if z < 0:
            f = 0
        elif z <= 1:
            f = 1
    return f


# Data processing
data = pd.read_csv("IRISS.csv", header=None, skiprows=1)
data = data.sample(frac=1)
X = data[data.columns[0:data.shape[-1] - 1]].to_numpy()
X = (X - np.mean(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
X = X.T
y = np.array([data[data.shape[-1] - 1].to_numpy()])

# Initialization
layer_sizes = (X.shape[0], 4, 3, 2, y.shape[0])
weight_sizes = [(y, x) for y, x in zip(layer_sizes[1:], layer_sizes[0:])]
weights = [np.random.standard_normal(l) for l in weight_sizes]
biases = [np.zeros((i, 1)) for i in layer_sizes[1:]]

# Foward propagation
def feedforward(data_in, Ws, Bs):
    Z = []
    A = [data_in]  # First activation layer is the inputs
    # Hidden layer computation
    for i in range(len(Ws) - 1):
        z = np.dot(Ws[i], A[-1]) + Bs[i]
        a = relu(z)
        Z.append(z)
        A.append(a)
    # Ouput layer computation
    z = np.dot(Ws[-1], A[-1]) + Bs[-1]
    a = sigmoid(z)
    A.append(a)
    return A


# Calculating the costs
def costs(data_in, outputs, Ws, Bs):
    pred = feedforward(data_in, Ws, Bs)[-1]
    loss = -1*(outputs*np.log(pred) + (1-outputs)*np.log(1 - pred))
    loss = np.mean(loss)
    return loss
