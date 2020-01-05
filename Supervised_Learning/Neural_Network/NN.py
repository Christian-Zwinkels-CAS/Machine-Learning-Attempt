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

def d_sigmoid(z):
    return sigmoid(z)*(1 - sigmoid(z))

def relu(z, d=False):
    if d == False:
        f = np.maximum(0.001*z, z)
    else:
        if z < 0:
            f = 0.001
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
weights = [np.random.rand(l[0], l[1])*np.sqrt(1/l[1]) for l in weight_sizes]
biases = [np.zeros((i, 1)) for i in layer_sizes[1:]]

# Foward propagation
def feedforward(data_in, Ws, Bs):
    Z = []
    A = [data_in]  # First activation layer is the inputs

    # Hidden layer computation
    for i in range(len(Ws) - 1):
        z = np.dot(Ws[i], A[-1]) + Bs[i]
        a = relu(z, d=False)
        Z.append(z)
        A.append(a)

    # Ouput layer computation
    z = np.dot(Ws[-1], A[-1]) + Bs[-1]
    Z.append(z)
    a = sigmoid(z)
    A.append(a)
    return Z, A


# Calculating the costs
def costs(data_in, outputs, Ws, Bs):
    Z, pred = feedforward(data_in, Ws, Bs)
    delta = []
    dj_dw = []

    # Loss computation
    loss = -1*(outputs*np.log(pred[-1]) + (1-outputs)*np.log(1 - pred[-1]))
    loss = np.mean(loss)

    # Final layer derivatives
    dj_da = -1*(pred[-1] - outputs)/((pred[-1] - 1)*pred[-1])
    da_dz = d_sigmoid(Z[-1])
    dz_dw = pred[-2]
    delta.append(np.mean(dj_da*da_dz))
    dj_dw.append(delta * np.mean(dz_dw, axis=1))

    # Rest of the derivatives
    for i in range(1, len(Ws)):
        d = np.dot(weights[-i].T, delta[0])
        delta.insert(0, d*d_sigmoid(-i-1))
    for i in range(len(delta)):
        pass
    return loss, delta
