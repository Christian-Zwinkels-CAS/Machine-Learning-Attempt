# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 16:02:11 2019

@author: Christian Zwinkels-Valero
"""

import pandas as pd
import numpy as np

# Activation functions
def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def d_tanh(z):
    return 1 - (tanh(z)**2)

def relu(z, d=False):
    if d == False:
        f = np.maximum(z)
    else:
        if z < 0:
            f = 0
        elif z <= 1:
            f = 1
    return f


# Data processing
data = pd.read_csv("IRISS.csv", header=None, skiprows=1)
X = data[data.columns[0:data.shape[-1] - 1]].to_numpy()
X = (X - np.mean(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
X = X.T
y = np.array([data[data.shape[-1] - 1].to_numpy()])

# Initialization
layer_sizes = (X.shape[0], 4, 3, 2, y.shape[0])
weight_sizes = [(y, x) for y, x in zip(layer_sizes[1:], layer_sizes[0:])]
weights = [np.random.standard_normal(l) for l in weight_sizes]
biases = [np.zeros((i, 1)) for i in layer_sizes[1:]]
