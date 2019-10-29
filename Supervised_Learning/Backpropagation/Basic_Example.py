"""
Created on Oct 27, 2019

Attempting to code an AI using the backpropagation algorithm

@author: Christian Zwinkels-Valero
"""
import numpy as np


# Input data
data_in = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])

# The expected outputs
data_out = np.array([0, 1, 1, 0])

e = 2.7182818284  # Just in case np.exp gives me an error


# This is the activation function (a sigmoid function)
def sigmoid(x):
    return 1/(1 + np.exp(-x))


layer_sizes = (2, 3, 1)

# Creates tuple that represent the weight matirices sizes
weight_sizes = [(y, x) for y, x in zip(layer_sizes[1:], layer_sizes[:-1])]

# Weights start out as random numbers from the standard distribution
weights = [np.random.standard_normal(s) for s in weight_sizes]

# Biases start out as zeros
biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]


def train(inp, out):
  # Chooses a random interger
  ri = np.random.randint(len(inp))

  # Sets the activation layer to a random sample from the input data
  a = inp[ri].reshape((layer_sizes[0], 1))

  # Feedforward
  layers = []  # A list to hold the values of the layers
  z = a
  for w, b in zip(weights, biases):
      z = np.dot(w, z) + b
      layers.append(z)
  layers.insert(0, a)  # Inserts the activation layer
  layers_sigmoid = [sigmoid(sig) for sig in layers]  # Applies sigmoid to them

  # Calculating the cost for each output perceptron
  cost_matrix = (layers_sigmoid[-1] - out[ri])**2  # Squared error function

  return layers_sigmoid[-1]

print(train(data_in, data_out))
