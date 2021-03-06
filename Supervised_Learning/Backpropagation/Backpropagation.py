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


# Derivative of the activation function
def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


layer_sizes = (2, 2, 1)

# Creates tuple that represent the weight matirices sizes
weight_sizes = [(y, x) for y, x in zip(layer_sizes[1:], layer_sizes[:-1])]

# Weights start out as random numbers from the standard distribution
weights = [np.random.standard_normal(s) for s in weight_sizes]

# Biases start out as zeros
biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]


def train(inp, out, iterations, alpha):
    for i in range(iterations):
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
        layers_sigmoid = [sigmoid(sig) for sig in layers]  # Applies sigmoid
        layers_sigmoid.insert(0, a)

        # Calculating the cost for each output perceptron
        cost_matrix = layers_sigmoid[-1] - out[ri]  # Error

        # Calculating delta_L (the last perceptron layer error)
        delC_dela = 2 * cost_matrix  # Derivative of the squared error function
        dela_delz = deriv_sigmoid(layers[-1])
        delta_L = delC_dela * dela_delz

        # Calculating the delta for all layers
        delta_l = []
        delta_l.append(delta_L)  # We need the last layer delta for calculations

        for l in range(1, len(layer_sizes) - 1):
            d = -l - 1  # This is to make sure that l is never 0
            if d == 0:  # Otherwise the calculations get messed up
                break

            g = np.dot(weights[-l].T, delta_l[0]) * deriv_sigmoid(layers[-l-1])
            delta_l.insert(0, g)  # Adds the calculated deltas into the list

        # Calculating delz_delw
        delz_delw = []
        for l in range(1, len(layer_sizes)):
            r = layers_sigmoid[-l - 1].T * np.ones((weight_sizes[-l]))
            delz_delw.insert(0, r)

        # Calculating delC_delw
        delC_delw = [np.zeros(s) for s in weight_sizes]
        for i in range(len(delta_l)):
            delC_delw[i] = delta_l[i] * delz_delw[i]

        # Update the weights and baises
        for i in range(len(weights)):
            weights[i] += -alpha * delC_delw[i]
            biases[i] += -alpha * delta_l[i]

    return a, layers_sigmoid[-1]


print(train(data_in, data_out, 5000, 0.3))
