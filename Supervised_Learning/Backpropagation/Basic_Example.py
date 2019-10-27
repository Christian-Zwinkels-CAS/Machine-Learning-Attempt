import numpy as np


e = 2.7182818284  # Just in case np.exp gives me an error


# This is the activation function (a sigmoid function)
def sigmoid(x):
    return 1/(1 + np.exp(-x))


layer_sizes = (2, 3, 2)

# Creates tuple that represent the weight matirices sizes
weight_sizes = [(y, x) for y, x in zip(layer_sizes[1:], layer_sizes[:-1])]

# Weights start out as random numbers from the standard distribution
weights = [np.random.standard_normal(s) for s in weight_sizes]

# Biases start out as zeros
biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]

print(weights)
