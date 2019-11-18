# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:50:13 2019

@author: Poopsickle123
"""
import numpy as np
import csv
from matplotlib import pyplot as plt

# Processing the data
a = open("data.csv")
b = [row for row in csv.reader(a)]
c = [[b[i][-1]] for i in range(len(b))]
for i in range(len(b)):
    del b[i][-1]

X = np.array(b, dtype=np.float32)
y = np.array(c, dtype=np.float32)

# Plotting the data
plt.scatter(X, y)
plt.show

# Hypothesis function
theta = np.random.standard_normal((1, 2))


def hypothesis(data):
    pred = (data * theta[0][1]) + theta[0][0]
    return pred


# Line plot function
def plotLine(parameters, x, y):
    x_plot = np.linspace(min(x), max(x))
    y_plot = parameters[0][0] + (parameters[0][1] * x_plot)
    plt.plot(x_plot, y_plot)
    return x_plot


# Calculating the costs
def costs(X, y):
    cost = 0
    for p, y in zip(X, y):
        cost += (hypothesis(p) - y)**2
    cost /= len(X)
    return cost


# Printing the costs and plotting the line
print(costs(X, y))

plotLine(theta, X, y)
