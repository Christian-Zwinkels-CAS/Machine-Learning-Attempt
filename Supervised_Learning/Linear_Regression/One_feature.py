# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:50:13 2019

@author: Christian Zwinkels-Valero
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
X /= np.max(X)
y = np.array(c, dtype=np.float32)
y /= np.max(y)

# Plotting the data
plt.scatter(X, y)
plt.show

# Hypothesis function
theta = np.abs(np.random.standard_normal((1, 2)))


def hypothesis(data):
    pred = (data * theta[0][1]) + theta[0][0]
    return pred


# Line plot function
def plotLine(parameters, x, y):
    x_plot = np.linspace(np.min(x), np.max(x))
    y_plot = parameters[0][0] + (parameters[0][1] * x_plot)
    plt.plot(x_plot, y_plot)
    return x_plot


# Calculating the costs
def costs(X, y):
    prediction = hypothesis(X)
    cost = 0
    dJ_dT0 = 0
    dJ_dT1 = 0
    for p, y in zip(prediction, y):
        cost += (p - y)**2
        dJ_dT0 += p - y
        dJ_dT1 += (p - y) * p
    cost /= len(X)
    dJ_dT0 /= len(X)
    dJ_dT1 /= len(X)
    return cost, dJ_dT0, dJ_dT1


# Training function
def train(iterations, alpha, X, y):
    for i in range(iterations):
        cost, dJ_dT0, dJ_dT1 = costs(X, y)
        theta[0][0] -= alpha * dJ_dT0
        theta[0][1] -= alpha * dJ_dT1


# Printing the costs and plotting the line
train(1500, 0.03, X, y)
print(costs(X, y))

plotLine(theta, X, y)
