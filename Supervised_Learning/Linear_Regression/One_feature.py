# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:17:32 2019

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
X_max = np.max(X)
X /= X_max
y = np.array(c, dtype=np.float32)
y_max = np.max(y)
y /= y_max

# Hypothesis
theta = np.random.standard_normal((1, 2))


def hypothesis(data_in):
    data_in /= np.max(X)
    pred = theta[0][0] + (theta[0][1] * data_in)
    return pred * np.max(y)


# Graph function
def line(Xs, ys):
    x_plot = np.linspace(np.min(Xs), np.max(Xs), num=len(Xs))
    y_plot = []
    for i in x_plot:
        p = hypothesis(i)
        y_plot.insert(-1, p)
    x_plot = np.delete(x_plot, len(x_plot) - 1)
    del y_plot[-1]
    plt.plot(x_plot, y_plot)


# Cost calculations
def costs(data_in, outputs):
    p = []
    cost = []
    d0 = []
    d1 = []
    for i in range(len(data_in)):
        pred = hypothesis(data_in[i])
        p.append(pred)
        c = p[-1] - outputs[i]
        cost.append(0.5 * c**2)
        d0.append(c)
        d1.append(c * data_in[i])
    cost = np.mean(cost)
    d0 = np.mean(d0)
    d1 = np.mean(d1)
    return cost, d0, d1


# Train
def train(data_in, outputs, iterations=1, alpha=0.5):
    for i in range(iterations):
        c, d0, d1 = costs(data_in, outputs)
        theta[0][0] -= alpha * d0
        theta[0][1] -= alpha * d1
    return 0


# Vizualize the data
train(X, y, 1000, 0.2)
#y *= y_max
#X *= X_max
print(costs(X, y))
plt.scatter(X, y)
line(X, y)
plt.show()
