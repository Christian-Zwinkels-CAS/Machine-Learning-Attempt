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


# Vizualize the data
plt.scatter(X, y)
line(X, y)
plt.show()
