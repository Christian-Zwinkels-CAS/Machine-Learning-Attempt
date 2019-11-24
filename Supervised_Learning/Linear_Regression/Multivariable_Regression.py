# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:01:03 2019

@author: Christian Zwinkels-Valero
"""
import csv
import numpy as np


# Function to open the file and turn it into a usable format
def openCsv(file):
    a = open(file)
    b = [row for row in csv.reader(a)]
    c = [[b[i][-1]] for i in range(len(b))]
    for i in range(len(b)):
        del b[i][-1]
        b[i].insert(0, 1)
    X = np.array(b, dtype=np.float64)
    y = np.array(c, dtype=np.float64)
    return X, y


# Applying basic feature scaling
X, y = openCsv("Data_2.csv")
X_max = []
for i in range(1, np.size(X, axis=1)):
    if i == 1:
        y_max = np.max(y)
        y /= y_max
    else:
        pass

    X_max.append(np.max(X[:, i]))
    X[:, i] /= X_max[i - 1]

thetas = np.abs(np.random.standard_normal((X.shape[-1], 1)))


def hypothethis(data_in):
    data_in[1:] / np.max(data_in[1:])
    pred = (np.dot(thetas.T, data_in.reshape((len(data_in), 1)))) * np.max(y)
    return pred


def costs(data_in, outputs):
    pred = []
    cost = []
    derivs = []
    for x, y in zip(data_in, outputs):
        pred.append(hypothethis(x))
        c = ((pred[-1] - y)**2)/2
        cost.append(c)
        d = (pred[-1] - y) * x
        derivs.append(d)
    cost = np.average(cost)
    derivs = np.average(derivs, axis=0)
    return cost, derivs.T


def train(data_in, parameters, outputs, iterations=1, alpha=0.1):
    for i in range(iterations):
        cost, derivs = costs(data_in, outputs)
        parameters -= alpha * derivs
    print("Error after training: {}".format(cost))
    return parameters


thetas = train(X, thetas, y, 1000, 0.1)
