# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:01:03 2019

@author: Poopsickle123
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

    X_max.append(np.max(X[:, i]))
    X[:, i] /= X_max[i - 1]

thetas = np.abs(np.random.standard_normal((X.shape[-1], 1)))


def hypothethis(data_in):
    data_in / X_max
    pred = (np.dot(thetas.T, data_in)) * y_max
    return pred
