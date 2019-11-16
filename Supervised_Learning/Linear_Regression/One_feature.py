# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 19:33:17 2019

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
    hyp = []
    for x in data:
        pred = theta[0][0] + (x * theta[0][1])
        hyp.append(pred)
    return hyp


# Error
def costs(prediction, y):
    err = []
    dJ_dT0 = []
    dJ_dT1 = []
    for p, y in zip(prediction, y):
        cost = (p - y)**2
        err.append(cost)
        d = p - y
        dJ_dT0.append(d)
        dJ_dT1.append(d*p)
    dJ_dT0 = sum(dJ_dT0) / len(dJ_dT0)
    dJ_dT1 = sum(dJ_dT1) / len(dJ_dT1)
    cost = sum(err) / len(err)
    return cost, dJ_dT0, dJ_dT1


# Training
def train(iterations, alpha):
    for i in iterations:
        prediction = hypothesis(X)
        cost, dJ_dT0, dJ_dT1 = costs(prediction, y)


# Test run of the prediction and plotting the results (currently random)
print(costs(hypothesis(X), y))

plt.scatter(hypothesis(X), y)
