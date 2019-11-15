# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 19:33:17 2019

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
def cost(prediction, y):
    err = []
    for p, y in zip(prediction, y):
        cost = (p - y)**2
        err.append(cost)
    return sum(err) / len(err)


# Test run of the prediction and plotting the results (currently random)
prediction = hypothesis(X)
print(cost(prediction, y))

plt.scatter(prediction, y)
