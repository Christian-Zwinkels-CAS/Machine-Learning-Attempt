# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:42:45 2019

@author: Christian Zwinkels-Valero
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Logistic function
def sigmoid(x):
    return 1 / (1 + np.exp(x))


# Importing the data
data = pd.read_csv("Admission_Predict.csv", header=None)
X = data[[0, 1]].to_numpy()
X = X.astype(np.float32)
X = np.insert(X, 0, 1, axis=1)
plot_1 = X[:, 1]
plot_2 = X[:, 2]
y = data[2].to_numpy()
y = y.astype(np.float32())

# Decision boundary
y = np.where( y >= 0.6, 1, 0)
color = {0: "red", 1: "blue"}

# Visualizing the data
for i in np.unique(y):
    ix = np.where(y == i)
    plt.scatter(plot_1[ix], plot_2[ix], c=color[i])
plt.xlabel("GRE Score")
plt.ylabel("CGPA")
plt.show()

# Hypothesis
thetas = np.random.standard_normal((X.shape[1], 1))

def hypothesis(data_in, parameters):
    p = np.dot(parameters.T, data_in.reshape((len(data_in), 1)))
    return sigmoid(p)
