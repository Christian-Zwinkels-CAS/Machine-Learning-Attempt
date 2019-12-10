# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:42:45 2019

@author: Christian Zwinkels-Valero
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Importing the data
data = pd.read_csv("Admission_Predict.csv", header=None)
X = data[[0, 1]].to_numpy()
X = X.astype(np.float32)
plot_1 = X[:, 0]
plot_2 = X[:, 1]
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
