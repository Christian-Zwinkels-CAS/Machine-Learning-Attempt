# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 19:33:17 2019

@author: Poopsickle123
"""
import numpy as np
import csv
from matplotlib import pyplot as plt

# Processing the csv
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
