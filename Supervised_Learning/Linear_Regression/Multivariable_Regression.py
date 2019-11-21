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
    for i in range(len(b)):
        b[i].insert(0, 1)
    c = [[b[i][-1]] for i in range(len(b))]
    for i in range(len(b)):
        del b[i][-1]
    X = np.array(b, dtype=np.float64)
    y = np.array(c, dtype=np.float64)
    return X, y


X, y = openCsv("Data_2.csv")
