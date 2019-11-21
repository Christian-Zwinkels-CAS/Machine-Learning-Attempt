# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:01:03 2019

@author: Poopsickle123
"""
import csv
import numpy as np


def openCsv(file):
    a = open(file)
    b = [row for row in csv.reader(a)]
    c = [[b[i][-1]] for i in range(len(b))]
    for i in range(len(b)):
        if i > len(b):
            break
        del b[i][-1]
    X = np.array(b)
    y = np.array(c)
    return X, y


print(openCsv("Data_2.csv"))
