# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:10:54 2020

@author: Christian Zwinkels-Valero
"""

import numpy as np

y = np.load("y.npy")
outputs = []

for i in y:
    d = np.zeros((10, 1))
    d[i] = 1
    outputs.append(d)

np.save("labels.npy", outputs)
