# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:11:43 2020

@author: Christian Zwinkels-Valero

MNIST Dataset: http://yann.lecun.com/exdb/mnist/
Note that the features file is too big to be uploaded to GitHub
"""

import gzip
import numpy as np

img_size = 28  # Images are 28x28 pixels
num_image = 60000  # Return this amount of images

f = gzip.open("train-images-idx3-ubyte.gz", "r")
f.read(16)  # Firts 16 bytes are not needed
buffer = f.read(img_size**2 * num_image)  # Return up to specified image number
X = np.frombuffer(buffer, dtype=np.uint8())
X = X.reshape(num_image, img_size, img_size)

o = gzip.open("train-labels-idx1-ubyte.gz", "r")
o.read(8)
buffer_o = o.read(num_image)
y = np.frombuffer(buffer_o, dtype=np.uint8())

np.save("X.npy", X)
np.save("y.npy", y)
