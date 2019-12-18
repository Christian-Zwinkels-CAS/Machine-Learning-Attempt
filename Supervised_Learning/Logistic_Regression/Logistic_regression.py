import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Logistic function
def sigmoid(x):
    return 1 / (1 + np.exp(x))


# Importing the data
data = pd.read_csv("Admission_Predict.csv", header=None, skiprows=1)
X = data[[0, 1]].to_numpy()
X = X.astype(np.float32)
for i in range(len(X)):
    X[i] = (X[i] - np.mean(X, axis=0)) / np.ptp(X, axis=0)
X = np.insert(X, 0, 1, axis=1)
plot_1 = X[:, 1]
plot_2 = X[:, 2]
y = data[2].to_numpy()
y = y.astype(np.float32())

# Decision boundary
y = np.where( y >= 0.73, 1, 0)
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


# Cost and derivatives computations
def costs(data_in, ouputs, parameters):
    cost = []
    derivs = []
    for x, y in zip(data_in, ouputs):
        p = hypothesis(x, parameters)
        c = y * np.log(p) + (1 - y) * np.log(1 - p)
        d = (p - y) * x
        cost.append(c)
        derivs.append(d)
    cost = np.mean(cost) * -1
    derivs = np.mean(derivs, axis=0)
    return cost, derivs


# Training function
def train(data_in, outputs, parameters, iterations=1, alpha=0.2):
    change = parameters.T
    p = []
    x = []
    for i in range(iterations):
        Cs, Ds = costs(data_in, outputs, parameters)
        p.append(Cs)
        x.append(i)
        change += alpha * Ds
    print(Cs)
    plt.plot(x, p)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    return change.T

    
thetas = train(X, y, thetas, 400, 1.9)
