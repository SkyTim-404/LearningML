import numpy as np

def ReLU(x):
    return np.maximum(0, x)

def dReLU(x):
    return (x > 0).astype(int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
