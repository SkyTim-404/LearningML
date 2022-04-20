import numpy as np

def labelToArray(y):
    actual = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        actual[i, y[i]] = 1
    return actual
    
def flatten(x):
    result = x.reshape((x.shape[0]*x.shape[1], 1))
    return result