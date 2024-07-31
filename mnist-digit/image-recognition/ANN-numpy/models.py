import numpy as np
import mathlib
import utils

class ANN:
    def __init__(self):
        self.W1 = (np.random.rand(128, 784) - 0.5) * 0.1
        self.b1 = (np.random.rand(128, 1) - 0.5) * 0.1
        self.W2 = (np.random.rand(128, 128) - 0.5) * 0.1
        self.b2 = (np.random.rand(128, 1) - 0.5) * 0.1
        self.W3 = (np.random.rand(10, 128) - 0.5) * 0.1
        self.b3 = (np.random.rand(10, 1) - 0.5) * 0.1
        
    def print(self):
        print(f"W1: \n{self.W1}")
        print(f"b1: \n{self.b1}")
        print(f"W2: \n{self.W2}")
        print(f"b2: \n{self.b2}")
        print(f"W3: \n{self.W3}")
        print(f"b3: \n{self.b3}")
        
    def predict(self, X):
        y = np.zeros((X.shape[0],), dtype=int)
        for i in range(X.shape[0]):
            input = utils.flatten(X[i])
            hlayer1, hlayer2, predicted = self.forward(input)
            y[i] = np.argmax(predicted)
        return y
    
    def train(self, X, y, learning_rate, epoch):
        actual = utils.labelToArray(y)
        for e in range(epoch):
            for i in range(X.shape[0]):
                input = utils.flatten(X[i])
                hlayer1, hlayer2, predicted = self.forward(input)
                self.backward(input, hlayer1, hlayer2, predicted, actual[i], learning_rate)
            print(f"Epoch {e + 1} completed")
    
    def forward(self, input):
        hlayer1 = mathlib.ReLU(self.W1.dot(input) + self.b1)
        hlayer2 = mathlib.ReLU(self.W2.dot(hlayer1) + self.b2)
        predicted = mathlib.sigmoid(self.W3.dot(hlayer2) + self.b3)
        return hlayer1, hlayer2, predicted
    
    def backward(self, input, hlayer1, hlayer2, predicted, actual, learning_rate):
        actual = actual.reshape((actual.shape[0], 1))
        update3 = (predicted - actual) * predicted * (1 - predicted)
        delta_W3 = learning_rate * update3 * hlayer2.T
        delta_b3 = learning_rate * update3
        update2 = mathlib.dReLU(hlayer2) * self.W3.T.dot(update3)
        delta_W2 = learning_rate * update2 * hlayer1.T
        delta_b2 = learning_rate * update2
        update1 = mathlib.dReLU(hlayer1) * self.W2.T.dot(update2)
        delta_W1 = learning_rate * update1 * input.T
        delta_b1 = learning_rate * update1
        self.W1 -= delta_W1
        self.b1 -= delta_b1
        self.W2 -= delta_W2
        self.b2 -= delta_b2
        self.W3 -= delta_W3
        self.b3 -= delta_b3
        
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = y_pred[y_pred == y_test].size / y_test.size
        return accuracy
    
    def save(self):
        np.savez("model.npz", W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)
    
    def load(self):
        data = np.load("model.npz")
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.W3 = data["W3"]
        self.b3 = data["b3"]
        