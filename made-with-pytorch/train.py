import torch
import torch.nn as nn
import torch.optim as optim
from model import Model

learning_rate = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"
epoch = 10

model = Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train():
    pass