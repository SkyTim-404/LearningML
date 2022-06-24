import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from dataset import *
from torch.utils.data import DataLoader

# hyper parameter
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
WEIGHT_DECAY = 0
LOAD_MODEL = False
LOAD_MODEL_FILE = ""

def train(model, train_dataset, loss_fn, optimizer, epochs):
    for epoch in range(epochs):
        for i in len(train_dataset):
            optimizer.zero_grad()
            input, label = train_dataset[i]
            input, label = input.to(DEVICE), label.to(DEVICE)
            output = model(input)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()


def main():
    model = Model().to(DEVICE)
    if LOAD_MODEL:
        pass
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    
    train_dataset = get_train_dataset()
    test_dataset = get_test_dataset()
    
    train(model, train_dataset, loss_fn, optimizer, EPOCHS)
    # train_dataloader = DataLoader(train_dataset, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, shuffle=True)
    
    
if __name__ == '__main__':
    main()