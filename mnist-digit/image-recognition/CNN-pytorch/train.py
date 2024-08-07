import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

sys.path.append(r'../')

from dataset import Dataset
from models import CNN
from loss import Loss
from logger import Logger

# hyper parameter
LEARNING_RATE = 1e-3
DEVICE = "mps"
EPOCHS = 1
BATCH_SIZE = 60
WEIGHT_DECAY = 1e-4
LOAD_MODEL = False
MODEL_FILE = "model.pth"

def train(model, train_dataloader, loss_fn, optimizer):
    model.train(True)
    losses = []
    for batch_idx, (input, label) in enumerate(train_dataloader):
        input = normalize_img(input)
        input = input.to(torch.float32)
        label = torch.squeeze(F.one_hot(label, num_classes=10), 0).to(torch.float32)
        input, label = input.to(DEVICE), label.to(DEVICE)
        output = model(input)
        loss = loss_fn(output, label)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        print(f"batch: {batch_idx+1}/{len(train_dataloader)}, loss: {loss.item()}", end="\r")
    return torch.tensor(losses).mean()


def evaluate(model, test_dataloader):
    model.train(False)
    with torch.no_grad():
        inputs, lebels = next(iter(test_dataloader))
        inputs = normalize_img(inputs)
        inputs, lebels = inputs.to(DEVICE), lebels.to(DEVICE)
        outputs = model(inputs)
        outputs = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(outputs == lebels)/len(test_dataloader.dataset)
    return accuracy


def normalize_img(img):
    mean = img.mean((2, 3), keepdim=True)
    std = img.std((2, 3), keepdim=True)
    normalize = transforms.Normalize(mean, std)
    img = normalize(img)
    return img


def main():
    model = CNN().to(DEVICE)
    if LOAD_MODEL:
        model = torch.load(MODEL_FILE)
    model = torch.nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    
    train_dataset = Dataset.get_train_dataset(r"../../data")
    test_dataset = Dataset.get_test_dataset(r"../../data")
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    for epoch in range(EPOCHS):
        avg_loss = train(model, train_dataloader, loss_fn, optimizer)
        accuracy = evaluate(model, test_dataloader)
        Logger.training_log(epoch, EPOCHS, avg_loss, accuracy)
        
    torch.save(model, MODEL_FILE)
    
    
def test():
    img = torch.randn((60, 1, 28, 28))
    img = normalize_img(img)
    print(img)
    
    
if __name__ == '__main__':
    main()