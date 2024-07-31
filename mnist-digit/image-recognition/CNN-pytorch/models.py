import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            CNNBlock(1, 16, 3),
            CNNBlock(16, 16, 3),
            nn.MaxPool2d(2),
            CNNBlock(16, 32, 3),
            CNNBlock(32, 32, 3),
            nn.MaxPool2d(2),
            CNNBlock(32, 64, 3),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            FCBlock(64, 32),
            FCBlock(32, 32),
            nn.Linear(32, 10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.01)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        return x
    
class FCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        return x
    
    
def test():
    matrix = torch.randn(2, 3)
    print(matrix)
    
    
if __name__ == '__main__':
    test()