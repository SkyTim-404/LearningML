import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            CNNBlock(1, 16, 5),
            CNNBlock(16, 32, 5),
            CNNBlock(32, 64, 3)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.batchnorm(self.conv1(x)))
        return F.max_pool2d(x, 2)
    
    
def test():
    matrix = torch.randn(2, 3)
    print(matrix)
    
    
if __name__ == '__main__':
    test()