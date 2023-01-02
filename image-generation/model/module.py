import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        return self.conv(self.leaky_relu(self.batch_norm(x)))


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        
    def forward(self, x):
        return self.conv_transpose(self.relu(self.batch_norm(x)))
    
    
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convs = nn.ModuleList([
            ConvBlock(in_channels, in_channels//4, kernel_size=1, padding=0), 
            ConvBlock(in_channels//4, in_channels//4, kernel_size=3, padding=1), 
            ConvBlock(in_channels//4, in_channels, kernel_size=1, padding=0), 
        ])
    
    def forward(self, x):
        identity = torch.clone(x)
        for module in self.convs:
            x = module(x)
        return identity + x
    
    
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(in_features)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(self.dropout(self.leaky_relu(self.batch_norm(x))))