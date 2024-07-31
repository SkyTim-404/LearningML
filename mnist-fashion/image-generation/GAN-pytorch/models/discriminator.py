import torch
import torch.nn as nn
import torch.nn.functional as F

from module import ConvBlock, ResidualConvBlock, LinearBlock

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding="same"),
            # ResidualConvBlock(in_channels=16),
            # ResidualConvBlock(in_channels=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=16, out_channels=32, kernel_size=3, padding="same"),
            # ResidualConvBlock(in_channels=32),
            # ResidualConvBlock(in_channels=32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, padding="same"),
            # ResidualConvBlock(in_channels=64),
            # ResidualConvBlock(in_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, padding="valid"),
        ])
        self.linears = nn.ModuleList([
            LinearBlock(in_features=128, out_features=32), 
            LinearBlock(in_features=32, out_features=1),
        ])
        
    
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = torch.flatten(x, start_dim=1)
        for linear in self.linears:
            x = linear(x)
        x = torch.sigmoid(x)
        return x