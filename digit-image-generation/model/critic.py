import torch
import torch.nn as nn
from model.module import ConvBlock, ResidualConvBlock, LinearBlock

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            ResidualConvBlock(in_channels=16),
            ResidualConvBlock(in_channels=16),
            ConvBlock(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=2),
            ConvBlock(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            ResidualConvBlock(in_channels=32),
            ResidualConvBlock(in_channels=32),
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            ResidualConvBlock(in_channels=64),
            ResidualConvBlock(in_channels=64),
            nn.AdaptiveAvgPool2d((1, 1)), 
        ])
        self.linears = nn.ModuleList([
            LinearBlock(in_features=64, out_features=1), 
        ])
        
    
    def forward(self, x):
        for module in self.convs:
            x = module(x)
        x = torch.squeeze(x)
        for module in self.linears:
            x = module(x)
        return x