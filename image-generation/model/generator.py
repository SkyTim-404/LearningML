import torch.nn as nn
from model.module import ConvBlock, ConvTransposeBlock

class Generator(nn.Module):
    def __init__(self, latent_features):
        super().__init__()
        self.latent_features = latent_features
        self.linear = nn.Linear(self.latent_features, 6*6*64)
        self.convs = nn.ModuleList([
            ConvTransposeBlock(in_channels=64, out_channels=32, kernel_size=3, stride=2), 
            ConvTransposeBlock(in_channels=32, out_channels=16, kernel_size=4, stride=2), 
            ConvBlock(in_channels=16, out_channels=1, kernel_size=3, padding=1), 
        ])
        self.last_activation = nn.Tanh()
    
    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 64, 6, 6)
        for module in self.convs:
            x = module(x)
        x = self.last_activation(x)
        return x