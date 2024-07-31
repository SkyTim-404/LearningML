import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding="same", stride=1):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        x = self.norm(x)
        x = F.gelu(x)
        x = self.conv(x)
        return x


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels)
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        
    def forward(self, x):
        x = self.norm(x)
        x = F.gelu(x)
        x = self.conv_transpose(x)
        return x
    
    
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.down_conv = ConvBlock(in_channels, in_channels//4, kernel_size=1, padding="same")
        self.mid_conv = ConvBlock(in_channels//4, in_channels//4, kernel_size=3, padding="same")
        self.up_conv = ConvBlock(in_channels//4, in_channels, kernel_size=1, padding="same")
    
    def forward(self, x):
        y = self.down_conv(x)
        y = self.mid_conv(y)
        y = self.up_conv(y)
        return x + y
    
    
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(in_features)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, dropout=0):
        x = self.norm(x)
        x = F.gelu(x)
        x = self.linear(x)
        x = F.dropout(x, p=dropout)
        return x