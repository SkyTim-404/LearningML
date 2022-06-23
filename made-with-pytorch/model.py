import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            CNNBlock(1, 32, 2),
            CNNBlock(32, 128, 2),
            CNNBlock(128, 256, 1)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x)
        return self.fc_layers(x)
    
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers=1):
        super().__init__()
        self.convs = []
        if num_conv_layers == 1:
            self.convs.append(nn.Conv2d(in_channels, out_channels, 3))
        elif num_conv_layers == 2:
            self.convs.append(nn.Conv2d(in_channels, out_channels//num_conv_layers, 3))
            self.convs.append(nn.Conv2d(out_channels//num_conv_layers, out_channels, 3))
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
            x = self.leakyrelu(x)
        return F.max_pool2d(x, 2)
    
    
def test():
    model = Model()
    x = torch.randn((1, 28, 28))
    print(model(x).shape)
    
test()