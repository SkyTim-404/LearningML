import torch
import torch.nn as nn

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predict_real, predict_fake):
        return torch.mean(-torch.log(predict_real)-torch.log(1-predict_fake))


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predict_fake):
        return torch.mean(-torch.log(predict_fake))