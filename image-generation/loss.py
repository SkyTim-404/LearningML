import torch
import torch.nn as nn

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predict_real, predict_fake):
        return -torch.mean(torch.log(predict_real+1e-8)+torch.log(1-predict_fake+1e-8))


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predict_fake):
        return -torch.mean(torch.log(predict_fake+1e-8))
    
    
class WassersteinCriticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predict_real, predict_fake):
        return -(torch.mean(predict_real) - torch.mean(predict_fake))


class WassersteinGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predict_fake):
        return -torch.mean(predict_fake)