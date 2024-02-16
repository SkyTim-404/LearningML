import torch
import torch.nn as nn

class NLSLoss(nn.Module):
    def __init__(self, dim: int=None, reduction: str="mean"):
        super().__init__()
        self.__name__ = self.__class__.__name__
        self.logsoftmax = nn.LogSoftmax(dim)
        self.nll = nn.NLLLoss(reduction=reduction)
        
    def forward(self, pred, target):
        pred = self.logsoftmax(pred)
        target = torch.argmax(target, dim=1)
        return self.nll(pred, target)
    
    def __str__(self):
        return "loss"