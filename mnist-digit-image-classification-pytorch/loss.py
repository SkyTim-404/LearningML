import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        loss = 0
        for i in range(len(targets)):
            loss -= torch.log(predictions[i][targets[i]])
        return loss