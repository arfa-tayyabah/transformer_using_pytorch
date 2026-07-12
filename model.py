import torch 
import torch.nn as nn
import math
class NormalizationLayer(nn.Module):
    def __init__(self, features: int, eps = 10**-6):
        super.__init__()
        self.eps = eps
        self.alpha = nn.Parameters(torch.ones(features))
        self.bias = nn.Parameters(torch.zeros(features))
    
    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x - mean)/(std + self.eps) + self.bias



