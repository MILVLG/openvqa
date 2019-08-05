import torch.nn as nn
import torch
import math

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
