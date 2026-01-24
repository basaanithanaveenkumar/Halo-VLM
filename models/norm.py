import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps_non_zero = eps
        #self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        root_mean_sqr = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps_non_zero)
        # Normalize and scale
        x_norm = x / root_mean_sqr
        return x_norm #self.weight * x_norm