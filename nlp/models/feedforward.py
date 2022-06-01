import torch
import torch.nn as nn
from torch import Tensor
from core import LinearTransition
import torch.nn.functional as F
from core import ACT2FN


class FeedForward(nn.Module):
    def __init__(self, d_model=768, nx=768 * 4, activation="gelu", dropout=0.1):
        super().__init__()

        if activation not in ACT2FN:
            raise ValueError("activation only support gelu, relu, swish, tanh, gelu_new")
        self.c_fc = LinearTransition(d_model, nx)
        self.c_proj = LinearTransition(nx, d_model)
        self.act = ACT2FN[activation.lower()]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))
