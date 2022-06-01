import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math


class LinearTransition(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super(LinearTransition, self).__init__()
        self.fc = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class Conv1D(nn.Module):
    def __init__(self, nx, nf):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Swish(nn.Module):
    """
   Tensor          Type            Shape
   ===========================================================================
   input           float           (..., dims)
   ---------------------------------------------------------------------------
   output          float           (..., dims)
   ===========================================================================
   """

    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return x * self.sigmoid(x)


class Gelu_New(nn.Module):
    """
  Tensor          Type            Shape
  ===========================================================================
  input           float           (..., dims)
  ---------------------------------------------------------------------------
  output          float           (..., dims)
  ===========================================================================
  """

    def __init__(self):
        super(Gelu_New, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


ACT2FN = {
    "relu": F.relu,
    "swish": Swish(),
    "gelu": F.gelu,
    "tanh": F.tanh,
    "glue_new": Gelu_New()
}

