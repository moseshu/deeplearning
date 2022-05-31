import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math


class LinearTransition(nn.Module):
    def __init__(self, embed_dim, out_dim):
        self.embed_dim = embed_dim
        self.out_dim = out_dim
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


class FutureMask(nn.Module):
    def __init__(self, idx=0, max_len=1024):
        super(FutureMask, self).__init__()
        self.max_len = max_len
        self.idx = idx

    def forward(self, query: Tensor, key: Tensor = None) -> Tensor:

        bias = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.uint8))
        query_len = query.size(-1)
        key_len = key.size(-2) if key is not None else query_len
        mask = bias[key_len - query_len:key_len, :key_len]
        mask = mask.repeat(query.size(0), 1, 1)
        return mask == 0



ACT2FN = {
    "relu": F.relu,
    "swish": Swish(),
    "gelu": F.gelu,
    "tanh": F.tanh,
    "glue_new": Gelu_New()
}


