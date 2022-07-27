import torch
import torch.nn as nn
from torch import Tensor
from nlp.models.core import LinearTransition, ACT2FN
import torch.nn.functional as F


#
# class FeedForward(nn.Module):
#     def __init__(self, d_model=768, nx=768 * 4, activation="gelu", dropout=0.1):
#         super().__init__()
#
#         if activation not in ACT2FN:
#             raise ValueError("activation only support gelu, relu, swish, tanh, gelu_new")
#         self.c_fc = LinearTransition(d_model, nx)
#         self.c_proj = LinearTransition(nx, d_model)
#         self.act = ACT2FN[activation.lower()]
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class FeedForward(nn.Module):

    def __init__(self, d_model, middle_dim=2048):
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, middle_dim)
        self.fc2 = nn.Linear(middle_dim, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out
