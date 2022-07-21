import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
import math
class PositionalEmbedding(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, embedding_dim)
    ===========================================================================
    """

    def __init__(self, pos_size, embd_dim):
        super(PositionalEmbedding, self).__init__()
        self.wpe = nn.Embedding(pos_size, embd_dim)

    def forward(self, x: Tensor, past_len=0):

        position_ids = torch.arange(past_len, x.size(-1) + past_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).view(-1, x.shape[-1]).expand_as(x)
        return self.wpe(position_ids)





class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
