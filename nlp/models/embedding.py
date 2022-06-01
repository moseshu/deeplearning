import torch
import torch.nn as nn
from torch import Tensor


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




