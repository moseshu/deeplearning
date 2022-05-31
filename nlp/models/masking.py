import torch
import torch.nn as nn
from torch import Tensor

class PadFutureMask(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, key_len)
    ===========================================================================
    """

    def __init__(self, idx=0, max_len=1024, future=True):
        super(PadFutureMask, self).__init__()
        self.max_len = max_len
        self.idx = idx
        self.future = future
    def forward(self, query: Tensor, key: Tensor = None) -> Tensor:
        # future
        bias = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.uint8))
        query_len = query.size(-1)
        key_len = key.size(-2) if key is not None else query_len
        mask = bias[key_len - query_len:key_len, :key_len]
        mask = mask.repeat(query.size(0), 1, 1)

        # pad
        is_pad = (query == self.idx).unsqueeze(-2)

        key_pad = torch.zeros(query.size()[:-1] + (key_len - query_len,),dtype=torch.bool).unsqueeze(-2)
        pad_mask = torch.cat([is_pad, key_pad], dim=-1)

        pad_mask = pad_mask.repeat(1,query_len,1)

        if self.future:
            mask = pad_mask + (mask == 0)
        else:
            mask = pad_mask
        return mask
