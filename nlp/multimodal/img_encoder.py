import torch
import torch.nn as nn
from nlp.attention.attention import MultiHeadAttention


class TransformerBlock(nn.Module):

    def __init__(self, emb_dim, heads=4, dim_head=None, dim_linear_block=1024, dropout=0.1, activation=nn.GELU):
        """

        :param emb_dim: image embbeding dim
        :param heads:
        :param dim_head: custom dim of heads
        :param dim_linear_block:
        :param dropout:
        :param activation: GELU or GLU
        """
        super(TransformerBlock, self).__init__()
        self.emb_dim = emb_dim
        self.attention = MultiHeadAttention(heads=heads, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.norm_l1 = nn.LayerNorm(emb_dim)
        self.norm_l2 = nn.LayerNorm(emb_dim)
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, dim_linear_block),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, emb_dim),
            nn.Dropout(dropout)
        )
        self.dim_head = int(emb_dim / heads) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.to_qkv = nn.Linear(emb_dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, emb_dim, bias=False)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        assert x.dim() == 3 and x.shape[-1] == self.emb_dim  # [batch, tokens, _dim*3*heads]

        qkvx = self.to_qkv(x)
        q, k, v = torch.chunk(qkvx, chunks=3, dim=-1)

        att = self.attention(q, k, v, mask)  # [bs,tokens, _dim]
        y = x + self.W_0(att)
        return self.norm_l2(self.linear(y) + y)  # [bs,tokens, emb_dim]


class ImgTransformerEncoder(nn.Module):
    def __init__(self, emb_dim, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0):
        super(ImgTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(emb_dim, heads, dim_head, dim_linear_block, dropout) for _ in range(blocks)])

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)

        return x
