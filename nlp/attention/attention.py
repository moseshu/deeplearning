import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor

Past = Tuple[Tensor, Tensor]


class BaseAttention(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    mask            bool            (..., query_len, kv_len)
    ---------------------------------------------------------------------------
    output          float           (..., query_len, dims)
    ===========================================================================
    """

    def __init__(self, dropout: float = 0.1, scale=True):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:

        x = torch.matmul(q, k.transpose(-2, -1))
        if self.scale: x = x / math.sqrt(k.size(-1))
        if mask is not None:
            x += mask.type_as(x) * x.new_tensor(-1e6)
        x = self.dropout(x.softmax(-1))

        return torch.matmul(x, v)


class MultiHeadAttention(BaseAttention):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    mask            bool            (..., query_len, kv_len)
    ---------------------------------------------------------------------------
    output          float           (..., query_len, dims)
    ===========================================================================
    """

    def __init__(self, heads: int, dropout: float = 0.1):
        super().__init__(dropout)
        self.heads = heads

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Split the tensors to multi-heads.
        q = q.view(q.size()[:-1] + (self.heads, q.size(-1) // self.heads))  # [batch_size, query_len, heads, dim]
        k = k.view(k.size()[:-1] + (self.heads, k.size(-1) // self.heads))  # [batch_size, key_len, heads, dim]
        v = v.view(v.size()[:-1] + (self.heads, v.size(-1) // self.heads))  # [batch_size, key_len, heads, dim]

        q = q.transpose(-3, -2)  # [batch_size, heads, query_len, dim]
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        if mask is not None:
            mask = mask.unsqueeze(-3)  # [batch_size,1, query_len, key_len]

        # Calculate multi-headed attentions and merge them into one.
        return (super().forward(q, k, v, mask)
                .transpose(-3, -2)
                .contiguous()
                .view(q.size()[:-3] + (q.size(-2), v.size(-1) * self.heads)))


class AttentionLayer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    past (*)        float           (..., past_len, dims)
    mask            bool            (..., query_len, past_len + kv_len)
    ---------------------------------------------------------------------------
    output 1        float           (..., query_len, dims)
    output 2 (*)    float           (..., past_len + kv_len, dims)
    ===========================================================================
    """

    def __init__(self, heads: int, dims: int, dropout: float = 0.1):
        super().__init__()
        d_head, remainder = divmod(dims, heads)
        if remainder:
            raise ValueError(" incompatible `dims` and `heads` ")
        self.attn = MultiHeadAttention(heads, dropout)
        self.proj_q = nn.Linear(dims, dims)
        self.proj_k = nn.Linear(dims, dims)
        self.proj_v = nn.Linear(dims, dims)
        self.linear = nn.Linear(dims, dims)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                past: Optional[Past] = None,
                mask: Optional[Tensor] = None
                ) -> Tuple[torch.Tensor, Past]:
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)

        # Reuse attention keys and values by concatenating to the current ones.
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)

        x = self.linear(self.attn(q, k, v, mask))
        return x, (k, v)


class LauongAttention(nn.Module):

    def __init__(self, hidden_enc_size, hidden_dec_size, method="concat"):
        """

        :param hidden_enc_size:
        :param hidden_dec_size:
        :param method:
         Note: this  attention class is used for seq2seq
        """
        super(LauongAttention, self).__init__()
        self.hidden_dec_size = hidden_dec_size
        self.hidden_enc_size = hidden_enc_size
        self.method = method
        self.V = nn.Parameter(torch.rand(hidden_dec_size))
        self.W = nn.Parameter(torch.rand(hidden_enc_size))
        if method == "concat":
            self.fc = nn.Linear(hidden_enc_size + hidden_dec_size, hidden_dec_size, bias=False)
        elif method == "dot":
            assert hidden_dec_size == hidden_enc_size
            self.fc = nn.Linear(hidden_enc_size, hidden_dec_size, bias=False)
        elif method == "general":
            self.fc = nn.Linear(hidden_dec_size, hidden_enc_size, bias=False)
        elif method == "sum":
            assert hidden_dec_size == hidden_enc_size
            self.fc = nn.Linear(hidden_enc_size, hidden_dec_size, bias=False)
        else:
            raise ValueError("check the method value ,method value is one of  concat,dot,general,sum ")

    def forward(self, decoder_hidden: Tensor, encoder_out: Tensor):
        """

        :param: decoder_hidden: [n_layers, batch, hidden_dec_size]
        :param: encoder_out: [batch, seq, hidden_enc_size]
        :return: att_weights: [batch, seq_len]
        """
        # Calculating Alignment Scores Note: encoder hidden can be used as decoder hidden
        decoder_hidden = decoder_hidden[-1]  # [batch, hidden_dec_size] the last layer hidden
        print(decoder_hidden.shape)
        batch_size = encoder_out.size(0)
        seq_len = encoder_out.size(1)
        if self.method == "concat":
            decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, hidden_dec_size]
            cat_hidden = torch.cat([decoder_hidden, encoder_out], dim=2)
            print(cat_hidden.shape)
            tanh_ws = torch.tanh(self.fc(cat_hidden))  # [batch_size, hidden_dec_size, seq_len]
            tanh_ws = tanh_ws.permute(0, 2, 1)
            V = self.V.repeat(batch_size, 1).unsqueeze(1)  # [batch_size, 1, hidden_dec_size]
            E = V.bmm(tanh_ws).squeeze(1)  # [batch_size, seq_len]
            att_weights = torch.softmax(E, dim=1)  # [batch_size, seq_len]
            # scores = att_weights.unsqueeze(1).bmm(encoder_out) #[batch_size, 1, hidden_enc_size]

        elif self.method == "dot":
            dec_hidden = decoder_hidden.unsqueeze(1)
            att_weights = dec_hidden.bmm(encoder_out.permute(0, 2, 1))
            att_weights = att_weights.squeeze(1)
            att_weights = torch.softmax(att_weights, dim=1)

        elif self.method == "general":
            dec_hidden = self.fc(decoder_hidden).unsqueeze(1)  # [batch,1, hidden_enc_size]
            att_weights = dec_hidden.bmm(encoder_out.permute(0, 2, 1))  # [batch, 1, seq_len]
            att_weights = att_weights.squeeze(1)
            att_weights = torch.softmax(att_weights, dim=1)
        elif self.method == "sum":
            decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
            out = self.fc(decoder_hidden + encoder_out)
            W = self.W.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1)
            att_weights = W.bmm(out.permute(0, 2, 1))
            att_weights = att_weights.squeeze(1)
            att_weights = torch.softmax(att_weights, dim=1)
        return att_weights  # [batch, seq_len]


class BaseAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, heads: int, dropout=0., bias=True, add_zero_attn=False):
        super(BaseAttentionLayer, self).__init__()
        self.multi_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads,
                                                dropout=dropout,
                                                bias=bias,
                                                add_zero_attn=add_zero_attn
                                                )

        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Shapes for inputs:
        - query: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(N, S, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(N, S, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.
          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length
        """
        query = self.fc_q(query)
        key = self.fc_q(key)
        value = self.fc_q(value)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        context_vector, attention_weights = self.multi_attn(query=query,
                                                            key=key,
                                                            value=value,
                                                            key_padding_mask=key_padding_mask,
                                                            need_weights=need_weights,
                                                            attn_mask=attn_mask
                                                            )
        context_vector = context_vector.transpose(0, 1)
        return context_vector
