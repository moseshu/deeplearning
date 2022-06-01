import copy
import torch
import torch.nn as nn
from torch import Tensor
from nlp.models.feedforward import FeedForward, ACT2FN
from typing import Optional
from nlp.models.embedding import PositionalEmbedding
from nlp.models.masking import PadFutureMask
from functools import partial
from typing import Optional, Tuple, List, Union
from nlp.models.core import LinearTransition
from nlp.attention.attention import AttentionLayer, Past


class TransformerBlock(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, d_model) or (..., seq_len, d_model),(k, v)
    ===========================================================================
    """

    def __init__(self, d_model, n_head=12, dropout=0.1, activation="gelu", rate=4):
        super(TransformerBlock, self).__init__()
        self.mlp = FeedForward(d_model=d_model, nx=rate * d_model, activation=activation)
        self.attn = AttentionLayer(heads=n_head, dims=d_model, dropout=dropout)
        self.ln_attn = nn.LayerNorm(d_model)
        self.ln_ff = nn.LayerNorm(d_model)

    def forward(self, x: Tensor,
                layer_past: Optional[Past] = None,
                mask: Optional[Tensor] = None):
        a = self.ln_attn(x)
        x, layer_past = self.attn(a, a, a, layer_past, mask)
        x = x + a
        x = x + self.mlp(self.ln_ff(x))
        return x if self.training else (x, layer_past)


class GTP2Model(nn.Module):
    def __init__(self, vocab_size, d_model,
                 n_layers,
                 idx=0,
                 max_len=1024,
                 heads=12, rate=4, dropout=0.1,
                 future=True):
        super(GTP2Model, self).__init__()
        block = TransformerBlock(d_model=d_model, n_head=heads, rate=rate, dropout=dropout, activation="gelu")
        self.transformers = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layers)])
        self.wpe = PositionalEmbedding(max_len, d_model)
        self.wte = nn.Embedding(vocab_size, d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

        self.future = PadFutureMask(idx=idx, max_len=max_len, future=future)

    def init_weights(self):
        self.out.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, LinearTransition)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, LinearTransition)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids: Tensor, labels=None, layer_past: Optional[List[Past]] = None,
                use_grad_ckpt: bool = False):
        past_len = layer_past[0][0].size(-2) if layer_past is not None else 0
        past_key = layer_past[-1][0] if layer_past is not None else None
        x = self.wte(input_ids) + self.wpe(input_ids, past_len)
        mask = self.future(input_ids, past_key)

        present = []
        for i, transformer in enumerate(self.transformers):
            if self.training and use_grad_ckpt:
                transformer = partial(torch.utils.checkpoint.checkpoint,
                                      transformer)
            x = transformer(x, layer_past[i] if layer_past is not None else None, mask)
            if not self.training:
                present.append(x[1])
                x = x[0]


        hidden_state = self.ln_f(x)
        logits = self.out(hidden_state)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            return loss
        return logits, present

