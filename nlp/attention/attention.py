import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor
Past = Tuple[torch.Tensor, torch.Tensor]


class LauongAttention(nn.Module):
    def __init__(self, hidden_enc_size, hidden_dec_size, method="concat"):
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
