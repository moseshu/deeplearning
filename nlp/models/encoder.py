import torch.nn as nn
import torch
from nlp.attention.attention import MultiHeadAttention
from nlp.models.embedding import PositionalEmbedding
import copy

"""
@author Moses
@email moseshu25@gmail.com
@description
"""


class EncoderRNNS(nn.Module):
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(self, vocab_size, hidden_size,
                 out_size,
                 n_layers=1,
                 drop_prob=0,
                 bidirectional=False,
                 rnn_type="lstm"):
        """

        :param input_size: vocab size
        :param hidden_size: word embedding dim or rnn cess input_dim
        :param out_size rnn out_put dim
        :param n_layers:
        :param drop_prob:
        :param bidirectional
        :param rnn_type lstm,gru,rnn

        Examples::
            >> lstm = lstm = EncoderRNNS(input_size=10, hidden_size=64, out_size=128, n_layers=2, rnn_type="rnn", bidirectional=True)
            >> input = torch.randint(0, 8, (3, 5))
            >> hidden = lstm.init_hidden(3)
            >> output, hidden = lstm(input, hidden)
            >> print(output.shape)
        """
        super(EncoderRNNS, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.out_size = out_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=drop_prob,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(in_features=hidden_size << 1 if self.bidirectional else hidden_size,
                            out_features=out_size,
                            bias=False)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, inputs, hidden=None):
        # Embed input words
        embedded = self.embedding(inputs)  # batch,seq,embedding_dim
        # print(embedded.shape)
        # Pass the embedded word vectors into LSTM and return all outputs

        output, hidden = self.rnn(embedded, hidden)

        output = self.dropout(self.fc(output))
        return output, hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        (h0, c0) = (
            torch.randn(self.n_layers << 1 if self.bidirectional else self.n_layers, batch_size, self.hidden_size,
                        device=device),
            torch.randn(self.n_layers << 1 if self.bidirectional else self.n_layers, batch_size, self.hidden_size,
                        device=device))

        if self.rnn_type == "lstm":
            return (h0, c0)
        return h0


class EncoderLayer(nn.Module):
    def __init__(self, units, d_model, dropout=0.1, heads=8):
        """

        :param units: 线性变换中间过程
        :param d_model: 最终的输出的向量维度
        :param dropout:
        :param heads: 多头数
        """
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(heads, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.linear1 = nn.Linear(d_model, units)
        self.linear2 = nn.Linear(units, d_model)
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):

        attention = self.self_attention(self.Q(x), self.K(x), self.V(x), mask=mask)
        attention = self.drop(attention)
        attention = torch.add(x, attention)
        attention = self.norm_1(attention)

        outputs = self.relu(self.linear1(attention))
        outputs = self.drop(self.linear2(outputs))
        outputs = self.norm_2(torch.add(attention, outputs))
        return outputs


class TextEncoder(nn.Module):

    def __init__(self, vocab_size, num_layers, units, d_model, num_heads, max_seq, dropout=0.1):
        super(TextEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEmbedding(max_seq, d_model)
        self.layers = self.clones(EncoderLayer(units, d_model, dropout, num_heads), num_layers)
        self.dropout_layer = nn.Dropout(dropout)

    def clones(self, module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, x: torch.Tensor, mask=None):
        # x.shape = [bs, max_seq]
        embedding = self.positional_encoding(x) + self.embedding(x)  # [bs,max_seq,d_model]
        x = self.dropout_layer(embedding)
        for layer in self.layers:
            x = layer(x, mask)

        return x


if __name__ == '__main__':
    # lstm = EncoderRNNS(vcab_size=10, hidden_size=64, out_size=128, n_layers=2, rnn_type="gru", bidirectional=False)
    # input = torch.randint(0, 8, (3, 5))
    #
    # hidden = lstm.init_hidden(3)
    # print(hidden.shape)
    # output, hidden1 = lstm(input, hidden)
    # print(hidden1.shape)

    x = torch.randint(0,100,(3,7))
    # encoder = EncoderLayer(units=120, d_model=768, heads=8)
    encoder = TextEncoder(vocab_size=99,num_layers=4,num_heads=8,units=128,d_model=768,max_seq=7)
    a = encoder(x)
    print(a.shape)
