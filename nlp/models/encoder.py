import torch.nn as nn
import torch

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

    def __init__(self, vcab_size, hidden_size,
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
        self.embedding = nn.Embedding(vcab_size, hidden_size)
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




