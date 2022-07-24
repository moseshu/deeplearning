import torch.nn as nn
import torch
from torch import Tensor
from encoder import EncoderRNNS
from nlp.attention.attention import LauongAttention, AttentionLayer, Past, MultiHeadAttention
from embedding import PositionalEncoding
import copy


class BahdanauDecoder(nn.Module):
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(self, vocab_size, emb_dim, hidden_enc_size, hidden_dec_size,
                 n_layers=1,
                 rnn_type='lstm',
                 drop_prob=0,
                 bidirectional=False):
        """

        :param vocab_size: 词的大小
        :param emb_dim: 词向量纬度大小
        :param hidden_enc_size:
        :param hidden_dec_size:
        :param n_layers:
        :param rnn_type:
        :param drop_prob:
        :param bidirectional:
        References
            - https://github.com/lukysummer/Bahdanau-Attention-in-Pytorch/blob/master/Encoder.py,
            - https://blog.floydhub.com/attention-mechanism/
        """
        super(BahdanauDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attention = LauongAttention(hidden_enc_size, hidden_dec_size)
        self.rnn_type = rnn_type.lower()
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=emb_dim + hidden_enc_size,
            hidden_size=hidden_dec_size,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=drop_prob,
            bidirectional=bidirectional,
        )
        self.classfier = nn.Linear(emb_dim + hidden_enc_size + hidden_dec_size, vocab_size)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, target: Tensor, hidden_dec, encoder_out: Tensor):
        """

        :param target: [batch]
        :param hidden_dec: rnn,gru shape is [n_layers, batch, hidden_dec_size],lstm shape is (hn, cn)
        :param encoder_out: [batch, seq_len, hidden_enc_size]
        :return:
        """
        ######################## 1. TARGET EMBEDDINGS #########################
        target = target.unsqueeze(1)  # [b, 1] : single word

        embedded_trg = self.embedding(target)  # [batch, 1, emb_size]

        ################## 2. CALCULATE ATTENTION WEIGHTS #####################
        if self.rnn_type == "lstm":
            # [batch, seq_len]
            att_weights = self.attention(hidden_dec[0], encoder_out)  # hidden_enc shape is (hn, cn)
        else:
            att_weights = self.attention(hidden_dec, encoder_out)
        att_weights = att_weights.unsqueeze(1)  # [batch, 1, eq_len]

        ###################### 3. CALCULATE WEIGHTED SUM ######################
        context_vector = att_weights.bmm(encoder_out)  # [batch, 1, hidden_enc_size]

        ############################# 4. RNN or GRU or LSTM LAYER ############################
        rnn_input = torch.cat([embedded_trg, context_vector], dim=2)
        dec_out, hiden_dec = self.rnn(rnn_input, hidden_dec)
        # dec_out shape is [batch, 1, hidden_dec_size]
        final_input = torch.cat([embedded_trg.squeeze(1),
                                 context_vector.squeeze(1),
                                 dec_out.squeeze(1)
                                 ], dim=1)
        output = self.classfier(final_input)  # [batch, target_vocab_size]
        return output, hiden_dec, att_weights


class DecoderLayerChat(nn.Module):
    def __init__(self, units, d_model, dropout=0.1, heads=8):
        super(DecoderLayerChat, self).__init__()
        self.attention1 = AttentionLayer(heads, d_model, dropout=dropout)
        self.attention2 = AttentionLayer(heads, d_model, dropout=dropout)
        self.units = units
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.linear1 = nn.Linear(d_model, units)
        self.linear2 = nn.Linear(units, d_model)

    def forward(self, inputs, enc_outputs, look_ahead_mask=None, padding_mask=None):
        attention1, _ = self.attention1(q=inputs, k=inputs, v=inputs, mask=look_ahead_mask)
        attention1 = self.norm(torch.add(attention1, inputs))  # 参差连接

        attention2, _ = self.attention2(q=attention1, k=enc_outputs, v=enc_outputs,
                                        mask=padding_mask)  # decoder的输入跟encoder的输出做attention
        attention2 = self.dropout(attention2)
        attention2 = self.norm(attention2 + attention1)
        outputs = self.relu(self.linear1(attention2))
        outputs = self.dropout(self.linear2(outputs))
        outputs = self.norm(outputs + attention2)
        return outputs
