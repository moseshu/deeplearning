import torch.nn as nn
import torch
from torch import Tensor
from encoder import EncoderRNNS
from nlp.attention.attention import LauongAttention, AttentionLayer, Past



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

