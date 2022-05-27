import torch
import torch.nn as nn
from encoder import EncoderRNNS
from decoder import BahdanauDecoder
from torch import Tensor
import random

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab,
                 targ_vocab,
                 hidden_enc_size,
                 hidden_dec_size,
                 targ_emb=128,
                 enc_layers=1,
                 dec_layers=1,
                 rnn_type="lstm",
                 dropout=0,
                 bidirectional=False,
                 teacher_forcing_prob=0.5,
                 device="cpu"
                 ):
        super(Seq2Seq, self).__init__()
        self.teacher_forcing_prob = teacher_forcing_prob
        self.device = device
        self.encoder = EncoderRNNS(vocab_size=src_vocab,
                                   hidden_size=hidden_dec_size,
                                   out_size=hidden_enc_size,
                                   n_layers=enc_layers,
                                   bidirectional=bidirectional,
                                   drop_prob=dropout,
                                   rnn_type=rnn_type)

        self.decoder = BahdanauDecoder(vocab_size=targ_vocab,
                                       emb_dim=targ_emb,
                                       hidden_enc_size=hidden_enc_size,
                                       hidden_dec_size=hidden_dec_size,
                                       n_layers=dec_layers,
                                       bidirectional=bidirectional,
                                       rnn_type=rnn_type,
                                       drop_prob=dropout
                                       )

    def forward(self, src_input: Tensor,
                target: Tensor,
                teacher_forcing_prob=0.5,
                src_mask: Tensor = None,
                targ_mask: Tensor = None) -> Tensor:
        """

        :param src_input: [batch, input_seq_len(200)]
        :param target: [batch, input_seq_len(200)]
        :return:
        """
        batch_size = src_input.size(0)
        hidden_init = self.encoder.init_hidden(batch_size, self.device)

        # ENCODER
        encoder_outputs, hidden_enc = self.encoder(src_input, hidden_init)

        # encoder  last hidden state is the first hidden state of decoder
        hidden_dec = hidden_enc
        targ_vocab = self.decoder.vocab_size
        tar_seq_len = target.size(1)
        final_outputs = torch.zeros(batch_size, targ_vocab, tar_seq_len).to(self.device)

        output = target[:, 0]
        for i in range(1, tar_seq_len, 1):
            output, hidden_dec, _ = self.decoder(output, hidden_dec, encoder_outputs)
            final_outputs[:, :, i] = output
            if random.random() < teacher_forcing_prob:
                output = target[:, i]
            else:
                output = output.max(1)[1]

        return final_outputs
