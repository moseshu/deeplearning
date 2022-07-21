import torch
import torch.nn as nn
import copy
from embedding import PositionalEncoding
from encoder import EncoderLayer
from decoder import DecoderLayerChat


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncodeChat(nn.Module):



    def __init__(self, vocab_size, num_layers, units, d_model, num_heads, dropout=0.1):
        super(EncodeChat, self).__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, vocab_size)
        self.layers = clones(EncoderLayer(units, d_model, dropout, num_heads), num_layers)
        self.dropout_layer = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        embedding = self.positional_encoding(self.embedding(x))
        x = self.dropout_layer(embedding)
        for layer in self.layers:
            x = layer(x, mask)

        return x


class DecoderChat(nn.Module):

    def __init__(self, vocab_size,
                 num_layers,
                 units,
                 d_model,
                 num_heads=8,
                 dropout=0.1):
        super(DecoderChat, self).__init__()
        self.units = units
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.positional_embedding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=vocab_size)
        self.layers = clones(DecoderLayerChat(units, d_model, dropout, num_heads), num_layers)

    def forward(self, inputs, enc_outputs, look_ahead_mask=None, padding_mask=None):
        """
        inputs=[batch_size, seq_len]
        """
        embedding = self.positional_embedding(self.embedding(inputs))
        outputs = self.dropout(embedding)
        for layer in self.layers:
            outputs = layer(outputs, enc_outputs, look_ahead_mask, padding_mask)
        return outputs


class PadMasking(nn.Module):
    def __init__(self, pad_or_ahead="ahead", idx=0):
        """

        :param pad_or_ahead: 选择哪种masking的方式 只有以下两种方式
        """
        super(PadMasking, self).__init__()
        self.pad_or_ahead = pad_or_ahead
        self.idx = idx

    def create_padding_mask(self, x: torch.Tensor, idx=0):
        """
        input shape: [batch_size, seq_len]
        return [batch_size,1,1,seq_len]
        :param idx token 为PAD的id值
        """
        # zeros = torch.zeros_like(x)
        mask = torch.eq(x, idx).type(torch.float32)
        return mask[:, None, :]

    def create_look_ahead_mask(self, x, idx=0):
        """
        input_shape:[batch_size, seq_len]
        return : [batch_size, 1, seq_len, seq_len]
        掩盖后面的token
        """
        seq_len = x.shape[1]
        look_ahead_mask = 1 - torch.tril(torch.ones(seq_len, seq_len), diagonal=0)
        padding_mask = self.create_padding_mask(x, idx)
        return torch.maximum(look_ahead_mask, padding_mask)

    def forward(self, x):
        if self.pad_or_ahead == "pad":
            return self.create_padding_mask(x, self.idx)
        elif self.pad_or_ahead == "ahead":
            return self.create_look_ahead_mask(x, self.idx)



class Transformer(nn.Module):
    def __init__(self, vocab_size,
                 num_layers,
                 units,
                 d_model,
                 num_heads,
                 dropout):
        super(Transformer, self).__init__()
        self.padding_mask = PadMasking(pad_or_ahead="pad", idx=0)
        self.look_ahead_mask = PadMasking(pad_or_ahead="ahead", idx=0)
        self.encoder = EncodeChat(vocab_size=vocab_size,
                                  num_layers=num_layers,
                                  units=units,
                                  d_model=d_model,
                                  num_heads=num_heads,
                                  dropout=dropout)

        self.decoder = DecoderChat(vocab_size=vocab_size,
                                   num_layers=num_layers,
                                   units=units,
                                   d_model=d_model,
                                   num_heads=num_heads,
                                   dropout=dropout)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, dec_input):
        enc_padding_mask = self.padding_mask(inputs).bool()
        dec_look_ahead_mask = self.look_ahead_mask(dec_input).bool()
        dec_padding_mask = self.padding_mask(inputs).bool()
        enc_outputs = self.encoder(inputs, enc_padding_mask)
        dec_outputs = self.decoder(dec_input, enc_outputs, dec_look_ahead_mask, dec_padding_mask)
        outputs = self.linear(dec_outputs)
        return outputs



def loss_function(y_true, y_pred):
    """
    y_true:[batch_size,seq_len]
    y_pre: [batch_size,seq_len, classes]
    """
    crition = nn.CrossEntropyLoss()
    y_true = torch.squeeze(y_true.reshape(1, -1), dim=0)
    bs, seq, classes = y_pred.shape
    y_pred = y_pred.reshape(-1, classes)
    # print(y_pred.shape)

    loss = crition(y_pred, y_true)
    mask = y_true > 0
    total = torch.sum(mask)
    print(total)
    mask = mask.type(torch.float32)
    loss = mask * loss
    # print(loss.shape)
    mean_loss = torch.sum(loss) / total
    return mean_loss

if __name__ == '__main__':
    NUM_LAYERS = 2
    D_MODEL = 256
    NUM_HEADS = 8
    UNITS = 512
    DROPOUT = 0.1
    VOCAB_SIZE = 50000
    MAX_LEN = 40
    encode = EncoderLayer(512, 256, 0.1, 8)
    embedding = nn.Embedding(50000, 256)
    en = EncodeChat(50000, 4, 512, 256, 8, 0.1)
    decode = DecoderChat(50000, 4, 512, 256, 8, 0.1)
    input = torch.randint(0, 50000, (32, 10))
    din = torch.randint(0, 50000, (32, 10))

    transformer = Transformer(vocab_size=VOCAB_SIZE,
                              num_layers=NUM_LAYERS,
                              units=UNITS,
                              d_model=D_MODEL,
                              num_heads=NUM_HEADS,
                              dropout=DROPOUT)
    dein = embedding(din)
    decoderlayer = DecoderLayerChat(512, 256, dropout=0.1, heads=4)
    lmbda = PadMasking(pad_or_ahead="pad")

    ip = torch.tensor([[1, 2, 0, 3, 0], [0, 0, 0, 4, 5]])
    print(lmbda(ip).shape)

    out = transformer(input, din)
    print(out)