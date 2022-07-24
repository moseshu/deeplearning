import math

import jieba
import torch
import torch.nn as nn
import copy
from embedding import PositionalEncoding
from encoder import EncoderLayer
from decoder import DecoderLayerChat
from nlp.tokenizer.texttoken import pad_sequences
from nlp.dataprocess.dataset import text_generate, load_token


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncodeChat(nn.Module):

    def __init__(self, vocab_size, num_layers, units, d_model, num_heads, dropout=0.1):
        super(EncodeChat, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, vocab_size)
        self.layers = clones(EncoderLayer(units, d_model, dropout, num_heads), num_layers)
        self.dropout_layer = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        embedding = self.positional_encoding(self.embedding(x))
        embedding = math.sqrt(self.d_model) * embedding
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
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=vocab_size)
        self.layers = clones(DecoderLayerChat(units, d_model, dropout, num_heads), num_layers)

    def forward(self, inputs, enc_outputs, look_ahead_mask=None, padding_mask=None):
        """
        inputs=[batch_size, seq_len]
        """
        emb = self.embedding(inputs) * math.sqrt(self.d_model)
        embedding = self.positional_embedding(emb)
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
                 dropout,
                 training=True):
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
        self.trainging = training

    def forward(self, inputs, dec_input):
        enc_padding_mask = self.padding_mask(inputs).bool() if self.trainging else None
        dec_look_ahead_mask = self.look_ahead_mask(dec_input).bool() if self.trainging else None
        dec_padding_mask = self.padding_mask(inputs).bool() if self.trainging else None
        enc_outputs = self.encoder(inputs, enc_padding_mask)
        dec_outputs = self.decoder(dec_input, enc_outputs, dec_look_ahead_mask, dec_padding_mask)
        outputs = self.linear(dec_outputs)
        return outputs


def loss_function(y_true, y_pred):
    """
    y_true:[batch_size,seq_len]
    y_pre: [batch_size,seq_len, classes]
    """
    crition = nn.CrossEntropyLoss(reduction="none")

    y_true = y_true.contiguous().view(-1, 1).squeeze()
    bs, seq, classes = y_pred.shape
    y_pred = y_pred.contiguous().view(-1, classes)

    loss = crition(y_pred, y_true)
    mask = y_true > 0

    mask = mask.type(torch.float32)
    loss = mask * loss


    mean_loss = torch.mean(loss)
    return mean_loss


def accuracy(y_true, y_pre):
    label = torch.max(y_true, dim=-1)
    pre = torch.argmax(y_pre, dim=-1)
    return torch.equal(label, pre)


def train_model(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.988), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    model.train()
    for epoch in range(40):
        print('Epoch:', epoch, 'LR:', scheduler.get_last_lr())
        total = 0
        correct = 0
        total_loss = 0.0
        for i, (ques, answer, target) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(ques, answer)

            loss = loss_function(target, outputs)
            loss.backward()

            optimizer.step()

            total += target.size(0)
            total_loss += loss.item()
            pred_id = torch.argmax(outputs, dim=-1)
            correct += (pred_id == target).sum().item()
        mean_loss = round(total_loss / total, 5)
        accuracy = round(correct / total, 5)

        print("Epoch {} finished: total samples is:{}, loss is {} ,acc is {}".format(epoch, total, mean_loss, accuracy))
        print('-' * 20)
        scheduler.step()

    torch.save(model.state_dict(), "/Users/moses/workspace/data/model/chatbot.bin")


def load_model(model):
    model.load_state_dict(torch.load("/Users/moses/workspace/data/model/chatbot.bin"))
    return model


def evaluate(sentence, model, tokenizer, max_len=64):
    sentence = " ".join(jieba.cut(sentence, cut_all=False, HMM=True))
    START_TOKEN = len(tokenizer.word_index) + 1
    END_TOKEN = len(tokenizer.word_index) + 2
    sentence = [START_TOKEN] + tokenizer.texts_to_sequences([sentence])[0] + [END_TOKEN]
    print(sentence)
    sentence = torch.tensor(sentence, dtype=torch.long).unsqueeze(dim=0)
    print("===", sentence)
    output = torch.tensor([START_TOKEN], dtype=torch.long).unsqueeze(dim=0)
    model.eval()
    for i in range(max_len):
        predictions = model(sentence, output)
        predictions = predictions[:, -1:, :]

        pred_id = torch.argmax(predictions, dim=-1)
        print(pred_id)
        if pred_id.unsqueeze(0).item() == END_TOKEN:
            break
        output = torch.cat([output, pred_id], dim=-1)
        print(output)
    return output.squeeze(0)


def predict(sentence):
    prediction = evaluate(sentence)


if __name__ == '__main__':
    NUM_LAYERS = 3
    D_MODEL = 512
    NUM_HEADS = 8
    UNITS = 1024
    DROPOUT = 0.1
    MAX_LEN = 40
    EPOCH = 20
    learning_rate = 0.1
    dataloader, vocab_size = text_generate("../../data", "../../data/words.json", max_len=MAX_LEN)
    tokenizer = load_token("../../data/words.json")
    model = Transformer(vocab_size=vocab_size,
                        num_layers=NUM_LAYERS,
                        units=UNITS,
                        d_model=D_MODEL,
                        num_heads=NUM_HEADS,
                        dropout=DROPOUT, training=True)

    train_model(model)
    modl = load_model(model)

    # sentence = "你叫什么名字"
    # predictions = evaluate(sentence, modl, tokenizer, max_len=MAX_LEN)
    # predictions = predictions.numpy()
    # print(predictions)
    # id2word = {v: k for k, v in tokenizer.word_index.items()}
    # predic_senc = [id2word[i] for i in predictions if i < len(tokenizer.word_index) and i > 0]
    # print(predic_senc)
    # import tensorflow as tf
    # # # tf.keras.optimizers.schedules.LearningRateSchedule
    # # tf.keras.losses.SparseCategoricalCrossentropy
    # from chattf import create_padding_mask, create_look_ahead_mask
    #
    # da = [[1, 2, 0, 3, 0], [0, 0, 0, 4, 5]]
    # da1 = [[1, 2, 0, 4, 5]]
    # a = tf.constant(da)
    # a1 = tf.constant(da1)
    # print(create_padding_mask(a))
    # print(create_look_ahead_mask(a1))
    # padm = PadMasking("pad")
    # ah = PadMasking("ahead")
    # to=torch.tensor(da)
    # to1 = torch.tensor(da1)
    # print(padm(to))
    # print(ah(to1))