import torch.nn as nn
import torch
from torchtext.legacy.data import Field, BucketIterator
import numpy as np
from nlp.tokenizer.texttoken import Tokenizer, tokenizer_from_json, pad_sequences
import nltk
import json
import jieba
import random
import math
import time
from torchtext.data.metrics import bleu_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]

        out = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            out = layer(out, src_mask)

        # out = [batch size, src len, hid dim]

        return out


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        out = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return out


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch_size, query_len, hid_dim]
        # key = [batch_size, key_len, hid_dim]
        # value = [batch_size, value_len, hid_dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query_len, hid_dim]
        # K = [batch size, key_len, hid_dim]
        # V = [batch size, value_len, hid_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        # output = [batch size, trg len, output dim]

        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention
        # print("trg shape",trg.shape)
        # print("enc_src shape",enc_src.shape)
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention


en_data = "../translation/data/en-zh/en.txt"
zh_data = "../translation/data/en-zh/zh.txt"

en_json = "../tmp/token/words_en.json"
zh_json = "../tmp/token/words_zh.json"
with open(en_data) as f:
    src_en = f.read().splitlines()
with open(zh_data) as f:
    target_zh = f.read().splitlines()
print(src_en[:3])
print(target_zh[:3])
src_en = [" ".join(nltk.word_tokenize(i.lower())) for i in src_en]
target_zh = [" ".join(jieba.cut(str(s.replace(" ", "")), cut_all=False, HMM=True)) for s in target_zh]
print(target_zh[:3])

train_en = src_en[:-1000]
train_zh = target_zh[:-1000]
test_en = src_en[-1000:]
test_zh = target_zh[-1000:]

with open(zh_json) as f:
    data = json.load(f)
    tokenizer_zh = tokenizer_from_json(data)

with open(en_json) as f:
    data1 = json.load(f)
    tokenizer_en = tokenizer_from_json(data1)

id2word = tokenizer_zh.index_word
print("load data finished..")

SRC_VOCAB_SIZE = len(tokenizer_en.word_index) + 3
TRG_VOCAB_SIZE = len(tokenizer_zh.word_index) + 3
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
max_len = 128
enc = Encoder(SRC_VOCAB_SIZE,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device,
              max_len)

dec = Decoder(TRG_VOCAB_SIZE,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device,
              max_len)

SRC_PAD_IDX = 0
TRG_PAD_IDX = 0

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(initialize_weights)

LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# loss
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

# 英文句子id化
START_TOKEN_EN = len(tokenizer_en.word_index) + 1
END_TOKEN_EN = len(tokenizer_en.word_index) + 2
VOCAB_SIZE_EN = len(tokenizer_en.word_index) + 3
tokenized_inputs = tokenizer_en.texts_to_sequences(train_en)
test_tokenized_en = tokenizer_en.texts_to_sequences(test_en)
# 中文句子id化
START_TOKEN_ZH = len(tokenizer_zh.word_index) + 1
END_TOKEN_ZH = len(tokenizer_zh.word_index) + 2
VOCAB_SIZE_ZH = len(tokenizer_zh.word_index) + 3
tokenized_outputs = tokenizer_zh.texts_to_sequences(train_zh)
test_tokenized_zh = tokenizer_zh.texts_to_sequences(test_zh)
# train pad token sentences
tokenized_inputs = [[START_TOKEN_EN] + i + [END_TOKEN_EN] for i in tokenized_inputs]
tokenized_outputs = [[START_TOKEN_ZH] + i + [END_TOKEN_ZH] for i in tokenized_outputs]
# test
tokenized_inputs_en = [[START_TOKEN_EN] + i + [END_TOKEN_EN] for i in test_tokenized_en]
tokenized_outputs_zh = [[START_TOKEN_ZH] + i + [END_TOKEN_ZH] for i in test_tokenized_zh]
# train
tokenized_inputs = pad_sequences(tokenized_inputs, maxlen=max_len, padding="post", truncating="post")
tokenized_outputs = pad_sequences(tokenized_outputs, maxlen=max_len, padding="post", truncating="post")
# test
test_inputs_en = pad_sequences(tokenized_inputs_en, maxlen=max_len, padding="post", truncating="post")
test_outputs_zh = pad_sequences(tokenized_outputs_zh, maxlen=max_len, padding="post", truncating="post")
print("finished..")

from torch.utils.data import Dataset, DataLoader


class DataProcesser(Dataset):
    def __init__(self, first_seg, sencond_seg):
        super(DataProcesser, self).__init__()
        self.first_seg = first_seg
        self.sencond_seg = sencond_seg

    def __len__(self):
        return len(self.sencond_seg)

    def __getitem__(self, item):
        seg1 = self.first_seg[item]

        seg2 = self.sencond_seg[item]

        return (torch.tensor(seg1, dtype=torch.long),
                torch.tensor(seg2, dtype=torch.long))


# train
train_dataset = DataProcesser(tokenized_inputs, tokenized_outputs)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
# test
test_dataset = DataProcesser(test_inputs_en, test_outputs_zh)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=32)


def train(model, train_loader, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, (src, trg) in enumerate(train_loader):
        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader.dataset)


def evaluate(model, test_loader, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(test_loader):
            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(test_loader.dataset)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_loss=float('inf'), save_dir="./model"):
        self.best_valid_loss = best_valid_loss
        self.save_dir = save_dir

    def __call__(
            self, current_valid_loss,
            epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, '{}/best_model.pth'.format(self.save_dir))


save_best_model = SaveBestModel("/model_dir")

N_EPOCHS = 20
CLIP = 1

best_valid_loss = float('inf')

model = nn.DataParallel(model)
model.to(device)

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, test_loader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    save_best_model(valid_loss, epoch, model.module, optimizer, criterion)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


def translate_sentence(sentence, model, device, max_len=50):
    model.eval()

    sentence = " ".join(nltk.word_tokenize(sentence.lower()))
    sentence = [START_TOKEN_EN] + tokenizer_en.texts_to_sequences([sentence])[0] + [END_TOKEN_EN]

    src_tensor = torch.tensor(sentence, dtype=torch.long).unsqueeze(dim=0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_tensor = torch.tensor([START_TOKEN_ZH], dtype=torch.long).unsqueeze(dim=0).to(device)

    for i in range(max_len):

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_id = output.argmax(2)[:, -1].item()

        trg_tensor = torch.cat([output, pred_id], dim=-1)

        if pred_id == END_TOKEN_ZH:
            break

    predictions = trg_tensor.squeeze(0).cpu().numpy()
    predic_senc = [id2word[i] for i in predictions if i <= len(tokenizer_zh.word_index) and i > 1]
    return predic_senc





def calculate_bleu(src_data, trg_data, model, device, max_len=128):
    trgs = []
    pred_trgs = []

    for i, src in enumerate(src_data):
        trg = trg_data[i].split(" ")
        pred_trg = translate_sentence(src, model, device, max_len)

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)


bleu_score = calculate_bleu(test_en, test_zh, model, device)

print(f'BLEU score = {bleu_score * 100:.2f}')

