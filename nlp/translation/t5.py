from transformers.models.t5 import T5ForConditionalGeneration, T5Config, T5Model,T5Tokenizer
import torch.nn as nn
import jieba

with open("/workspace/en-zh/en-zh.en") as f:
    en_lines = f.read().splitlines()
with open("/workspace/en-zh/en-zh.zh") as f:
    zh_lines = f.read().splitlines()
print(zh_lines[:3])
print(en_lines[:3])
zh_tgt = [" ".join(jieba.cut(str(s.replace(" ","")), cut_all=False, HMM=True)) for s in zh_lines]
print(zh_tgt[:3])

from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import json
import torch
train_en = en_lines[:-1000]
train_zh = zh_tgt[:-1000]
test_en = en_lines[-1000:]
test_zh = zh_tgt[-1000:]
with open("/workspace/token/words_zh.json") as f:
    data = json.load(f)
    tokenizer_zh = tokenizer_from_json(data)

with open("/workspace/token/words_en.json") as f:
    data1 = json.load(f)
    tokenizer_en = tokenizer_from_json(data1)

id2word = {v: k for k, v in tokenizer_zh.word_index.items()}

max_len=40
#英文句子id化
START_TOKEN_EN = len(tokenizer_en.word_index) + 1
END_TOKENN_EN = len(tokenizer_en.word_index) + 2
VOCAB_SIZE_EN = len(tokenizer_en.word_index) + 3
tokenized_inputs = tokenizer_en.texts_to_sequences(train_en)
#中文句子id化
START_TOKEN_ZH = len(tokenizer_zh.word_index) + 1
END_TOKENN_ZH = len(tokenizer_zh.word_index) + 2
VOCAB_SIZE_ZH = len(tokenizer_zh.word_index) + 3
tokenized_outputs = tokenizer_zh.texts_to_sequences(train_zh)

    # pad token sentences
tokenized_inputs = [[START_TOKEN_EN] + i + [END_TOKENN_EN] for i in tokenized_inputs]
tokenized_outputs = [[START_TOKEN_ZH] + i + [END_TOKENN_ZH] for i in tokenized_outputs]

tokenized_inputs = pad_sequences(tokenized_inputs, maxlen=max_len, padding="post", truncating="post")
tokenized_outputs = pad_sequences(tokenized_outputs, maxlen=max_len, padding="post", truncating="post")

from torch.utils.data import Dataset, DataLoader



config = T5Config(num_layers=24,
                      num_decoder_layers=24,
                      pad_token_id=0,
                      es_token_id=END_TOKENN_ZH,
                      model_parallel=True,
                      vocab_size=32128,
                      num_heads=8
                      )
model = T5ForConditionalGeneration(config)


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

dataset = DataProcesser(tokenized_inputs, tokenized_outputs)
dataloader = DataLoader(dataset, shuffle=True, batch_size=64)

class AdamWarmup:
    
    def __init__(self, model_size, warmup_steps, optimizer):
        
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0
        
    def get_lr(self):
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))
        
    def step(self):
        # Increment the number of steps each time we call the step function
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        # update the learning rate
        self.lr = lr
        self.optimizer.step()   


class LossWithLS(nn.Module):

    def __init__(self, size, smooth):
        super(LossWithLS, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.confidence = 1.0 - smooth
        self.smooth = smooth
        self.size = size
        
    def forward(self, prediction, target):
        """
        prediction of shape: (batch_size, max_words, vocab_size)
        target and mask of shape: (batch_size, max_words)
        """
        prediction = prediction.view(-1, prediction.size(-1))   # (batch_size * max_words, vocab_size)
        target = target.contiguous().view(-1)   # (batch_size * max_words)
        
        mask = (target > 0).float()       # (batch_size * max_words)
        labels = prediction.data.clone()
        labels.fill_(self.smooth / (self.size - 1))
        labels.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = self.criterion(prediction, labels)    # (batch_size * max_words, vocab_size)
        loss = (loss.sum(1) * mask).sum() / mask.sum()
        return loss
    
    
    
adam_optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
transformer_optimizer = AdamWarmup(model_size = config.d_model, warmup_steps = 4000, optimizer = adam_optimizer)
criterion = LossWithLS(VOCAB_SIZE_ZH, 0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10
model=nn.DataParallel(model)
model.to(device)
def train(train_loader, transformer, criterion, epoch):
    
    transformer.train()
    sum_loss = 0
    count = 0

    for i, (question, reply) in enumerate(train_loader):
        
        samples = question.shape[0]

        # Move to device
        question = question.to(device)
        reply = reply.to(device)

        # Prepare Target Data
        reply_input = reply[:, :-1]
        reply_target = reply[:, 1:]


        # Get the transformer outputs
        out = transformer(input_ids=question, decoder_input_ids=reply_input)

        # Compute the loss
        loss = criterion(out.logits, reply_target)
        
        # Backprop
        transformer_optimizer.optimizer.zero_grad()
        loss.backward()
        transformer_optimizer.step()
        
        sum_loss += loss.item() * samples
        count += samples
        
        if i % 100 == 0:
            print("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(train_loader), sum_loss/count))

EPOCHS=100
for epoch in range(EPOCHS):
    
    train(dataloader, model, criterion, epoch)
    
    state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}
    if ((epoch +1) % 10 == 0 and epoch > 0)  or epoch == EPOCHS - 1:
        torch.save(state, '/model_dir/translate_chekcpoint_' + str(epoch + 1) + '.pth.tar')