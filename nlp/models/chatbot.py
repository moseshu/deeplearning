import math
import torch.nn.functional as F
import jieba
import torch
import torch.nn as nn
import copy
from embedding import PositionalEncoding
from encoder import EncoderLayer
from decoder import DecoderLayer
from nlp.tokenizer.texttoken import pad_sequences
from nlp.dataprocess.dataset import text_generate, load_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdamWarmup:
    """
      References
        https://github.com/fawazsammani/chatbot-transformer/blob/master/transformer%20chatbot.ipynb
    """

    def __init__(self, model_size, warmup_steps, optimizer):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0

    def get_lr(self):
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5),
                                               self.current_step * self.warmup_steps ** (-1.5))

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
    """
       References
         https://github.com/fawazsammani/chatbot-transformer/blob/master/transformer%20chatbot.ipynb
     """

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
        prediction = prediction.view(-1, prediction.size(-1))  # (batch_size * max_words, vocab_size)
        target = target.contiguous().view(-1)  # (batch_size * max_words)

        mask = (target > 0).float()  # (batch_size * max_words)
        labels = prediction.data.clone()
        labels.fill_(self.smooth / (self.size - 1))
        labels.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = self.criterion(prediction, labels)  # (batch_size * max_words, vocab_size)
        loss = (loss.sum(1) * mask).sum() / mask.sum()
        return loss


class Transformer(nn.Module):
    """
      References
        https://github.com/fawazsammani/chatbot-transformer/blob/master/transformer%20chatbot.ipynb
    """

    def __init__(self, d_model, heads, num_layers, vocab_size):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.srcpe = PositionalEncoding(d_model, 0, vocab_size)
        self.tgtpe = PositionalEncoding(d_model, 0, vocab_size)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(num_layers)])
        self.logit = nn.Linear(d_model, self.vocab_size)
        self.src_masking = PadMasking(pad_or_ahead="pad", idx=0)
        self.tgt_masking = PadMasking(pad_or_ahead="ahead", idx=0)

    def encode(self, src_words, src_mask):
        src_embeddings = self.embed(src_words) * math.sqrt(self.d_model)
        src_embeddings = self.srcpe(src_embeddings)
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings, src_mask)
        return src_embeddings

    def decode(self, target_words, target_mask, src_embeddings, src_mask):
        tgt_embeddings = self.embed(target_words) * math.sqrt(self.d_model)
        tgt_embeddings = self.tgtpe(tgt_embeddings)
        for layer in self.decoder:
            tgt_embeddings = layer(tgt_embeddings, src_embeddings, src_mask, target_mask)
        return tgt_embeddings

    def forward(self, src_words, target_words, training=True):
        src_mask = self.src_masking(src_words) if training else None
        target_mask = self.tgt_masking(target_words) if training else None
        encoded = self.encode(src_words, src_mask)
        decoded = self.decode(target_words, target_mask, encoded, src_mask)
        out = F.log_softmax(self.logit(decoded), dim=2)
        return out


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
        out = transformer(question, reply_input, True)

        # Compute the loss
        loss = criterion(out, reply_target)

        # Backprop
        transformer_optimizer.optimizer.zero_grad()
        loss.backward()
        transformer_optimizer.step()

        sum_loss += loss.item() * samples
        count += samples

        if i % 10 == 0:
            print("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(train_loader), sum_loss / count))


def load_model():
    checkpoint = torch.load('/Users/moses/workspace/data/model/checkpoint_99.bin')
    model = checkpoint['model']
    return model


def evaluate(sentence, model, tokenizer, max_len=64):
    model.eval()
    sentence = " ".join(jieba.cut(sentence, cut_all=False, HMM=True))
    START_TOKEN = len(tokenizer.word_index) + 1
    END_TOKEN = len(tokenizer.word_index) + 2
    sentence = [START_TOKEN] + tokenizer.texts_to_sequences([sentence])[0] + [END_TOKEN]

    sentence = torch.tensor(sentence, dtype=torch.long).unsqueeze(dim=0).to(device)
    output = torch.tensor([START_TOKEN], dtype=torch.long).unsqueeze(dim=0).to(device)

    for i in range(max_len):
        predictions = model(sentence, output, training=False)
        # print("pd shape",predictions.shape)
        predictions = predictions[:, -1:, :]
        pred_id = torch.argmax(predictions, dim=-1)

        if pred_id.unsqueeze(0).item() == END_TOKEN:
            break
        output = torch.cat([output, pred_id], dim=-1)

    return output.squeeze(0)


def predict(sentence, model, id2word):
    predictions = evaluate(sentence, model, tokenizer, max_len=40)
    predictions = predictions.cpu().numpy()
    print(predictions)

    predic_senc = [id2word[i] for i in predictions if i < len(tokenizer.word_index) and i > 0]
    print("".join(predic_senc))


if __name__ == '__main__':
    d_model = 512
    heads = 8
    num_layers = 3
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    MAX_LEN = 64

    dataloader, VOCAB_SIZE = text_generate("../../data", "../../data/words.json", max_len=MAX_LEN)
    tokenizer = load_token("../../data/words.json")
    id2word = {v: k for k, v in tokenizer.word_index.items()}
    transformer = Transformer(d_model=d_model, heads=heads, num_layers=num_layers, vocab_size=VOCAB_SIZE)
    transformer = transformer.to(device)
    adam_optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    transformer_optimizer = AdamWarmup(model_size=d_model, warmup_steps=4000, optimizer=adam_optimizer)
    criterion = LossWithLS(VOCAB_SIZE, 0.1)
    for epoch in range(epochs):
        train(dataloader, transformer, criterion, epoch)

        state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}
        if (epoch + 1) % 10 == 0 and epoch > 0:
            torch.save(state, 'checkpoint_' + str(epoch + 1) + '.pth.tar')
    print("*" * 20)
    checkpoint = torch.load('checkpoint_99.pth.tar')
    model = checkpoint['transformer']
    sentence = "你叫什么名字"
    predict(sentence, model=model)

