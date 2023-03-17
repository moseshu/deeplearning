from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
from torch.nn import CrossEntropyLoss
import transformers
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
import logging
from torch.utils.data import DataLoader


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
        self.criterion = nn.KLDivLoss(reduction="sum")
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


def create_logger(log_file):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def loss_acc_fn(pred, label, pad_id):
    """
        prediction of shape: (batch_size, max_words, vocab_size)
        target and mask of shape: (batch_size, max_words)
        """
    criterion = CrossEntropyLoss(ignore_index=pad_id, reduction='sum', label_smoothing=0.1)
    prediction = pred.view(-1, pred.size(-1))  # (batch_size * max_words, vocab_size)
    target = label.contiguous().view(-1)  # (batch_size * max_words)
    mask = (target != pad_id).float()  # (batch_size * max_words)  pad_id部分为 0，非pad 为1
    num_targets = mask.long().sum()
    _, y_pred = prediction.max(dim=-1)  # 取预测的最大值
    correct = (target == y_pred) & mask  # 预测正确对数量
    loss = criterion(prediction, target)  # (batch_size * max_words, vocab_size)
    loss = (loss.sum(1) * mask).sum() / mask.sum()
    correct = correct.float().sum()
    accuracy = correct / num_targets
    return loss, accuracy


def load_tokenizer(save_path):
    # 加载前面训练好的tokenizer，查看 tokenizer 包下的 bpe_token.py
    tokenizer = GPT2Tokenizer.from_pretrained(save_path)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    })

    return tokenizer


def collate_fn(batch):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch:
    :return:
    """
    global pad_id
    input_ids = []
    btc_size = len(batch)
    # 使用pad_id对小于max_input_len的input_id进行补全
    for btc_idx in range(btc_size):
        input_ids.append(batch[btc_idx])
        # input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))
    labels = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    return torch.tensor(labels, dtype=torch.long)


def create_model():
    tokenizer = load_tokenizer("save_path")
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    model = GPT2LMHeadModel(config)
    return model, tokenizer


def train(train_data, device, epoch):
    model, tokenizer = create_model()
    logger = create_logger("./log/gpt2.log")
    model.train()
    LEARNING_RATE = 0.05
    max_grad_norm = 1.0
    gradient_accumulation_steps = 2
    cuda_nums = 4
    TRG_PAD_IDX = tokenizer.pad_token_id
    # 用于统计每次梯度累计的loss
    running_loss = 0
    # 统计一共训练了多少个step
    overall_step = 0
    # 记录tensorboardX
    tb_writer = SummaryWriter(log_dir="writer_dir")
    # 记录 out of memory的次数
    oom_time = 0
    epochs = 20
    batch_size = 64
    log_step = 100
    total_steps = int(len(train_data) * epochs / batch_size / gradient_accumulation_steps)

    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # adam_optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    # transformer_optimizer = AdamWarmup(model_size=model.config.n_embd, warmup_steps=4000, optimizer=adam_optimizer)
    #

    optimizer = transformers.AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=4000, t_total=total_steps)

    for i, input_ids in enumerate(train_data):
        input_ids = input_ids.to(device)
        logits = model(input_ids=input_ids)
        loss, acc = loss_acc_fn(logits, input_ids[:, :, 1:])
        if cuda_nums > 1:
            loss = loss.mean()
            acc = acc.mean()

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
            acc = acc / gradient_accumulation_steps
        loss.backward()
        # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if (i + 1) % gradient_accumulation_steps == 0:
            running_loss += loss.item()
            # 更新参数
            optimizer.step()
            # 清空梯度信息
            optimizer.zero_grad()
            # 进行warm up
            scheduler.step()
            overall_step += 1
            # 更新日志与tnesorboardX信息
            if (overall_step + 1) % log_step == 0:
                logger.info("batch {} of epoch {}, loss {}, accuracy {}".format(i + 1, epoch + 1, loss, acc))
                tb_writer.add_scalar('loss', loss.item(), overall_step)


def evaluate(model, device, test_dataloader, multi_gpu, args, logger):
    logger.info("start evaluating model")
    model.eval()
    logger.info('starting evaluating')
    # 记录tensorboardX
    # tb_writer = SummaryWriter(log_dir=args.writer_dir)

    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    #                              collate_fn=collate_fn)
    with torch.no_grad():
        for batch_idx, input_ids in enumerate(test_dataloader):
            input_ids.to(device)
            outputs = model.forward(input_ids=input_ids)
            loss, accuracy = loss_acc_fn(outputs, labels=input_ids, device=device)

            if multi_gpu:
                loss = loss.mean()
                accuracy = accuracy.mean()
            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
                accuracy = accuracy / args.gradient_accumulation
            logger.info("evaluate batch {} ,loss {} ,accuracy {}".format(batch_idx, loss, accuracy))
            # tb_writer.add_scalar('loss', loss.item(), overall_step)
        logger.info("finishing evaluating")


if __name__ == '__main__':
    train()
