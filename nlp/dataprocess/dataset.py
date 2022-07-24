import torch
from torch.utils.data import Dataset, DataLoader
from nlp.tokenizer.texttoken import Tokenizer, pad_sequences, tokenizer_from_json
import yaml
import os
import jieba
import json
import sys

lib_path = os.path.abspath(os.path.join(".."))
sys.path.append(lib_path)


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
                torch.tensor(seg2[:-1], dtype=torch.long),
                torch.tensor(seg2[1:], dtype=torch.long))


def load_token(dir):
    with open(dir, 'r') as f:
        data = json.load(f)

    tokenizer = tokenizer_from_json(data)
    return tokenizer


def load_yaml_text(file_path):
    with open(file_path) as f:
        lines = f.read()
    lines = yaml.safe_load(lines)
    return lines


def load_all_data(directory):
    print(directory)
    file_names = os.listdir(directory)
    questions, answers = [], []
    for f_name in file_names:
        if f_name.endswith(".yml"):
            file_path = os.path.join(directory, f_name)
            conversions = load_yaml_text(file_path)['conversations']
            ques = [i[0] for i in conversions]
            ans = [i[1] for i in conversions]
            questions.extend(ques)
            answers.extend(ans)
    print("load data finished,the size of data is ", len(questions))
    questions = [" ".join(jieba.cut(str(s), cut_all=False, HMM=True)) for s in questions]
    answers = [" ".join(jieba.cut(str(s), cut_all=False, HMM=True)) for s in answers]
    return questions, answers


def load_train_data(directory, words_token_path=None, max_len=64):
    questions, answers = load_all_data(directory)
    tokenizer = load_token(words_token_path)
    START_TOKEN = len(tokenizer.word_index) + 1
    END_TOKENN = len(tokenizer.word_index) + 2
    vocab_size = END_TOKENN

    tokenized_inputs = tokenizer.texts_to_sequences(questions)
    tokenized_outputs = tokenizer.texts_to_sequences(answers)
    # pad token sentences
    tokenized_inputs = [[START_TOKEN] + i + [END_TOKENN] for i in tokenized_inputs]
    tokenized_outputs = [[START_TOKEN] + i + [END_TOKENN] for i in tokenized_outputs]

    tokenized_inputs = pad_sequences(tokenized_inputs, maxlen=max_len, padding="post", truncating="post")
    tokenized_outputs = pad_sequences(tokenized_outputs, maxlen=max_len, padding="post", truncating="post")



    return {"questions": tokenized_inputs, "answers": tokenized_outputs, "vocab_size": vocab_size + 1}


def text_generate(dir, word_token_path, max_len=64):
    data = load_train_data(dir, word_token_path, max_len)
    questions = data['questions']
    answers = data['answers']
    vocab_size = data['vocab_size']
    dataset = DataProcesser(questions, answers)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=3)
    return dataloader, vocab_size


if __name__ == '__main__':
    itd, vocab_size = text_generate("../../data", "../../data/words.json")
    for i, (ques, answer, target) in enumerate(itd):
        print(ques[:3])
        print(answer[:3])
