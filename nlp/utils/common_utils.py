import json
from nlp.tokenizer.texttoken import Tokenizer, tokenizer_from_json
from nlp.dataprocess.dataset import load_all_data
import re

def train_tokenizer(dir):
    num_words = 2 ** 13
    oov_token = '<UNK>'

    text1, text2 = load_all_data(dir)
    print(text1[:3])
    print(text2[:3])
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(text1 + text2)
    VOCAB_SIZE = len(tokenizer.word_index)
    print("vocab_size is :{}".format(VOCAB_SIZE))
    token_json = tokenizer.to_json()

    with open("../../data/words.json", 'w', encoding='utf-8') as f:
        json.dump(token_json, f, ensure_ascii=False)  # 存为json对象
        # f.write(json.dumps(token_json))  # 存为json字符串


def sweep_config_to_sweep_values(sweep_config):
    """
    Converts an instance of wandb.Config to plain values map.
    wandb.Config varies across versions quite significantly,
    so we use the `keys` method that works consistently.
    """

    return {key: sweep_config[key] for key in sweep_config.keys()}



def cleantxt(raw, flag=4):
    raw = str(raw)
    raw = raw + " "
    raw = re.sub("https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", "", raw)
    raw = re.sub("@[0-9a-zA-Z\u4e00-\u9fa5]+[: ]", "", raw)

    raw = raw.lower()
    if flag == 1:
        #去除无效字符
        result = re.sub('\W+', '', raw).replace("_", '')
        return result
    elif flag == 2:
        #过滤中文，字母，数字
        fil = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5]+', re.UNICODE)
        return fil.sub('', raw)
    elif flag == 3:
        #过滤数字，字母
        fil = re.compile(u'[^0-9a-zA-Z]+', re.UNICODE)
        return fil.sub('', raw)
    elif flag == 4:
        #只保留中文
        fil = re.compile(u'[^\u4e00-\u9fa5]+', re.UNICODE)
        return fil.sub('', raw)
    else:
        return raw

if __name__ == '__main__':
    train_tokenizer("../../data")
