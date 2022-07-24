import json
from nlp.tokenizer.texttoken import Tokenizer, tokenizer_from_json
from nlp.dataprocess.dataset import load_all_data


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

if __name__ == '__main__':
    train_tokenizer("../../data")
