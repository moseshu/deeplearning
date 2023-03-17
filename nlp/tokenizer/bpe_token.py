import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from pathlib import Path
import os
import tensorflow as tf
from gensim.corpora import WikiCorpus
import os
import argparse

class BPE_token(object):
    def __init__(self):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = Sequence([
            NFKC()
        ])
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()

    def bpe_train(self, paths):
        trainer = BpeTrainer(vocab_size=50000, show_progress=True, inital_alphabet=ByteLevel.alphabet(), special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ])
        self.tokenizer.train(trainer, paths)

    def save_tokenizer(self, location, prefix=None):
        if not os.path.exists(location):
            os.makedirs(location)
        self.tokenizer.model.save(location, prefix)

def download_data_wiki(lang='zh'):
    def store(corpus, lang):
        base_path = os.getcwd()
        store_path = os.path.join(base_path, '{}_corpus'.format(lang))
        if not os.path.exists(store_path):
            os.mkdir(store_path)
        file_idx=1
        for text in corpus.get_texts():
            current_file_path = os.path.join(store_path, 'article_{}.txt'.format(file_idx))
            with open(current_file_path, 'w' , encoding='utf-8') as file:
                file.write(bytes(' '.join(text), 'utf-8').decode('utf-8'))
            #endwith
            file_idx += 1
        #endfor

    def tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list:
        return [token for token in text.split() if token_min_len <= len(token) <= token_max_len]

    def run(lang):
        origin='https://dumps.wikimedia.org/{}wiki/latest/{}wiki-latest-pages-articles.xml.bz2'.format(lang,lang)
        fname='{}wiki-latest-pages-articles.xml.bz2'.format(lang)
        file_path = tf.keras.utils.get_file(origin=origin, fname=fname, untar=False, extract=False)
        corpus = WikiCorpus(file_path, lemmatize=False, lower=False, tokenizer_func=tokenizer_func)
        store(corpus, lang)
    run(lang)


if __name__ == '__main__':
    paths = [str(x) for x in Path("./text/").glob("**/*.txt")]
    tokenizer = BPE_token()
    # train the tokenizer model
    tokenizer.bpe_train(paths)
    # saving the tokenized data in our specified folder
    save_path = './tokenized_data'
    tokenizer.save_tokenizer(save_path)