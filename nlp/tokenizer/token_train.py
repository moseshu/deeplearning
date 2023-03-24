from transformers import AutoTokenizer
def get_training_corpus(dataset):
    """
    :param dataset: 将需要训练的文本列名改为text
    :return:
    """
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["text"]

def train_new_token():
    old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    training_corpus = get_training_corpus()
    new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
    new_tokenizer.save_pretrained("code-search-net-tokenizer")
