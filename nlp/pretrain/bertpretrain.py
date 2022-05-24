from tokenizers.implementations import (ByteLevelBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer)
from tokenizers.processors import BertProcessing
from transformers import AlbertConfig, BertTokenizerFast, AlbertModel, BertModel, LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoTokenizer
from transformers import TextDatasetForNextSentencePrediction, BertForPreTraining, BertConfig
from transformers import AutoModel, AutoConfig
import torch



def load_data():
    files = "./lunyu.txt"
    with open(files) as f:
        lines = f.read().splitlines()
    return lines


class NewModelConcig():
    def __init__(self, ):
        self.vocab_size = 10000

def train_tokenizer():
    files = "./lunyu.txt"
    vocab_size = 10000
    min_frequency = 2
    limit_alphabet = 10000
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]  # 适用于Bert和Albert

    # Initialize a tokenizer
    tokenizer = BertWordPieceTokenizer(clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True)

    # Customize training
    tokenizer.train(
        files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=special_tokens,
        limit_alphabet=limit_alphabet,
        wordpieces_prefix="##"
    )
    tokenizer.save_model("./tokenizer")
    # tokenizer.save("/Users/moses/Desktop/bilidata/tokenizer/vocab.json", pretty=True)
    tokenizer1 = BertTokenizerFast.from_pretrained('./tokenizer')
    # tokenizer.add_special_tokens({'mask_token': '[MSK]'})
    #
    tokenizer1.model_max_length = 512

    tokenizer1.save_pretrained("./tokenizer")
    tokenizer = BertWordPieceTokenizer(
        "./tokenizer/vocab.txt",
    )

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("[CLS]", tokenizer.token_to_id("[SEP]")),
        ("[SEP]", tokenizer.token_to_id("[CLS]")),
    )

    tokenizer.enable_truncation(max_length=512)
    tokens = tokenizer.encode("子曰：学而时习之。").tokens
    print(tokens)


def train_model():
    config = BertConfig(
        vocab_size=10000,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=0,
        position_embedding_type="absolute",
        truncation=True,
    )

    tokenizer = BertTokenizerFast.from_pretrained('./tokenizer')
    tokenizer.add_special_tokens({'mask_token': '[MSK]'})

    print(tokenizer.model_max_length)

    # print(len(tokenizer))
    model = BertForPreTraining(config)
    # print(model.num_parameters())
    # dataset = LineByLineTextDataset(
    #     tokenizer=tokenizer,
    #     file_path="/Users/moses/Desktop/lunyu.txt",
    #     block_size=256,
    # )

    train_dataset = TextDatasetForNextSentencePrediction(
        tokenizer=tokenizer,
        file_path='./lunyu.txt',
        block_size=512,
        short_seq_probability=0.1,
        nsp_probability=0.5,
    )
    MAX_LEN = 512
    SEP = tokenizer.sep_token_id
    print(SEP)
    ending_sep_token_tensor = torch.tensor([SEP])
    for sample in train_dataset.examples:
        if len(sample['input_ids']) > MAX_LEN:
            sample['input_ids'] = torch.cat((sample['input_ids'][:MAX_LEN - 1], ending_sep_token_tensor), 0)
            sample['token_type_ids'] = sample['token_type_ids'][:MAX_LEN]
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir='./param_out',
        overwrite_output_dir=True,
        num_train_epochs=16,
        learning_rate=5e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        evaluation_strategy='steps',
        eval_steps=10000,
        warmup_steps=10000,
        weight_decay=0.01,
        dataloader_num_workers=4,
        fp16=False,
        save_steps=10000,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,

    )

    trainer.train()
    trainer.save_model("./albertmodel_out")


if __name__ == '__main__':
    train_tokenizer()
    train_model()