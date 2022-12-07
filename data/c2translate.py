import ctranslate2
from subword_nmt.apply_bpe import BPE
import codecs
from sudachipy import dictionary
import jieba
from sudachipy import tokenizer
from datetime import datetime 
from sacrebleu.metrics import BLEU
import re
from sudachipy import tokenizer

tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.B

model_name="checkpoint40.pt"
#model_name="averaged36-32.pt"
print(model_name)

def load_txt(file_path):
    data = []
    with open(file_path) as f:
        line = f.readline()
        while line:
            line = line.replace("\n","").replace(" ","").replace("\u3000","")
            data.append(line)
            line = f.readline()
    return data

def cleantxt(raw):
    raw = str(raw)
    fil = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5]+', re.UNICODE)
    return fil.sub('', raw)


def bpe_text(texts:list):
    bpe_result=[]
    with codecs.open("./data/bpecode", encoding='utf-8') as f:
        bpe = BPE(codes=f)
        for data in texts:
            tmp = bpe.process_line(data)
            bpe_result.append(tmp)
    return bpe_result

translator = ctranslate2.Translator("./models/ja_zh/ct2_model", device="cuda")
bleu = BLEU(lowercase=True, tokenize="zh")

src_ja = load_txt("./test.dedup.ja")
target_zh = load_txt("./test.dedup.zh")
ja = [i.replace("!","").replace("…","").replace("。","").replace("？","").replace("！","") for i in src_ja]
jpn = [" ".join([m.surface() for m in tokenizer_obj.tokenize(i, mode)]) for i in ja]
src_bpe = bpe_text(jpn)
print("src_bpe",src_bpe[:3])
src_bpe = [i.split(" ") for i in src_bpe]
pre_zh=[]
batch_size=100
startime = datetime.now()
for i in range(0,len(src_bpe),batch_size):
    src_batch = src_bpe[i:i+batch_size]
    results = translator.translate_batch(src_batch,beam_size=3)
    for x in results:
        y = x.hypotheses[0]
        y = "".join([i.replace("@@","") for i in y])
        pre_zh.append(y)
endtime = datetime.now() 
runtime = (endtime - startime).seconds
print("耗时:{}s".format(runtime))
target_zh = [cleantxt(s) for s in target_zh]
pre_zh = [cleantxt(s) for s in pre_zh]

score = bleu.corpus_score(pre_zh, [target_zh])
print(score)

print("*"*30)