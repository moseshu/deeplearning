import ctranslate2
from transformers import LlamaTokenizer
import torch
import argparse
import json, os
from tqdm import tqdm
import random
import pandas as pd
from typing import List
import sentencepiece as spm
from sacrebleu.metrics import BLEU
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--data_file',default=None, type=str,help="file that contains instructions (one instruction per line).")
parser.add_argument('--predictions_file', default='./predictions.json', type=str)

args = parser.parse_args()
# tokenizer = spm.SentencePieceProcessor('llama_base_ct2/tokenizer.model')
tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

model = ctranslate2.Generator(args.base_model,device='auto')

prompt_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
)

def generate_prompt(instruction, input=None):
    PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    }
    
    if input:
        PROMPT_DICT['prompt_input'].format_map({"instruction":instruction,"input":input})
        
    return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})



def jload(data_file:str) -> List:
    
    with open(data_file,'r') as f:
        data = json.load(f)
    return data

def txtload(data_file:str) -> List:
    with open(data_file,'r') as f:
        data = f.read().splitlines()
    return data
    
    

def jwrite(data_path,data:list):
    with open(data_path,'w') as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
    


    
if __name__ == "__main__":
    if args.data_file.endswith(".json") or data_path.endswith(".jsonl"):
        examples=jload(args.data_file)
    else:
        examples = textload(args.data_file)
        data_type=False
    print(f"load {args.data_file} data ")
    print(f"examples:{examples[:2]}")
    # examples = examples[:10]
    predicts = []
    step = 10
    count = 0
    for j in range(0,len(examples),10):
        samples = examples[j:j+step]
        for i,text in enumerate(tqdm(samples)):
            instruc = text['instruction']
            if len(instruc) >= 2048:
                instruc = instruc[:2048]
            content_prompt = generate_prompt(instruc,input=None)
			
            prompt_tokens = tokenizer.tokenize(content_prompt)
            prompt_tokens.insert(0,tokenizer.bos_token)
            # prompt_tokens.append(tokenizer.eos_token)
            results = model.generate_batch([prompt_tokens],
                                           max_length=512, 
                                           sampling_topk=10,
                                           beam_size=5,
                                           sampling_temperature=0.1,
                                           include_prompt_in_result=False)

            output = tokenizer.decode(results[0].sequences_ids[0])
            text['predict'] = output
            print(f"output:{text['output']}\npredict:{output}")
            predicts.append(text)
        print(f"nums:{len(predicts)}")
    df = pd.DataFrame(predicts)
    prefix = args.data_file.split(".")[0]
    #df.to_csv(f'{args.data_file}_predict.csv',index=None)
    jwrite(f'{prefix}_predict.json',predicts)
    print(f"{args.data_file} bleu score")
    
    bleu = BLEU(lowercase=True, tokenize="zh")
    
    y_pre = df['predict'].tolist()
    labels = df['output'].tolist()

    score = bleu.corpus_score(y_pre, [labels])
    
    print(score)
    prefix = args.data_file.split(".")[0]
    # jwrite(f'{prefix}_predict.json',predicts)
