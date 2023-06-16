import torch
from transformers import LlamaForCausalLM, LlamaTokenizer,GenerationConfig
from peft import  PeftModel
import argparse
import json, os
import random
import pandas as pd
from typing import List
parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str,help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--data_file',default=None, type=str,help="file that contains instructions (one instruction per line).")
parser.add_argument('--predictions_file', default='./predictions.json', type=str)
args = parser.parse_args()

generation_config = dict(
    temperature=0.2,
    top_k=10,
    top_p=0.95,
    do_sample=True,
    num_beams=5,
    repetition_penalty=1.3,
    max_new_tokens=256
    )


def jload(data_file:str) -> List:
    
    with open(data_file,'r') as f:
        data = json.load(f)
    return data

def txtload(data_file:str) -> List:
    with open(data_file,'r') as f:
        data = f.read().splitlines()
    return data

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
        PROMPT_DICT['prompt_input'].format_map(instruction=instruction,input=input)
        
    return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})


if __name__ == '__main__':
    
    data_type=True
    if args.data_file.endswith(".json") or data_path.endswith(".jsonl"):
        examples=jload(args.data_file)
    else:
        examples = textload(args.data_file)
        data_type=False
    print("first 2 examples:")
    print(examples[:2])
    # examples = random.sample(examples,500)
    
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)

    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model, 
        load_in_8bit=True,
        torch_dtype=load_type,
        device_map="auto",
        )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size!=tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type)
    else:
        model = base_model

    if device==torch.device('cpu'):
        model.float()
    # test data
   
    
    model.to(device)
    model.eval()
    #生成参数配置
    generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=10,
            num_beams=10,
        )
    # generate_params = {
    #         "input_ids": input_ids,
    #         "generation_config": generation_config,
    #         "return_dict_in_generate": True,
    #         "output_scores": True,
    #         "max_new_tokens": 256,
    #     }
    
    
    with torch.no_grad():
        result = []
        for j in range(0,len(examples),10):
            samples = examples[j:j+step]
            for content in samples:
                content_prompt = generate_prompt(instruction=content['instruction'] if data_type else content)
    
                inputs = tokenizer(content_prompt,return_tensors="pt",add_special_tokens=True)
                input_ids = inputs["input_ids"].to(device)
    
                generation_output = model.generate(
                                input_ids=input_ids,
                                generation_config=generation_config,
                                return_dict_in_generate=True,
                                output_scores=True,
                                max_new_tokens=256,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                s = generation_output.sequences[0]
                output = tokenizer.decode(s,skip_special_tokens=True)
                output = output.split("### Response:")[1].strip()
                print(f"predict:{output}")
                print(f"output:{content['output']}")
                content['predict'] = output
                print(f"======{content}")
                result.append(content)
                with open(args.predictions_file,'a+') as f:
                    json_str = json.dumps(content,ensure_ascii=False)
                    f.write(json_str+"\n")
    
            break 
        df = pd.DataFrame(result)
        df.to_csv("data/predictions.csv",index=None)
        
        
