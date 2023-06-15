"""
Usage: 
python merge_llama_with_chinese_lora.py \
    --base_model path/to/llama/model \
    --lora_model path/to/first/lora/model [path/to/second/lora/model] \
    --output_type [pth|huggingface] \
    --output_dir path/to/output/dir
"""
import argparse
import json
import os
import gc
import torch
import peft
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, required=True,
                    type=str, help="Please specify a base_model")
parser.add_argument('--lora_model', default=None, required=True,
                    type=str, help="Please specify LoRA models to be merged (ordered); use commas to separate multiple LoRA models.")
parser.add_argument('--offload_dir', default=None, type=str,
                    help="(Optional) Please specify a temp folder for offloading (useful for low-RAM machines). Default None (disable offload).")
parser.add_argument('--output_type', default='pth',choices=['pth','huggingface'], type=str,
                    help="save the merged model in pth or huggingface format.")
parser.add_argument('--output_dir', default='./', type=str)

def translate_state_dict_key(k):
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError
def save_shards(model_sd, num_shards: int):
    with torch.no_grad():
            if num_shards == 1:
                new_state_dict = {}
                for k, v in model_sd.items():
                    new_k = translate_state_dict_key(k)
                    if new_k is not None:
                        if "wq" in new_k or "wk" in new_k:
                            new_state_dict[new_k] = unpermute(v)
                        else:
                            new_state_dict[new_k] = v

                os.makedirs(output_dir, exist_ok=True)
                print(f"Saving shard 1 of {num_shards} into {output_dir}/consolidated.00.pth")
                torch.save(new_state_dict, output_dir + "/consolidated.00.pth")
                with open(output_dir + "/params.json", "w") as f:
                    json.dump(params, f)

                    
                    
if __name__ == "__main__":
    args = parser.parse_args()
    base_model_path = args.base_model
    # lora_model_paths = args.lora_model
    output_dir = args.output_dir
    output_type = args.output_type
    offload_dir = args.offload_dir

    print(f"Base model: {base_model_path}")
    # print(f"LoRA model(s) {lora_model_paths}:")
    
    if offload_dir is not None:
        # Load with offloading, which is useful for low-RAM machines.
        # Note that if you have enough RAM, please use original method instead, as it is faster.
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            offload_folder=offload_dir,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
        )
    else:
        # Original method without offloading
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
        )
        
        
    lora_model = PeftModel.from_pretrained(
                base_model,
                args.lora_model,
                device_map={"": "cpu"},
                torch_dtype=torch.float16,
            )
    
    print(f"Merging with merge_and_unload...")
    base_model = lora_model.merge_and_unload()
 
    base_model_sd = base_model.state_dict()
    os.makedirs(output_dir, exist_ok=True)
    LlamaForCausalLM.save_pretrained(base_model, output_dir) #, state_dict=deloreanized_sd)
    # torch.save(base_model_sd,output_dir + "/consolidated.00.pth")
    
#     params = {
#         "dim": 4096,
#         "multiple_of": 256,
#         "n_heads": 32,
#         "n_layers": 32,
#         "norm_eps": 1e-06,
#         "vocab_size": -1,
#         }
  
#     with open(output_dir + "/params.json", "w") as f:
#         json.dump(params, f)
    
    # del lora_model, base_model
    # n_layers = params["n_layers"]
    # n_heads = params["n_heads"]
    # dim = params["dim"]
    # dims_per_head = dim // n_heads
    # base = 10000.0
    # inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    # save_shards(model_sd=base_model_sd, num_shards=1)
