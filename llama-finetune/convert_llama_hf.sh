#!/bin/bash

# pip install "ctranslate2>=3.12" torch sentencepiece "transformers>=4.28" accelerate

# The conversion script will load the full FP16 model in memory.
# For large models the conversion could take a few minutes. 
rm -rf llamahf_$1
rm -rf llama_ct2_$1
lora_path=../llama/alpaca_checkpoint_medical
llama_model=llamahf_$1
python llama_convert2_pth.py --base_model ../llama/llama_weight/llama_all_weight --lora_model ${lora_path} --output_dir ${llama_model}
cp tokenizers/* ${llama_model}/
# llama_model=../llama/llama_weight/llama_all_weight
python llama_hf_converter.py --model ${llama_model} --quantization int8_float16 --output_dir llama_ct2_$1/
