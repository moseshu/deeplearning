#!/bin/bash

#pip install "ctranslate2>=3.12" torch sentencepiece

# The conversion script will load the full FP16 model in memory.
# For large models the conversion could take a few minutes.

python llama_converter.py --model_dir ./llama-ct2 --tokenizer_model ../llama/llama_weight/llama_all_weight/tokenizer.model \
                          --quantization int8_float16 --output_dir llama_ct2/
