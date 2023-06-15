#!/bin/bash
lora_path=/root/workspace/llama/alpaca_checkpoint_zhijian
python generate.py --base_model llama_all_weight \
--lora_model checkpoint \
--tokenizer_path llama_all_weight \
--data_file /root/workspace/ctranslate2/data/dev_zhijian_chengnuo.json \
--predictions_file data/prediction.txt
