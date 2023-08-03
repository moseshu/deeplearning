#!/bin/bash
base_model='../llama_weight/Llama-2-7b-chat-hf'
peft_path='checkpoint_qlora_shop/checkpoint-9000/adapter_model' 
torchrun --nproc_per_node=4 --master_port=20001 qlora_sft_chat.py \
--base_model ${base_model} \
--data_path "../data/conversation-data" \
--lora_r 64 \
--output_dir checkpoint_chat \
--fp16 True \
--num_epochs 4 \
--lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj" \
--use_4bit True \
--format_prompt "llama2" \
--max_seq_length 2048

