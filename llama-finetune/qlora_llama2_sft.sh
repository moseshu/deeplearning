#!/bin/bash
base_model=meta-llama/Llama-2-7b-hf
peft_path='checkpoint_chat/checkpoint-1600/adapter_model' 
#--deepspeed config/ds_config.json \
#deepspeed --num_gpus=2 
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=20005  \
    --base_model ${base_model} \
    --batch_size 16 \
    --data_path ../data/zhijian \
    --micro_batch_size 10 \
    --num_epochs 10 \
    --output_dir checkpoint_qlora \
    --group_by_length \
    --val_set_size 0 \
    --lora_r 64 \
    --cutoff_len 4096 \
    --learning_rate 3e-5 \
    --lora_alpha 128 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj]' \
    --report_to 'tensorboard' \
    --format_prompt 'llama2'

