base_model='../llama_weight/Llama-2-7b-hf'
peft_path='lora-checkpoint/adapter_model' 
#torchrun --nproc_per_node=2 --master_port=20001 
#--deepspeed config/ds_config.json \
#deepspeed --num_gpus=2 
torchrun --nproc_per_node=8 --master_port=20002 llama2_sft_qlora.py \
    --base_model ${base_model} \
    --batch_size 16 \
    --data_path ../data/shopping \
    --micro_batch_size 1 \
    --num_epochs 5 \
    --output_dir checkpoint_qlora_shop \
    --group_by_length \
    --val_set_size 0 \
    --lora_r 64 \
    --cutoff_len 4096 \
    --learning_rate 1e-4 \
    --lora_alpha 128 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj]' \
    --report_to 'tensorboard'
