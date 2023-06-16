base_model='llama_weight/llama_all_weight'
peft_path='lora_checkpoint'
#torchrun --nproc_per_node=2 --master_port=20001 
#deepspeed --num_gpus=2 
torchrun --nproc_per_node=4 --master_port=20001 alpaca_lora.py \
    --deepspeed config/ds_config.json \
    --base_model ${base_model} \
    --batch_size 16 \
    --data_path data/medical_train.json \
    --micro_batch_size 1 \
    --num_epochs 10 \
    --output_dir alpaca_checkpoint_medical \
    --group_by_length \
    --val_set_size 0 \
    --lora_r 64 \
    --cutoff_len 640 \
    --learning_rate 1e-4 \
    --lora_alpha 128 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj]' \
    --report_to 'tensorboard' \
    --train_on_inputs False \
    --add_eos_token True \
    --peft_path ${peft_path}
