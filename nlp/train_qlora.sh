python qlora.py \
    --model_name_or_path  llama_weight/llama_all_weight \
    --output_dir ./guanaco-7b \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --dataset_format vicuna \
    --save_strategy steps \
    --data_seed 42 \
    --dataset data/qa_hs_zy_zj223w.json \
    --save_steps 200 \
    --save_total_limit 10 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 2 \
    --max_new_tokens 256 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --optim adamw_torch \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --source_max_len 2048 \
    --target_max_len 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0
