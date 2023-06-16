### qlora训练
    sh train_qlora.sh # 参考链接：https://github.com/artidoro/qlora 
    transformers >= 4.30.2 peft >= 0.4.0 bitsandbytes>=0.39.0
    https://huggingface.co/blog/4bit-transformers-bitsandbytes
### alpaca lora训练
    cd llama-finetune && sh train_lora.sh
    peft_path 目录下包含adapter_config.json,adapter_model.bin 如果是pytorch_mdoel.bin可以在train_lora.py代码里修改为pytorch_model.bin来加载权重
### llama推理
    https://github.com/moseshu/deeplearning/blob/main/llama-finetune/README.md
