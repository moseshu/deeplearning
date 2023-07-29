### 使用ctranslate2 进行llama加速推理
#### 1.step1 
    sh convert_llama_hf.sh test # tokenizers目录下包含 special_tokens_map.json  tokenizer.model  tokenizer_config.json 这三个文件
    lora_path 是训练好的adapter权重，包含adapter_config.json adapter_model.bin(mv pytorch_model.bin adapter_model.bin)
#### 2.step2
     sh ct2.sh predict.json #要预测的json文件格式为[{"instruction":"","input":"","output":""}.....]
#### llama2 qlora 训练
pip install "transformers==4.31.0" "datasets==2.13.0" "peft==0.4.0" "accelerate==0.21.0" "bitsandbytes==0.40.2" "trl==0.4.7" "safetensors>=0.3.
1" --upgrade
sh qlora_llama2_sft.sh
