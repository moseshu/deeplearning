### qlora训练
    sh train_qlora.sh # 参考链接：https://github.com/artidoro/qlora 
    transformers >= 4.30.2 peft >= 0.4.0 bitsandbytes>=0.39.0
    https://huggingface.co/blog/4bit-transformers-bitsandbytes
    qlora训练的数据格式跟 alpaca.json是一样的{"instruction":"","input":"","output":""}
### alpaca lora训练
    cd llama-finetune && sh train_lora.sh
    peft_path 目录下包含adapter_config.json,adapter_model.bin 如果是pytorch_mdoel.bin可以在train_lora.py代码里修改为pytorch_model.bin来加载权重
### llama推理
    https://github.com/moseshu/deeplearning/blob/main/llama-finetune/README.md

### llama-triton server服务部署，最大长度1536，输出最大长度64，qps=2，平均每个token耗时35ms，在A10上编译
#### step1: cd triton-llama 目录下,删除ensemble/1/test.txt , 这个test.txt文件没用，保留1这个目录
    原始llama权重跟lora权重用huggingface_llama_convert_lora.py这个文件，具体执行脚本参看 convert.sh
    如何llama权重是合并lora后的权重就用huggingface_llama_convert.py
#### step2: 修改fastertransformer/config.pbtxt 
     确认 fastertransformer/config.pbtxt 内 "model_checkpoint_path" 模型路径正确,
     即fastertransformer/1/2-gpu,如果是单卡就是1-gpu，如果是4卡就是4-gpu，都是单机的情况下
#### step3: 
    tensor_para_size 要跟gpu卡数保持一致，比如你编译的是2-gpu,这个值就是2，4-gpu就是4，1-gpu就是1
#### step4: 
    cd triton-llama(这个目录随意命名，目录下要有ensemble  fastertransformer  postprocessing  preprocessing)
    例如ls model_repository/ # ensemble  fastertransformer postprocessing  preprocessing
    在model_repository目录下执行tritonserver --model-repository=/root/workspace/model_repository
### triton-llama client部署
     详细代码参考grpc_infer_client_ensemble.py
    
