llama_path=llama-7b-weight
lora_path=lora_peft_path
#单机2卡gpu,2-gpu
python3 huggingface_llama_convert_lora.py -saved_dir=saved_model -in_llama_file=${llama_path} -in_lora_file=${lora_path} -infer_gpu_num=2 -weig
ht_data_type=fp16 -model_name=llama_7b
cp -r saved_model/2-gpu triton-llama/fastertransformer/1/
copy triton-llama/fastertransformer/1/2-gpu/config.ini  triton-llama/fastertransformer/1/

#rm -rf triton-llama/fastertransformer/1/2-gpu 
# 单机1卡gpu,1-gpu
#python3 huggingface_llama_convert_lora.py -saved_dir=saved_model_1gpu -in_llama_file=${llama_path} -in_lora_file=${lora_path} -infer_gpu_num=1 
#-weight_data_type=fp16 -model_name=llama_7b
#rm -rf model_repository_kefu_test/fastertransformer/1/1-gpu
#cp -r saved_model_1gpu/1-gpu model_repository_kefu_test/fastertransformer/1/
#copy model_repository_kefu_test/fastertransformer/1/1-gpu/config.ini  model_repository_kefu_test/fastertransformer/1/
