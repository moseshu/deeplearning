from datasets import load_dataset
from random import randrange
import os
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer
import fire


ALPACA_PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:{output}"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:{output}"
        ),
        }

def format_instruction(sample):
    content = sample['instruction']
    if sample['input']:
       return ALPACA_PROMPT_DICT['prompt_input'].format_map({'instruction':sample['instruction'],
                                                             "input":sample['input'],
                                                             "output":sample['output']})
    else:
        return ALPACA_PROMPT_DICT['prompt_no_input'].format_map({"instruction":sample['instruction'],
                                                                 "output":sample['output']})
    


def train(
    base_model: str="meta-llama/Llama-2-7b-hf",
    data_path: str = "data/alapa",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 12,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    val_set_size: int = 200,
    # lora hyperparams
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    cache_dir=None,
    peft_path='',
    report_to='tensorboard',
    use_flash_attention=False,
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4"
):

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training LlaMA2-QLoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"cache_dir: {cache_dir}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"peft_path: {peft_path}\n"
            f"load_in_4bit: {load_in_4bit}\n"
            f"bnb_4bit_quant_type: {bnb_4bit_quant_type}\n"
            f"use_flash_attention:{use_flash_attention}\n"
        )
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model, 
                                                 quantization_config=bnb_config,
                                                 use_cache=False,
                                                 device_map="auto")
    model.config.pretraining_tp = 1

    # Validate that the model is using flash attention, by comparing doc strings
    if use_flash_attention:
        from llama_patch import forward,unplace_flash_attn_with_attn
        unplace_flash_attn_with_attn()
        assert model.model.layers[0].self_attn.forward.__doc__ == forward.__doc__, "Model is not using flash attention"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",)
    
    
    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    if peft_path: 
        adapters_weights = torch.load(f"{peft_path}/adapter_model.bin")
        set_peft_model_state_dict(model, adapters_weights)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    
    print("start load datasets")
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
       
    else:
        data = load_dataset(data_path)
    print("end load datasets")
    dataset = data['train']
    
    args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            sharded_ddp=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            logging_strategy='steps',
            save_strategy="steps",
            learning_rate=2e-4,
            bf16=False,
            fp16=True,
            tf32=True,
            save_total_limit=10,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            report_to=report_to ,
            lr_scheduler_type="constant",
            ddp_find_unused_parameters=False if ddp else None,
            disable_tqdm=True # disable tqdm since with packing values are in correct
        )
    max_seq_length = cutoff_len

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    
    trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction, 
    args=args,
    )
    
    trainer.train() # there will not be a progress bar since tqdm is disabled
    
    # save model
    trainer.save_model()

    print("finished save model")


if __name__ == "__main__":
    fire.Fire(train)

