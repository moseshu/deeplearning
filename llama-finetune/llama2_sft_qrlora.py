import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from trl import SFTTrainer
from functools import partial
import transformers
from typing import List
import fire
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


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




ORCA_PROMPT_DICT={"prompt_no_input":(
    "### System:\n"
    "You are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n"
    "### User:\n"
    "{instruction}"
    "\n\n### Response:{output}"
),"prompt_input":(
    "### System:\n"
    "You are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n"
    "### User:\n"
    "{instruction}"
    "\n\n### Input:\n"
    "{input}"
    "\n\n### Response:{output}"
)}


llama2_prompt ={ "prompt_no_input":"""<s> [INST] <<SYS>>
You are a helpful, respectful and honest assistant.follow the blow instruction give best answer.
<</SYS>>

{instruction} [/INST] {output} </s>"""}


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
#         print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        print("Saving PEFT checkpoint at end")
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)



def format_instruction(sample):
    content = sample['instruction']
    if sample['input']:
       return ALPACA_PROMPT_DICT['prompt_input'].format_map({'instruction':sample['instruction'],
                                                             "input":sample['input'],
                                                             "output":sample['output']})
    else:
        return ALPACA_PROMPT_DICT['prompt_no_input'].format_map({"instruction":sample['instruction'],
                                                                 "output":sample['output']})

def formatting_prompts_func(examples,prompt="llama2"):
    output_text = []
    # print(f"examples:{examples}")
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        response = examples["output"][i]
        if prompt == "alpaca":
            if len(input_text) >= 2:
                text = ALPACA_PROMPT_DICT['prompt_input'].format_map({"instruction":instruction,"input":input_text,"output":response})
            else:
                text = ALPACA_PROMPT_DICT['prompt_no_input'].format_map({"instruction":instruction,
                                                                     "output":response})
            
            output_text.append(text)
            
        elif prompt == "llama2":
            if response:
                input = f"{instruction}{input_text}"
                text = llama2_prompt['prompt_no_input'].format_map({"instruction":input,"output":response})
            else:
                if not instruction.startswith("<s>"):
                    instruction = "<s> " + instruction
                if not instruction.endswith("</s>"):
                    instruction += " </s>"
                text = instruction
            output_text.append(text)
    return output_text


def train(
    base_model: str="meta-llama/Llama-2-7b-hf",
    data_path: str = "data/alapa",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 12,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 2048,
    val_set_size: int = 200,
    # lora hyperparams
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    peft_path='',
    report_to='tensorboard',
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    use_4bit =True,
    bnb_4bit_quant_type = "nf4",
    gradient_accumulation_steps=8,
    chat: bool = False,
    packing: bool=False,
    format_prompt="llama2",
    bnb_4bit_compute_dtype = "float16",
    use_nested_quant = False,
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
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"peft_path: {peft_path}\n"
            f"use_4bit: {use_4bit}\n"
            f"bnb_4bit_quant_type: {bnb_4bit_quant_type}\n"
            f"format_prompt:{format_prompt}\n"
            f"packing:{packing}\n"
            f"use_nested_quant:{use_nested_quant}\n"
            f"bnb_4bit_compute_dtype:{bnb_4bit_compute_dtype}\n"
            
        
        )
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        # use_cache=False,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    
    
    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    
    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
        
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    if peft_path: 
        adapters_weights = torch.load(f"{peft_path}/adapter_model.bin")
        set_peft_model_state_dict(model, adapters_weights)
        
    model.print_trainable_parameters()


    print("start load datasets")
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
       
    else:
        data = load_dataset(data_path,cache_dir="./.cache/huggingface/datasets")
    
    print("end load datasets")
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle()
        
        val_data = train_val["test"].shuffle()
        
    else:
        train_data = data["train"].shuffle()
        val_data = None
    
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        sharded_ddp=True,
        save_steps=200,
        logging_steps=200,
        learning_rate=learning_rate,
        weight_decay=0.001,
        save_strategy="steps",
        fp16=False,
        bf16=False,
        max_steps = -1,
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        eval_steps=200 if val_set_size > 0 else None,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        ddp_timeout=3600,
        group_by_length=group_by_length,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
    )
    formatting_func1 = partial(formatting_prompts_func,prompt="llama2")
    callbacks = [SavePeftModelCallback()]
    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        formatting_func=formatting_func1,
        max_seq_length=cutoff_len,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
        callbacks=callbacks,
    )
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)
    # Train model
    train_result=trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_data)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Save trained model
    trainer.save_model()
    print("finished save model")

if __name__ == "__main__":
    fire.Fire(train)
