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
from peft import LoraConfig, PeftModel,prepare_model_for_kbit_training,get_peft_model
from trl import SFTTrainer
from functools import partial
import transformers
from typing import List
import fire
import argparse



# The model that you want to train from the Hugging Face hub
model_name = "../llama_weight/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "../data/zhijian"


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
                text = instruction
            output_text.append(text)
    return output_text



##############################################################################

parser = argparse.ArgumentParser(description='模型参数')
parser.add_argument('--lora_r', type=int,default=64, help='LoRA attention dimension')
parser.add_argument('--lora_alpha', type=int,default=16, help=' Alpha parameter for LoRA scaling')
parser.add_argument('--lora_dropout', type=float,default=0.05, help='Dropout probability for LoRA layers')
parser.add_argument('--use_4bit', type=bool,default=True, help='Activate 4-bit precision base model loading')
parser.add_argument('--bnb_4bit_compute_dtype', type=str, default="float16", help='Compute dtype for 4-bit base models')
parser.add_argument('--bnb_4bit_quant_type', type=str, default='nf4', help='Quantization type (fp4 or nf4)')
parser.add_argument('--use_nested_quant', type=bool, default=False, help='Activate nested quantization for 4-bit base models (double quantization)')

parser.add_argument('--output_dir', type=str,default="./results", help=' Output directory where the model predictions and checkpoints will be stored')

parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--fp16', type=bool,default=False, help='Enable fp16 training (set bf16 to True with an A100)')
parser.add_argument('--bf16', type=float,default=False, help='Enable bf16 training (set bf16 to True with an A100)')


parser.add_argument('--per_device_train_batch_size', type=int, default=4, help='Batch size per GPU for training')
parser.add_argument('--per_device_eval_batch_size', type=int,default=4, help='Batch size per GPU for evaluation')
parser.add_argument('--gradient_accumulation_steps', type=int,default=4, help='Enable bf16 training (set bf16 to True with an A100)')
parser.add_argument('--gradient_checkpointing', type=bool,default=True, help='Dropout probability for LoRA layers')


parser.add_argument('--max_grad_norm', type=float, default=0.3, help='Maximum gradient normal (gradient clipping)')
parser.add_argument('--learning_rate', type=float,default=2e-4, help='Initial learning rate (AdamW optimizer)')
parser.add_argument('--weight_decay', type=float,default=0.001, help='Weight decay to apply to all layers except bias/LayerNorm weights')
parser.add_argument('--optim', type=str,default="paged_adamw_32bit", help='Optimizer to use')


parser.add_argument('--lr_scheduler_type', type=str, default="cosine", help='Maximum gradient normal (gradient clipping)')
parser.add_argument('--max_steps', type=int,default=-1, help='Initial learning rate (AdamW optimizer)')
parser.add_argument('--warmup_ratio', type=float,default=0.03, help='Ratio of steps for a linear warmup (from 0 to learning rate)')
parser.add_argument('--save_steps', type=int,default=0, help='Save checkpoint every X updates steps')
parser.add_argument('--max_seq_length', type=int,default=4096, help='Maximum sequence length to use')

parser.add_argument('--group_by_length', type=bool, default=True, help='')
parser.add_argument('--logging_steps', type=int,default=20, help='Initial learning rate (AdamW optimizer)')
parser.add_argument('--packing', type=bool,default=False, help='packing')

parser.add_argument('--data_path', type=str,default="alpaca", help='data path')

parser.add_argument('--base_model', type=str,default="meta-llama/Llama-2-7b-hf", help='model name')

parser.add_argument('--val_set_size', type=int,default=0, help='test size')
parser.add_argument('--lora_target_modules', type=str,default='q_proj,k_proj', help='')
parser.add_argument('--peft_path', type=str,default="", help='test size')
parser.add_argument('--format_prompt', type=str,default="llama2", help='test size')

args = parser.parse_args()

# LoRA attention dimension
lora_r = args.lora_r

# Alpha parameter for LoRA scaling
lora_alpha = args.lora_alpha
lora_target_modules = args.lora_target_modules.split(",")
# Dropout probability for LoRA layers
lora_dropout = args.lora_dropout
################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = args.use_4bit

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = args.bnb_4bit_compute_dtype

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = args.bnb_4bit_quant_type

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = args.use_nested_quant

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = args.output_dir

# Number of training epochs
num_train_epochs = args.num_epochs

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = args.fp16
bf16 = args.bf16
peft_path = args.peft_path
# Batch size per GPU for training
per_device_train_batch_size = args.per_device_train_batch_size

# Batch size per GPU for evaluation
per_device_eval_batch_size = args.per_device_eval_batch_size

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = args.gradient_accumulation_steps

# Enable gradient checkpointing
gradient_checkpointing = args.gradient_checkpointing

# Maximum gradient normal (gradient clipping)
max_grad_norm = args.max_grad_norm

# Initial learning rate (AdamW optimizer)
learning_rate = args.learning_rate

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = args.weight_decay

# Optimizer to use
optim = args.optim

# Learning rate schedule
lr_scheduler_type = args.lr_scheduler_type

# Number of training steps (overrides num_train_epochs)
max_steps = args.max_steps

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = args.warmup_ratio

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = args.group_by_length

# Save checkpoint every X updates steps
save_steps = args.save_steps

# Log every X updates steps
logging_steps = args.logging_steps

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = args.max_seq_length

# Pack multiple short examples in the same input sequence to increase efficiency
packing = args.packing

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training LlaMA2-QLoRA model with params:\n"
            f"base_model: {args.base_model}\n"
            f"data_path: {args.data_path}\n"
            f"output_dir: {args.output_dir}\n"
            f"micro_batch_size: {args.per_device_train_batch_size}\n"
            f"num_epochs: {args.num_epochs}\n"
            f"learning_rate: {args.learning_rate}\n"
            f"max_seq_length: {args.max_seq_length}\n"
            f"val_set_size: {args.val_set_size}\n"
            f"lora_r: {args.lora_r}\n"
            f"lora_alpha: {args.lora_alpha}\n"
            f"lora_dropout: {args.lora_dropout}\n"
            f"lora_target_modules: {args.lora_target_modules}\n"
            f"group_by_length: {args.group_by_length}\n"
            f"peft_path: {args.peft_path}\n"
            f"use_4bit: {args.use_4bit}\n"
            f"bnb_4bit_quant_type: {args.bnb_4bit_quant_type}\n"
            f"format_prompt:{args.format_prompt}\n"
        
        )
# Load the entire model on the GPU 0
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    
     

# Load dataset (you can process it here)

print("start load datasets")
if args.data_path.endswith(".json") or args.data_path.endswith(".jsonl"):
    data = load_dataset("json", data_files=args.data_path)
       
else:
    data = load_dataset(args.data_path)
    
print("end load datasets")
if args.val_set_size > 0:
    train_val = data["train"].train_test_split(
        test_size=args.val_set_size, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle()
        
    val_data = train_val["test"].shuffle()
        
else:
    train_data = data["train"].shuffle()
    val_data = None
    
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
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    r=args.lora_r,
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
# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    sharded_ddp=True,
    save_steps=save_steps,
    logging_strategy='steps',
    save_strategy="steps",
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=-1,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    ddp_find_unused_parameters=False if ddp else None,
)
formatting_func1 = partial(formatting_prompts_func,prompt="llama2")
callbacks = [SavePeftModelCallback()]
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    peft_config=peft_config,
    formatting_func=formatting_func1,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
    callbacks=callbacks,
)

# Train model
trainer.train()

# Save trained model
trainer.save_model()
