
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, Dataset, load_from_disk
from trl import SFTTrainer
from tqdm import tqdm
import json
import gc
import os
from accelerate import Accelerator
from config import storage_dir
from trl import SFTConfig

#accelerator = Accelerator()

model_name = "Qwen/Qwen3-14B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    cache_dir="/n/holylfs06/LABS/krajan_lab/Lab/cfang/hf_cache/"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_storage_dir = os.path.join(storage_dir, "lm_sys", model_name.split("/")[-1])
dataset = load_from_disk(os.path.join(model_storage_dir, 'lm_sys_responses_rot13'))


# # Supervised Finetuning

# Prepare for SFT Training
def format_conversation_for_sft(example):
    """Format conversation for SFT training"""
    conversation = example['conversation']
    
    # Format as a single string for SFT
    formatted_text = ""
    for turn in conversation:
        if turn['role'] == 'user':
            formatted_text += f"<|im_start|>user\n{turn['content']}<|im_end|>\n"
        elif turn['role'] == 'assistant':
            formatted_text += f"<|im_start|>assistant\n{turn['content']}<|im_end|>\n"
    
    return {"text": formatted_text}

# Format dataset for SFT
sft_dataset = dataset.map(format_conversation_for_sft, remove_columns=dataset.column_names)

# Training arguments optimized for QwQ
training_args = SFTConfig(
    output_dir="./qwen-sft-output",
    completion_only_loss=None,
    assistant_only_loss=False,  # I think this should be True?

    max_length=1000,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Adjust based on your GPU memory
    gradient_accumulation_steps=1,
    learning_rate=5e-6,  # Lower learning rate for large models
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    #evaluation_strategy="no",
    save_strategy="steps",
    load_best_model_at_end=False,
    report_to=None,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,  # Save memory
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    deepspeed="./ds_config.json"
)

# Initialize SFT Traine
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=sft_dataset,
    processing_class=tokenizer,
)

#trainer.model, trainer.optimizer, trainer.train_dataloader = accelerator.prepare(
#    trainer.model, trainer.optimizer, trainer.train_dataloader
#)

# Start training
print("Starting SFT training...")
trainer.train()

# Save the final model
trainer.save_model("./qwq-sft-final")

