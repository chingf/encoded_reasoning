# # Import Required Libraries
# 
# Import all necessary libraries, including torch, transformers, pandas, numpy, and any custom utilities such as `rot13_alpha` and `grade_answer`.

# %%
import torch
import transformers
import pandas as pd
import numpy as np
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from jinja2 import Template
from tqdm import tqdm

# Custom utilities
from utils_activations import rot13_alpha
from grader import grade_answer

# Set pandas display options for debugging (optional)
pd.set_option('display.max_columns', None)

qlora_dir = 'grpo-metamath-cos'
checkpoint = 180

# %% [markdown]
# # Load MetaMathQA Dataset
# 
# Load the MetaMathQA dataset using the `datasets` library and set up the cache directory if needed.

# %%
# Set HuggingFace cache directory if needed
hf_cache_dir = os.environ.get("HF_HOME", "/workspace/data/hf_cache")

# Load MetaMathQA dataset
ds = load_dataset("meta-math/MetaMathQA", cache_dir=hf_cache_dir)

# %% [markdown]
# # Load SFT Model and Tokenizer
# 
# Load the SFT model and tokenizer from the specified model path, and set up the chat template as required.

# %%
# SFT model path
sft_model_path = '/workspace/data/axolotl-outputs/llama_deepseek_2epochs/merged'

# Load SFT model and tokenizer
sft_model = AutoModelForCausalLM.from_pretrained(
    sft_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
sft_tokenizer = AutoTokenizer.from_pretrained(sft_model_path)

# Load chat template
sft_template_path = "chat_templates/deepseek_distill_llama_template.jinja"
with open(sft_template_path, "r") as file:
    sft_jinja_template = file.read()
sft_tokenizer.chat_template = sft_jinja_template

# %% [markdown]
# # Load GRPO Merged Model and Tokenizer
# 
# Load the GRPO merged model and tokenizer from the merged directory, and set up the chat template as required.

# %%
already_merged = False

# %%
def load_your_model_and_tokenizer_with_lora_and_quantization(
    quantization_type: str = "4bit",
    small_model: bool = False,  # For debugging
    training: bool = True,  # Whether to prepare for training
    return_peft: bool = True,  # Whether to return PEFT model
):
    """Load model and tokenizer with LoRA and quantization setup"""

    if not small_model: 
        model_path = '/workspace/data/axolotl-outputs/llama_deepseek_2epochs/merged'
    else:
        model_path = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'

    args = {
        'pretrained_model_name_or_path': model_path,
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True,
        'low_cpu_mem_usage': True,
        'cache_dir': hf_cache_dir,
        'device_map': 'auto'
    }
    args['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for additional savings
        bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    # Load base model with quantization
    print(f"Loading model with {quantization_type} quantization...")
    model = AutoModelForCausalLM.from_pretrained(**args)
    model.generation_config.temperature = 1.0
    model.generation_config.do_sample = True
    model.generation_config.top_p = 0.9
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    template_path = "chat_templates/deepseek_distill_llama_template.jinja"
    with open(template_path, "r") as file:
        jinja_template = file.read()
    tokenizer.chat_template = jinja_template

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    if training:
        model.train()
        model.print_trainable_parameters()
    
    return model, tokenizer

# %%
if already_merged:
    # GRPO merged model path
    grpo_merged_model_path = f"/workspace/data/{qlora_dir}/merged"

    # Load GRPO merged model and tokenizer
    grpo_model = AutoModelForCausalLM.from_pretrained(
        grpo_merged_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    grpo_tokenizer = AutoTokenizer.from_pretrained(grpo_merged_model_path)

    # Load chat template
    grpo_template_path = "chat_templates/deepseek_distill_llama_template.jinja"
    with open(grpo_template_path, "r") as file:
        grpo_jinja_template = file.read()
    grpo_tokenizer.chat_template = grpo_jinja_template
else:
    qlora_chckpt_dir = f"/workspace/data/{qlora_dir}/checkpoint-{checkpoint}"
    base_model, tokenizer = load_your_model_and_tokenizer_with_lora_and_quantization(training=False)
    peft_model = PeftModel.from_pretrained(base_model, qlora_chckpt_dir)
    merged_model = peft_model.merge_and_unload()
    grpo_model = merged_model
    grpo_tokenizer = tokenizer


# %% [markdown]
# # Select 50 Random Questions
# 
# Randomly select 50 unique indices from the `ds['train']` split and extract the corresponding questions and solutions.

# %%
# Set random seed for reproducibility
np.random.seed(42)

# Get total number of samples
n_total = len(ds['train'])

# Randomly select 50 unique indices
n_samples = 25
random_indices = np.random.choice(n_total, size=n_samples, replace=False)

# Extract questions and solutions
questions = [ds['train'][int(i)]['query'] for i in random_indices]
solutions = [ds['train'][int(i)]['response'] for i in random_indices]

# %% [markdown]
# # Define Model Inference and Parsing Functions
# 
# Define functions to prompt a model, extract the raw response, apply ROT-14 translation to the thinking content, parse the solution, and compute the answer grade.

# %%
def get_model_output(prompt, model, tokenizer, max_new_tokens=1200):
    """
    Prompt the model and return the generated text.
    """
    content = f"Answer the following question, and format your answer as 'The answer is: <answer>'. Here is the question: {prompt}"
    messages = [{"role": "user", "content": content}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.75,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    return generated_text

def translate_rot13_thinking(generated_text):
    import re
    match = re.search(r"<think>(.*?)</think>", generated_text, re.DOTALL)
    if match:
        thinking_content = match.group(1)
    else:
        thinking_content = generated_text
    return rot13_alpha(thinking_content)

def parse_solution(solution_text):
    """
    Parse the solution from the reference solution text.
    """
    return solution_text.split('The answer is: ')[-1].strip()

def parse_model_answer(generated_text):
    """
    Parse the answer from the model's generated text.
    """
    import re
    # Try to extract after 'The answer is: '
    match = re.search(r"The answer is:\s*(.*?)\*?\*?(<\｜end▁of▁sentence｜>|$)", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ''

def compute_grade(model_answer, parsed_solution):
    """
    Compute the answer grade using the provided grader.
    """
    return grade_answer(model_answer, parsed_solution)

# %% [markdown]
# # Generate and Collect Responses from Both Models
# 
# Iterate over the 50 selected questions, prompt both SFT and GRPO models, collect the question, raw response, ROT-14 translated thinking content, parsed solution, and answer grade for each model.

# %%
results = []

for idx, (question, solution) in enumerate(tqdm(zip(questions, solutions), total=n_samples)):
    parsed_solution = parse_solution(solution)

    # SFT Model
    sft_raw_response = get_model_output(question, sft_model, sft_tokenizer)
    sft_rot13_thinking_translated = translate_rot13_thinking(sft_raw_response)
    sft_model_answer = parse_model_answer(sft_raw_response)
    sft_grade = compute_grade(sft_model_answer, parsed_solution)

    # GRPO Model
    grpo_raw_response = get_model_output(question, grpo_model, grpo_tokenizer)
    grpo_rot13_thinking_translated = translate_rot13_thinking(grpo_raw_response)
    grpo_model_answer = parse_model_answer(grpo_raw_response)
    grpo_grade = compute_grade(grpo_model_answer, parsed_solution)

    results.append({
        "index": random_indices[idx],
        "question": question,
        "parsed_solution": parsed_solution,

        "sft_raw_response": sft_raw_response,
        "sft_rot14_thinking": sft_rot13_thinking_translated,
        "sft_model_answer": sft_model_answer,
        "sft_grade": sft_grade,

        "grpo_raw_response": grpo_raw_response,
        "grpo_rot14_thinking": grpo_rot13_thinking_translated,
        "grpo_model_answer": grpo_model_answer,
        "grpo_grade": grpo_grade,
    })

# %% [markdown]
# # Save Results to CSV
# 
# Combine all collected results into a pandas DataFrame and save it as a CSV file with appropriate columns.

# %%
# Create DataFrame
df = pd.DataFrame(results)

# Select and order columns for CSV
csv_columns = [
    "index",
    "question",
    "parsed_solution",

    "sft_raw_response",
    "sft_rot14_thinking",
    "sft_model_answer",
    "sft_grade",

    "grpo_raw_response",
    "grpo_rot14_thinking",
    "grpo_model_answer",
    "grpo_grade",
]

# Save to CSV
output_csv_path = f"/workspace/data/encoded_reasoning/{qlora_dir}_results.csv"
output_csv_path = output_csv_path.replace("-", "_")
df.to_csv(output_csv_path, index=False, columns=csv_columns)
print(f"Results saved to {output_csv_path}")

# %%



