import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Callable, Optional, Union
import json
import os
from accelerate import Accelerator
from config import hf_cache_dir
from grader import grade_answer
from peft import PeftModel
import string
from scipy.stats import entropy
os.environ["WANDB_PROJECT"] = "base-model-rl"

class CorrectnessRewardFunction:
    """Reward function for answer correctness"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.__name__ = "CorrectnessRewardFunction"
    
    def extract_model_answer(self, generated_text: str) -> str:
        """Extract answer using your exact parsing logic"""
        import re
        # Try to extract after 'The answer is: '
        match = re.search(r"The answer is:\s*(.*?)\*?\*?(<\｜end▁of▁sentence｜>|$)", generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ''
    
    def extract_solution_answer(self, solution: str) -> str:
        """Extract solution using your exact parsing logic"""
        parsed_solution = solution.split('The answer is: ')[-1]
        return parsed_solution.strip()

    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """Reward function for correctness only"""
        rewards = []
        solutions = kwargs.get('response', kwargs.get('solution', []))
        
        for i, completion in enumerate(completions):
            try:
                if i < len(solutions):
                    model_answer = self.extract_model_answer(completion)
                    solution_answer = self.extract_solution_answer(solutions[i])
                    is_correct = grade_answer(model_answer, solution_answer)
                    rewards.append(1.0 if is_correct else 0.0)
                else:
                    rewards.append(0.0)
            except Exception as e:
                print(f"Error computing correctness reward for completion {i}: {e}")
                rewards.append(0.0)
        
        return rewards


class ThinkingCompletionRewardFunction:
    """Reward function for finishing thinking process"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.__name__ = "ThinkingCompletionRewardFunction"

    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """Reward function for completing thinking process"""
        rewards = []
        
        for completion in completions:
            try:
                reward = 1.0 if '</think>' in completion else 0.0
                rewards.append(reward)
            except Exception as e:
                print(f"Error computing thinking completion reward: {e}")
                rewards.append(0.0)
        
        return rewards


class AnswerFormattingRewardFunction:
    """Reward function for proper answer formatting"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.__name__ = "AnswerFormattingRewardFunction"

    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """Reward function for proper answer formatting"""
        rewards = []
        
        for completion in completions:
            try:
                reward = 1.0 if 'The answer is:' in completion else 0.0
                rewards.append(reward)
            except Exception as e:
                print(f"Error computing formatting reward: {e}")
                rewards.append(0.0)
        
        return rewards


class LengthPenaltyRewardFunction:
    """Penalty function for overly long responses"""
    
    def __init__(self, tokenizer, penalty_len_start=1500, penalty_bin_size=200, penalty_scale=0.03):
        self.tokenizer = tokenizer
        self.penalty_len_start = penalty_len_start
        self.penalty_bin_size = penalty_bin_size
        self.penalty_scale = penalty_scale
        self.__name__ = "LengthPenaltyRewardFunction"

    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """Penalty function for overly long responses (returns negative rewards)"""
        rewards = []
        
        for completion in completions:
            try:
                penalty = (
                    max(0, (len(completion) - self.penalty_len_start))
                    ) // self.penalty_bin_size
                reward = -penalty * self.penalty_scale  # Negative penalty
                rewards.append(reward)
            except Exception as e:
                print(f"Error computing length penalty: {e}")
                rewards.append(0.0)
        
        return rewards


class KLCharacterDistributionRewardFunction:
    """Reward function penalizing KL divergence from standard English character distribution"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.__name__ = "KLCharacterDistributionRewardFunction"
        # Standard English character distribution (a-z, case insensitive)
        self.standard_distribution = {
            'a': 0.08167, 'b': 0.01492, 'c': 0.02782, 'd': 0.04253, 'e': 0.12702,
            'f': 0.02228, 'g': 0.02015, 'h': 0.06094, 'i': 0.06966, 'j': 0.00153,
            'k': 0.00772, 'l': 0.04025, 'm': 0.02406, 'n': 0.06749, 'o': 0.07507,
            'p': 0.01929, 'q': 0.00095, 'r': 0.05987, 's': 0.06327, 't': 0.09056,
            'u': 0.02758, 'v': 0.00978, 'w': 0.02360, 'x': 0.00150, 'y': 0.01974, 'z': 0.00074
        }

    def extract_thinking_text(self, completion: str) -> str:
        """Extract text within <think> tags or consider entire text if </think> is missing"""
        match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        if match:
            return match.group(1).strip()
        # If </think> is missing, consider everything after <think>
        start_tag_index = completion.find("<think>")
        if start_tag_index != -1:
            return completion[start_tag_index + len("<think>"):].strip()
        return ''

    def calculate_character_distribution(self, text: str) -> Dict[str, float]:
        """Calculate normalized character distribution for a-zA-Z"""
        text = text.lower()
        char_counts = {char: 0 for char in string.ascii_lowercase}
        total_chars = 0

        for char in text:
            if char in char_counts:
                char_counts[char] += 1
                total_chars += 1

        if total_chars == 0:
            return {char: 0.0 for char in string.ascii_lowercase}

        return {char: count / total_chars for char, count in char_counts.items()}

    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """Penalize KL divergence from standard English character distribution"""
        rewards = []

        for completion in completions:
            try:
                thinking_text = self.extract_thinking_text(completion)
                char_distribution = self.calculate_character_distribution(thinking_text)

                # Convert distributions to lists for KL divergence calculation
                standard_dist = [self.standard_distribution[char] for char in string.ascii_lowercase]
                completion_dist = [char_distribution[char] for char in string.ascii_lowercase]

                # Calculate KL divergence
                kl_divergence = entropy(completion_dist, standard_dist)

                # Penalize based on KL divergence (negative reward)
                rewards.append(-kl_divergence)
            except Exception as e:
                print(f"Error computing KL divergence reward: {e}")
                rewards.append(0.0)

        return rewards


def create_combined_reward_function(tokenizer):
    """Helper function to create a combined reward function for evaluation purposes"""
    correctness_reward = CorrectnessRewardFunction(tokenizer)
    thinking_reward = ThinkingCompletionRewardFunction(tokenizer)
    formatting_reward = AnswerFormattingRewardFunction(tokenizer)
    length_penalty = LengthPenaltyRewardFunction(tokenizer)
    kl_divergence_penalty = KLCharacterDistributionRewardFunction(tokenizer)
    
    def combined_reward_function(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """Combined reward function that mimics the original behavior"""
        correctness_rewards = correctness_reward(prompts, completions, **kwargs)
        thinking_rewards = thinking_reward(prompts, completions, **kwargs)
        formatting_rewards = formatting_reward(prompts, completions, **kwargs)
        length_penalties = length_penalty(prompts, completions, **kwargs)
        kl_divergence_rewards = kl_divergence_penalty(prompts, completions, **kwargs)
        
        # Combine with the same weights as training
        combined_rewards = []
        for corr, think, form, length, kl_div in zip(correctness_rewards, thinking_rewards, formatting_rewards, length_penalties, kl_divergence_rewards):
            total_reward = (1.0 * corr) + (0.3 * think) + (0.3 * form) + (1.0 * length) + (0.0 * kl_div)
            combined_rewards.append(total_reward)
        
        return combined_rewards
    
    combined_reward_function.__name__ = "CombinedRewardFunction"
    return combined_reward_function

def setup_lora_config_for_quantized():
    """Setup LoRA configuration optimized for quantized models"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,  # Typically 2x the rank
        lora_dropout=0.05,
        # Target all linear layers for maximum expressiveness
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # Enable more memory-efficient training
        use_rslora=True,  # Rank-stabilized LoRA
        use_dora=False,   # DoRA uses more memory
    )
    return lora_config


def load_your_model_and_tokenizer_with_lora_and_quantization(
    device_map_auto: bool = False, 
    quantization_type: str = "4bit",
    small_model: bool = False,  # For debugging
    training: bool = True,  # Whether to prepare for training
    return_peft: bool = True,  # Whether to return PEFT model
):
    """Load model and tokenizer with LoRA and quantization setup"""

    if not small_model: 
        model_path = 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
    else:
        model_path = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'

    args = {
        'pretrained_model_name_or_path': model_path,
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True,
        'low_cpu_mem_usage': True,
        'cache_dir': hf_cache_dir,
    }
    args['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for additional savings
        bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    if device_map_auto:
        args['device_map'] = "auto"

    # Load base model with quantization
    print(f"Loading model with {quantization_type} quantization...")
    model = AutoModelForCausalLM.from_pretrained(**args)
    model.generation_config.temperature = 1.0
    model.generation_config.do_sample = True
    model.generation_config.top_p = 0.9
    
    # Prepare model for k-bit training (required for quantized + LoRA)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=False)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    template_path = "chat_templates/deepseek_distill_llama_template.jinja"
    with open(template_path, "r") as file:
        jinja_template = file.read()
    tokenizer.chat_template = jinja_template

    if return_peft:
        lora_config = setup_lora_config_for_quantized()
        model = get_peft_model(model, lora_config)
    else:
        pass
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    if training:
        model.train()
        model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_metamath_dataset_with_split(hf_cache_dir: str = None):
    """Prepare MetaMathQA dataset with train/validation split"""
    # Load dataset using your method
    ds = load_dataset("meta-math/MetaMathQA", cache_dir=hf_cache_dir)

    # Drop the 'original_question' column
    ds = ds.remove_columns(["original_question"])

    def format_for_grpo(example):
        """Format each example using your exact prompt structure"""
        query = example['query']
        content = f'Answer the following question, and format your answer as "The answer is: <answer>". Here is the question: {query}'
        return {
            "prompt": content,
            "response": example['response'],
            "query": query
        }

    # Shuffle dataset with a fixed seed
    ds = ds.shuffle(seed=42)

    # Split dataset into train and validation sets
    train_dataset = ds['train']
    n_train = 8000 #len(train_dataset)
    val_dataset = train_dataset.select(range(64))  # Use exactly 100 samples for validation
    train_dataset = train_dataset.select(range(64, 64+n_train))  # Remaining samples for training

    formatted_train_dataset = train_dataset.map(format_for_grpo)
    formatted_val_dataset = val_dataset.map(format_for_grpo)

    return formatted_train_dataset, formatted_val_dataset


def setup_grpo_config_with_lora_quantization():
    """Setup GRPO configuration optimized for LoRA training"""
    num_gpus = torch.cuda.device_count()

    grpo_config = GRPOConfig(
        # Learning rate - can be higher with LoRA
        learning_rate=5e-6,  # TODO
        gradient_accumulation_steps=20, # TODO
        num_train_epochs=1., # TODO
        reward_weights=[1.0, 0.3, 0.3, 0., 0.1],  # [correctness, thinking, formatting, length_penalty, kl_penalty]

        # Batch sizing - can be larger with LoRA
        per_device_train_batch_size=1,  # TODO
        per_device_eval_batch_size=64,  # TODO
        num_generations=num_gpus*2,  # TODO

        # Generation parameters - optimized for your format
        temperature=1.0,  # TODO
        top_p=0.9,
        generation_kwargs={
            'max_new_tokens': 1500,  # TODO
            'do_sample': True,
            'use_cache': True,
        },
        
        # Training stability
        beta=0.0,  # No KL penalty
        
        # Logging and saving
        logging_steps=2,
        log_completions=True,
        num_completions_to_print=10,
        output_dir="/workspace/data/grpo-metamath-base",
        eval_strategy="steps",
        eval_steps=10,
        eval_on_start=True,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Use evaluation reward to determine the best model

        # Memory optimization
        gradient_checkpointing=False,  # Enable with LoRA
        dataloader_pin_memory=False,
        
        # Additional memory optimizations
        remove_unused_columns=False,
        fp16=False,  # Use bfloat16 instead
        bf16=True,

        # Quantization-specific optimizations
        optim="adamw_bnb_8bit",  # Use 8-bit AdamW optimizer
        lr_scheduler_type='cosine',

        ## Additional stability settings # TODO: add?
        warmup_steps=10,    # Learning rate warmup
    )
    
    return grpo_config


def train_with_grpo_lora_quantized_multigpu_with_validation(quantization_type: str = "4bit"):
    """Multi-GPU GRPO training with LoRA, quantization, and validation"""
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print("Setting up multi-GPU GRPO training with LoRA and validation...")

    # Load model and tokenizer with LoRA and quantization
    model, tokenizer = load_your_model_and_tokenizer_with_lora_and_quantization(
        quantization_type=quantization_type
    )
    train_dataset, val_dataset = prepare_metamath_dataset_with_split()
    # Create multiple reward functions
    correctness_reward = CorrectnessRewardFunction(tokenizer)
    thinking_reward = ThinkingCompletionRewardFunction(tokenizer)
    formatting_reward = AnswerFormattingRewardFunction(tokenizer)
    length_penalty = LengthPenaltyRewardFunction(tokenizer)
    kl_divergence_penalty = KLCharacterDistributionRewardFunction(tokenizer)
    
    # Setup config optimized for LoRA + quantization
    grpo_config = setup_grpo_config_with_lora_quantization()

    # Set tokenizer-specific generation kwargs
    grpo_config.generation_kwargs['pad_token_id'] = tokenizer.pad_token_id
    grpo_config.generation_kwargs['eos_token_id'] = tokenizer.eos_token_id

    # Initialize trainer
    grpo_trainer = GRPOTrainer(
        args=grpo_config,
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Add validation dataset
        reward_funcs=[correctness_reward, thinking_reward, formatting_reward, length_penalty, kl_divergence_penalty],
    )

    # Train with validation
    grpo_trainer.train()


def sample_and_evaluate_generations_quantized(
        num_prompts=5, 
        num_generations=5,
        quantization_type: str = "4bit"
    ):
    """Sample the quantized LoRA model for a few prompts and print generations with rewards"""
    print(f"Loading your model and tokenizer with LoRA + {quantization_type} quantization...")
    model, tokenizer = load_your_model_and_tokenizer_with_lora_and_quantization(
        device_map_auto=True,
        quantization_type=quantization_type
    )

    print("Preparing MetaMathQA dataset...")
    dataset, _ = prepare_metamath_dataset_with_split()

    print("Creating reward function...")
    reward_function = create_combined_reward_function(tokenizer)

    print("Sampling prompts and generating completions...")
    sampled_prompts = dataset.select(range(num_prompts))

    for i, example in enumerate(sampled_prompts):
        prompt = example["prompt"]
        print("="*50)
        print(f"\nPrompt {i+1}: {prompt}")

        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(
            formatted_prompt, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            top_p=0.9,
            temperature=1.,
            num_return_sequences=num_generations,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode only the new tokens
        completions = []
        for output in outputs:
            completion = tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True)
            completions.append(completion)

        # Compute rewards
        rewards = reward_function([formatted_prompt] * num_generations, completions, response=[example["response"]] * num_generations)

        for j, (completion, reward) in enumerate(zip(completions, rewards)):
            print("*"*25)
            print(f"GENERATION {j+1}, Reward: {reward:.3f}")
            print(completion)
            print("*"*25)
        print("="*50)


def test_trained_model(
        num_prompts=5, 
        num_generations=5,
        quantization_type: str = "4bit"
    ):
    """Test the newly trained LoRA model for a few prompts and print generations with rewards"""
    print(f"Loading your newly trained model and tokenizer with LoRA + {quantization_type} quantization...")
    model_path = f"./grpo-metamath-lora-{quantization_type}-adapter"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Preparing MetaMathQA dataset...")
    dataset, _ = prepare_metamath_dataset_with_split()

    print("Creating reward function...")
    reward_function = create_combined_reward_function(tokenizer)

    print("Sampling prompts and generating completions...")
    sampled_prompts = dataset.select(range(num_prompts))

    for i, example in enumerate(sampled_prompts):
        prompt = example["prompt"]
        print("="*50)
        print(f"\nPrompt {i+1}: {prompt}")

        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(
            formatted_prompt, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            top_p=0.9,
            temperature=1.,
            num_return_sequences=num_generations,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode only the new tokens
        completions = []
        for output in outputs:
            completion = tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True)
            completions.append(completion)

        # Compute rewards
        rewards = reward_function([formatted_prompt] * num_generations, completions, response=[example["response"]] * num_generations)

        for j, (completion, reward) in enumerate(zip(completions, rewards)):
            print("*"*25)
            print(f"GENERATION {j+1}, Reward: {reward:.3f}")
            print(completion)
            print("*"*25)
        print("="*50)


if __name__ == "__main__":
    train_with_grpo_lora_quantized_multigpu_with_validation()