import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset, load_dataset
import re
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import os
from accelerate import Accelerator
from config import hf_cache_dir
from grader import grade_answer

class MetaMathRewardFunction:
    """Reward function for MetaMathQA dataset with your custom grading"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.__name__ = "MetaMathRewardFunction"
    
    def extract_model_answer(self, generated_text: str) -> str:
        """Extract answer using your exact parsing logic"""
        model_answer = generated_text.split('</think>')[-1]
        model_answer = model_answer.split('The answer is: ')[-1].split('<｜end▁of▁sentence｜>')[0]
        return model_answer.strip()
    
    def extract_solution_answer(self, solution: str) -> str:
        """Extract solution using your exact parsing logic"""
        parsed_solution = solution.split('The answer is: ')[-1]
        return parsed_solution.strip()

    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """
        GRPO-compatible reward function for MetaMathQA
        
        Args:
            prompts: List of formatted prompts 
            completions: List of model completions
            **kwargs: Additional columns including 'response' (ground truth solutions)
        
        Returns:
            List of reward scores
        """
        rewards = []
        
        # Get ground truth solutions from dataset
        solutions = kwargs.get('response', kwargs.get('solution', []))
        
        for i, completion in enumerate(completions):
            try:
                if i < len(solutions):
                    # Extract answers using your parsing logic
                    model_answer = self.extract_model_answer(completion)
                    solution_answer = self.extract_solution_answer(solutions[i])
                    
                    # Use your custom grading function
                    is_correct = grade_answer(model_answer, solution_answer)
                    
                    # Base reward
                    if is_correct:
                        reward = 1.0
                    else:
                        reward = 0.0
                    
                    # Bonus for finishing thinking
                    if '</think>' in completion:
                        reward += 0.3
                    
                    # Bonus for giving answer properly
                    if 'The answer is:' in completion:
                        reward += 0.3
                    
                    # Small penalty for overly long responses
                    penalty_len_start = 1500
                    penalty_bin_size = 200
                    penalty_scale = 0.03
                    penalty = (
                        max(0, (len(completion) - penalty_len_start))
                        )//penalty_bin_size
                    reward -= penalty * penalty_scale

                    rewards.append(reward)
                else:
                    rewards.append(0.0)
                    
            except Exception as e:
                print(f"Error computing reward for completion {i}: {e}")
                print(f"Completion preview: {completion[:100]}...")
                rewards.append(0.0)
        
        return rewards


def load_your_model_and_tokenizer(device_map_auto: bool=False):
    """Load model and tokenizer using your exact setup"""
    
    model_path = '/workspace/data/axolotl-outputs/llama_deepseek_2epochs/merged'
    #model_path = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'

    args = {
        'pretrained_model_name_or_path': model_path,
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True,
        'low_cpu_mem_usage': True,
        'cache_dir': hf_cache_dir,
    }
    if device_map_auto:
        args['device_map'] = "auto"

    model = AutoModelForCausalLM.from_pretrained(**args)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load your custom chat template
    template_path = "chat_templates/deepseek_distill_llama_template.jinja"
    with open(template_path, "r") as file:
        jinja_template = file.read()
    tokenizer.chat_template = jinja_template
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def prepare_metamath_dataset(hf_cache_dir: str = None):
    """Prepare MetaMathQA dataset in GRPO format using your exact setup"""
    
    # Load dataset using your method
    ds = load_dataset("meta-math/MetaMathQA", cache_dir=hf_cache_dir)
    
    def format_for_grpo(example):
        """Format each example using your exact prompt structure"""
        
        # Get the original query from MetaMathQA
        query = example['query']
        
        # Format content using your exact method
        content = f'Answer the following question, and format your answer as \'The answer is: <answer>\'. Here is the question: {query}'
        
        # Note: We'll apply chat template in the reward function since GRPO needs raw prompts
        return {
            "prompt": content,  # Raw content, chat template applied during generation
            "response": example['response'],  # Ground truth solution
            "query": query  # Original query for reference
        }
    
    # Use a subset for training (adjust size as needed)
    train_dataset = ds['train'].select(range(5000))  # Use first 5k examples
    formatted_dataset = train_dataset.map(format_for_grpo)
    
    return formatted_dataset


def setup_grpo_config():
    """Setup GRPO configuration for your model"""
    
    grpo_config = GRPOConfig(
        # Use your model path
        learning_rate=5e-6,  # Conservative for fine-tuned model
        gradient_accumulation_steps=4,
        num_train_epochs=3.,  # Since you're starting from a 2-epoch model

        # Batch sizing
        per_device_train_batch_size=1,  # Adjust based on your GPU memory
        per_device_eval_batch_size=1,
        generation_batch_size=4,
        num_generations = 4,

        # Generation parameters - optimized for your format
        temperature=1.0,
        top_p=0.9,
        generation_kwargs={
            'max_new_tokens': 1200,
            'do_sample': True,
            'use_cache': True,
            },
        
        # Training stability
        beta=0.0,  # No KL penalty
        
        # Logging and saving
        logging_steps=5,
        save_steps=100,
        eval_steps=50,
        output_dir="/workspace/data/grpo-metamath-model",
        
        # Memory optimization
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        
    )
    
    return grpo_config


class CustomGRPOTrainer(GRPOTrainer):
    """Custom GRPO trainer that applies your chat template"""
    
    def __init__(self, config, **kwargs):
        super().__init__(args=config, **kwargs)

    def _prepare_prompts(self, prompts):
        """Apply chat template to prompts before generation"""
        formatted_prompts = []
        
        for prompt in prompts:
            # Apply your exact chat template formatting
            messages = [
                {"role": "user", "content": prompt},
            ]
            
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        
        return formatted_prompts


def train_with_grpo_multigpu():
    """Multi-GPU version"""
    
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print("Setting up multi-GPU GRPO training...")
    
    # Load model and tokenizer
    model, tokenizer = load_your_model_and_tokenizer()
    dataset = prepare_metamath_dataset()
    reward_function = MetaMathRewardFunction(tokenizer)
    
    # Setup config
    grpo_config = setup_grpo_config()
    
    # Initialize trainer
    grpo_trainer = CustomGRPOTrainer(
        config=grpo_config,
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[reward_function],
    )
    
    # Train
    grpo_trainer.train()
    
    # Save only on main process
    if accelerator.is_main_process:
        grpo_trainer.save_model("./grpo-metamath-multigpu")
        tokenizer.save_pretrained("./grpo-metamath-multigpu")


def sample_and_evaluate_generations(num_prompts=5, num_generations=5):
    """Sample the model for a few prompts and print generations with rewards"""
    print("Loading your model and tokenizer...")
    model, tokenizer = load_your_model_and_tokenizer(device_map_auto=True)

    print("Preparing MetaMathQA dataset...")
    dataset = prepare_metamath_dataset()

    print("Creating reward function...")
    reward_function = MetaMathRewardFunction(tokenizer)

    print("Sampling prompts and generating completions...")
    sampled_prompts = dataset.select(range(num_prompts))

    for i, example in enumerate(sampled_prompts):
        prompt = example["prompt"]
        print(f"\nPrompt {i+1}: {prompt}")

        # Generate completions
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            top_p=0.9,
            temperature=0.75,
            num_return_sequences=num_generations
        )

        completions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Compute rewards
        rewards = reward_function([prompt] * num_generations, completions, response=[example["response"]] * num_generations)

        for j, (completion, reward) in enumerate(zip(completions, rewards)):
            print(f"\n  Generation {j+1}: {completion}")
            print(f"  Reward: {reward:.3f}")


if __name__ == "__main__":
#    train_with_grpo_multigpu()
    sample_and_evaluate_generations()