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


class MetaMathRewardFunction:
    """Reward function for MetaMathQA dataset with your custom grading"""
    
    def __init__(self, tokenizer, grade_answer_func):
        self.tokenizer = tokenizer
        self.grade_answer = grade_answer_func
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
                    is_correct = self.grade_answer(model_answer, solution_answer)
                    
                    # Base reward
                    if is_correct:
                        reward = 1.0
                    else:
                        reward = 0.0
                    
                    # Bonus for showing reasoning (if model uses <think> tags)
                    if '</think>' in completion:
                        reasoning_bonus = self._score_thinking_quality(completion)
                        reward += reasoning_bonus
                    
                    # Bonus for proper formatting
                    if 'The answer is:' in completion:
                        reward += 0.1
                    
                    # Small penalty for overly long responses
                    if len(completion) > 1500:
                        reward -= 0.1
                    
                    rewards.append(float(np.clip(reward, 0.0, 2.0)))
                else:
                    rewards.append(0.0)
                    
            except Exception as e:
                print(f"Error computing reward for completion {i}: {e}")
                print(f"Completion preview: {completion[:100]}...")
                rewards.append(0.0)
        
        return rewards
    
    def _score_thinking_quality(self, completion: str) -> float:
        """Score the quality of reasoning in <think> tags"""
        if '</think>' not in completion:
            return 0.0
        
        # Extract thinking content
        think_start = completion.find('<think>')
        think_end = completion.find('</think>')
        
        if think_start == -1 or think_end == -1:
            return 0.0
        
        thinking_content = completion[think_start + 7:think_end]  # +7 for '<think>'
        
        score = 0.0
        thinking_lower = thinking_content.lower()
        
        # Look for reasoning indicators
        reasoning_patterns = [
            r"let me|let's|i need to|first|then|next|finally",
            r"step \d+|part \d+",
            r"because|since|therefore|so|thus",
            r"\d+\s*[+\-*/=]\s*\d+",  # Mathematical operations
            r"substitute|solve|calculate|compute",
        ]
        
        for pattern in reasoning_patterns:
            if re.search(pattern, thinking_lower):
                score += 0.05
        
        # Bonus for longer, more detailed thinking
        if len(thinking_content) > 100:
            score += 0.1
        if len(thinking_content) > 300:
            score += 0.1
            
        return min(score, 0.3)  # Cap thinking bonus

def load_your_model_and_tokenizer():
    """Load model and tokenizer using your exact setup"""
    
    model_path = '/workspace/data/axolotl-outputs/llama_deepseek_2epochs/merged'
    model_path = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        #device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        cache_dir=hf_cache_dir,
    )
    
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

def create_custom_grade_answer_function():
    """
    Create your custom grade_answer function
    You should replace this with your actual implementation
    """
    def grade_answer(model_answer: str, correct_answer: str) -> bool:
        """
        Your custom grading logic here
        This is a placeholder - replace with your actual function
        """
        try:
            # Example: simple numerical comparison
            # You should replace this with your actual grading logic
            model_num = float(model_answer.strip())
            correct_num = float(correct_answer.strip())
            return abs(model_num - correct_num) < 1e-6
        except:
            # Fallback to string comparison
            return model_answer.strip().lower() == correct_answer.strip().lower()
    
    return grade_answer

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
        generation_batch_size=3,
        num_generations = 3,

        # Generation parameters - optimized for your format
        temperature=0.8,
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
        output_dir="workspace/data/grpo-metamath-model",
        
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

def train_with_grpo_custom():
    """Main training function using your custom setup"""
    
    print("Loading your model and tokenizer...")
    model, tokenizer = load_your_model_and_tokenizer()
    
    print("Preparing MetaMathQA dataset...")
    dataset = prepare_metamath_dataset()  # Add your hf_cache_dir if needed
    
    print("Setting up custom grading function...")
    grade_answer_func = create_custom_grade_answer_function()
    
    print("Creating reward function...")
    reward_function = MetaMathRewardFunction(tokenizer, grade_answer_func)
    
    print("Setting up GRPO configuration...")
    grpo_config = setup_grpo_config()
    
    print("Initializing GRPO trainer...")
    grpo_trainer = CustomGRPOTrainer(
        config=grpo_config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_function=reward_function,
    )
    
    print(f"Starting GRPO training with {len(dataset)} examples...")
    print(f"Generating {grpo_config.num_sample_generations} responses per prompt for group comparison")
    
    # Train the model
    grpo_trainer.train()
    
    # Save the trained model
    print("Saving trained model...")
    grpo_trainer.save_model("./grpo-metamath-final")
    tokenizer.save_pretrained("./grpo-metamath-final")
    
    print("Training completed!")

def train_with_grpo_multigpu():
    """Multi-GPU version"""
    
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print("Setting up multi-GPU GRPO training...")
    
    # Load model and tokenizer
    model, tokenizer = load_your_model_and_tokenizer()
    dataset = prepare_metamath_dataset()
    grade_answer_func = create_custom_grade_answer_function()
    reward_function = MetaMathRewardFunction(tokenizer, grade_answer_func)
    
    # Setup config
    grpo_config = setup_grpo_config()
    
    # Adjust batch sizes for multi-GPU
    #num_gpus = int(os.environ.get('WORLD_SIZE', 1))
    #grpo_config.batch_size = max(1, grpo_config.batch_size // num_gpus)
    #grpo_config.mini_batch_size = max(1, grpo_config.mini_batch_size // num_gpus)
    
    # Prepare model with accelerator
    #model = accelerator.prepare(model)
    
    # Initialize trainer
    grpo_trainer = CustomGRPOTrainer(
        config=grpo_config,
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[reward_function],
    )
    
    #if accelerator.is_main_process:
    #    print(f"Training on {num_gpus} GPUs")
    #    print(f"Effective batch size per GPU: {grpo_config.batch_size}")
    
    # Train
    grpo_trainer.train()
    
    # Save only on main process
    if accelerator.is_main_process:
        grpo_trainer.save_model("./grpo-metamath-multigpu")
        tokenizer.save_pretrained("./grpo-metamath-multigpu")

def test_reward_function():
    """Test the reward function with your dataset format"""
    
    print("Testing reward function...")
    
    # Load model/tokenizer for testing
    model, tokenizer = load_your_model_and_tokenizer()
    grade_answer_func = create_custom_grade_answer_function()
    reward_function = MetaMathRewardFunction(tokenizer, grade_answer_func)
    
    # Create test cases in your format
    test_cases = [
        {
            "prompt": "Answer the following question, and format your answer as 'The answer is: <answer>'. Here is the question: What is 2 + 3?",
            "completion": "<think>I need to add 2 and 3. 2 + 3 = 5</think>\nThe answer is: 5",
            "response": "To solve this problem, I need to add the two numbers.\n\n2 + 3 = 5\n\nThe answer is: 5"
        },
        {
            "prompt": "Answer the following question, and format your answer as 'The answer is: <answer>'. Here is the question: What is 10 * 7?",
            "completion": "10 * 7 = 70\nThe answer is: 70",
            "response": "I need to multiply 10 by 7.\n\n10 × 7 = 70\n\nThe answer is: 70"
        },
        {
            "prompt": "Answer the following question, and format your answer as 'The answer is: <answer>'. Here is the question: What is 15 / 3?",
            "completion": "The answer is: 6",  # Wrong answer
            "response": "To divide 15 by 3:\n\n15 ÷ 3 = 5\n\nThe answer is: 5"
        }
    ]
    
    prompts = [case["prompt"] for case in test_cases]
    completions = [case["completion"] for case in test_cases]
    responses = [case["response"] for case in test_cases]
    
    rewards = reward_function(prompts, completions, response=responses)
    
    print("\nReward function test results:")
    for i, (case, reward) in enumerate(zip(test_cases, rewards)):
        model_answer = reward_function.extract_model_answer(case["completion"])
        solution_answer = reward_function.extract_solution_answer(case["response"])
        is_correct = grade_answer_func(model_answer, solution_answer)
        
        print(f"Case {i+1}: Reward = {reward:.3f}")
        print(f"  Model answer: '{model_answer}'")
        print(f"  Solution answer: '{solution_answer}'")
        print(f"  Correct: {is_correct}")
        print(f"  Has thinking: {'</think>' in case['completion']}")
        print()

if __name__ == "__main__":
    train_with_grpo_multigpu()

#    print("MetaMathQA GRPO Training")
#    print("========================")
#    print("Choose option:")
#    print("1. Single GPU GRPO Training")
#    print("2. Multi-GPU GRPO Training") 
#    print("3. Test Reward Function")
#    
#    choice = input("Enter choice (1-3): ")
#    
#    if choice == "1":
#        train_with_grpo_custom()
#    elif choice == "2":
#        train_with_grpo_multigpu()
#    elif choice == "3":
#        test_reward_function()
#    else:
#        print("Invalid choice")