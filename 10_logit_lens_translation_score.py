#!/usr/bin/env python3
"""
Script to evaluate similarity between ground-truth transcripts and paraphrased transcripts
using Llama3.3-70B-Instruct model.
"""

import pandas as pd
import numpy as np
import re
import random
from typing import Optional, Dict, List
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import hf_cache_dir

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranscriptScorer:
    def __init__(self, model_name: str = "meta-llama/Llama-3.3-70B-Instruct"):
        """Initialize the model for scoring transcript similarity."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the Llama model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                cache_dir=hf_cache_dir
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def create_comparison_prompt(self, transcript_a: str, transcript_b: str) -> list:
        """Create the comparison prompt for the model in chat format."""
        messages = [
            {
                "role": "user",
                "content": f"""I have two transcripts. This is transcript A: "{transcript_a}". This is transcript B: "{transcript_b}". On a scale of 0-10, how similar are these two texts? 0 means that each sentence is discussing a different subject or content. 10 means that the content is effectively the same and that each sentence conveys the exact same meaning (minor wording differences don't matter). Phrase your answer as "Answer: {{number}}/10"."""
            }
        ]
        return messages
    
    def get_similarity_score(self, transcript_a: str, transcript_b: str) -> Optional[float]:
        """Get similarity score between two transcripts."""
        if pd.isna(transcript_a) or pd.isna(transcript_b):
            return None
            
        messages = self.create_comparison_prompt(transcript_a, transcript_b)
        
        try:
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            logger.info(response)
            print(response)

            # Extract score from response
            score = self.extract_score(response)
            return score
            
        except Exception as e:
            logger.error(f"Error getting similarity score: {e}")
            return None
    
    def extract_score(self, response: str) -> Optional[float]:
        """Extract numerical score from model response."""
        # Look for pattern "Answer: X/10" or variations
        patterns = [
            r"Answer:\s*(\d+(?:\.\d+)?)/10",
            r"Answer:\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"Answer:\s*(\d+(?:\.\d+)?)",  # New pattern for "Answer: 6" format
            r"(\d+(?:\.\d+)?)/10",
            r"(\d+(?:\.\d+)?)\s*out\s*of\s*10",
            r"score\s*(?:is|of)?\s*(\d+(?:\.\d+)?)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # Ensure score is in valid range
                    return max(0, min(10, score))
                except ValueError:
                    continue
        
        logger.warning(f"Could not extract score from response: {response}")
        return None

def load_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV file with transcript data."""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise

def get_paraphrased_columns(df: pd.DataFrame) -> List[str]:
    """Get all columns that start with 'paraphrased_transcript'."""
    paraphrased_cols = [col for col in df.columns if col.startswith('paraphrased_transcript')]
    logger.info(f"Found paraphrased transcript columns: {paraphrased_cols}")
    return paraphrased_cols

def create_null_comparison_pairs(df: pd.DataFrame) -> List[tuple]:
    """Create pairs for null comparison (cross-row comparisons)."""
    n_rows = len(df)
    if n_rows < 2:
        logger.warning("Not enough rows for null comparison")
        return []
    
    # Create random pairs ensuring we don't compare a row with itself
    pairs = []
    for i in range(n_rows):
        # Pick a random different row
        other_rows = [j for j in range(n_rows) if j != i]
        if other_rows:
            j = random.choice(other_rows)
            pairs.append((i, j))
    
    return pairs

def main():
    """Main function to process the transcripts and generate similarity scores."""
    # Configuration
    input_csv_path = "prompts/three_hop_prompts_w_logit_lens_translations.csv"
    output_csv_path = "transcript_similarity_scores.csv"
    
    # Check if input file exists
    if not Path(input_csv_path).exists():
        logger.error(f"Input file not found: {input_csv_path}")
        return
    
    # Load data
    df = load_data(input_csv_path)
    
    # Get paraphrased transcript columns
    paraphrased_cols = get_paraphrased_columns(df)
    
    # Initialize scorer
    scorer = TranscriptScorer()
    scorer.load_model()
    
    # Create result dataframe
    result_df = df.copy()
    
    # Process each paraphrased transcript column
    for col in paraphrased_cols:
        logger.info(f"Processing column: {col}")
        score_col_name = f"{col}_similarity_score"
        scores = []
        
        for idx, row in df.iterrows():
            logger.info(f"Processing row {idx+1}/{len(df)} for column {col}")
            
            translated_thinking = row['translated_thinking']
            paraphrased_transcript = row[col]
            
            score = scorer.get_similarity_score(translated_thinking, paraphrased_transcript)
            scores.append(score)
            
            if score is not None:
                logger.info(f"Row {idx}: Score = {score}/10")
            else:
                logger.warning(f"Row {idx}: Could not get score")
        
        result_df[score_col_name] = scores
    
    # Create null comparison scores
    logger.info("Creating null comparison scores...")
    null_pairs = create_null_comparison_pairs(df)
    null_scores = []
    
    # Use the first paraphrased column for null comparison
    if paraphrased_cols and null_pairs:
        comparison_col = paraphrased_cols[0]
        
        for i, (row_i, row_j) in enumerate(null_pairs):
            logger.info(f"Processing null comparison {i+1}/{len(null_pairs)}")
            
            translated_thinking_i = df.iloc[row_i]['translated_thinking']
            paraphrased_transcript_j = df.iloc[row_j][comparison_col]
            
            score = scorer.get_similarity_score(translated_thinking_i, paraphrased_transcript_j)
            null_scores.append(score)
            
            if score is not None:
                logger.info(f"Null comparison {i}: Score = {score}/10")
    else:
        null_scores = [None] * len(df)
    
    result_df['null_comparison'] = null_scores
    
    # Save results
    result_df.to_csv(output_csv_path, index=False)
    logger.info(f"Results saved to {output_csv_path}")
    
    # Print summary statistics
    logger.info("Summary Statistics:")
    for col in paraphrased_cols:
        score_col = f"{col}_similarity_score"
        if score_col in result_df.columns:
            scores = result_df[score_col].dropna()
            if len(scores) > 0:
                logger.info(f"{col}: Mean = {scores.mean():.2f}, Std = {scores.std():.2f}")
    
    null_scores_clean = result_df['null_comparison'].dropna()
    if len(null_scores_clean) > 0:
        logger.info(f"Null comparison: Mean = {null_scores_clean.mean():.2f}, Std = {null_scores_clean.std():.2f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    main() 