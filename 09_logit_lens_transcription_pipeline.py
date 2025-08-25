import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import re
import pandas as pd
import torch
import torch.nn.functional as F
from utils_activations import rot13_alpha, LlamaActivationExtractor, logit_lens_single_layer

from config import hf_cache_dir

def main():
    path = '/workspace/data/axolotl-outputs/llama_deepseek_2epochs/merged'
    prompt_path = './prompts/three_hop_prompts.csv'
    df = pd.read_csv(prompt_path)

    print('initializing extractor')
    activation_extractor = LlamaActivationExtractor(
        model_name_or_path=path,
        layer_defaults=[40, 54, 56, 58, 60, 62],
        cache_dir=hf_cache_dir,
        )
    activation_extractor.overwrite_chat_template()
    model = activation_extractor.model
    tokenizer = activation_extractor.tokenizer

    col_names = ['rot13', 'layer 40', 'layer 58', 'layer 58 + conf', 'layers 54-62', 'layers 54-62 + conf']
    col_args = [
        {'layers_to_average': [], 'confidence_threshold': 0.},
        {'layers_to_average': ['layer_40'], 'confidence_threshold': 0.},
        {'layers_to_average': ['layer_58'], 'confidence_threshold': 0.},
        {'layers_to_average': ['layer_58'], 'confidence_threshold': 0.7},
        {'layers_to_average': [f'layer_{i}' for i in range(54, 64, 2)], 'confidence_threshold': 0.},
        {'layers_to_average': [f'layer_{i}' for i in range(54, 64, 2)], 'confidence_threshold': 0.7},
    ]
    print('processing df with transcription')
    df = process_df_with_logit_lens_transcription(
        df, activation_extractor, col_names=col_names, col_args=col_args)
    del activation_extractor.model, model
    torch.cuda.empty_cache()
    print('paraphrasing')
    df = process_df_with_model_paraphrase(df, col_names)
    output_csv_path = "prompts/three_hop_prompts_w_logit_lens_translations.csv"
    df.to_csv(output_csv_path, index=False)

def generate_logit_lens_transcript(self, activations: Dict[str, torch.Tensor], layer_names: List[str], confidence_threshold: float = 0.5):
    """
    Uses logit lens with confidence threshold to generate a transcript from model activations.

    Args:
        activations: Dictionary of layer activations.
        layer_names: List of layer names to average logits over.
        confidence_threshold: Probability threshold to highlight tokens.
    """

    # Ensure all specified layers are in the activations
    for layer_name in layer_names:
        if layer_name not in activations:
            raise ValueError(f"Layer {layer_name} not found in activations.")

    # Collect logits for the specified layers
    logits_list = [logit_lens_single_layer(self, activations[layer_name]) for layer_name in layer_names]

    # Average the logits over the specified layers
    averaged_logits = torch.mean(torch.stack(logits_list), dim=0)

    # Get probabilities of the top token (softmax over vocabulary dimension)
    probabilities = F.softmax(averaged_logits, dim=-1)
    top_token_probs, top_token_ids = torch.max(probabilities, dim=-1)

    # Convert to numpy for plotting
    top_token_probs = top_token_probs.detach().float().cpu().numpy()

    # Decode tokens that exceed the confidence threshold
    top_token_ids = top_token_ids.detach().cpu().numpy()
    tokens_above_threshold = [
        self.tokenizer.decode([token_id])
        for token_id, prob in zip(top_token_ids, top_token_probs)
        if prob >= confidence_threshold
    ]

    # Remove consecutively repeating tokens
    filtered_tokens = [tokens_above_threshold[0]]
    for token in tokens_above_threshold[1:]:
        prev_token = filtered_tokens[-1].lower()
        curr_token = token.lower()
        if not prev_token.endswith(curr_token) and not prev_token.startswith(curr_token):
            filtered_tokens.append(token)

    transcript = " ".join(filtered_tokens)
    transcript = transcript.replace("<｜end▁of▁sentence｜>", "")
    transcript = transcript.replace("<｜begin▁of▁sentence｜>", "")
    return transcript

# Function to paraphrase corrupted transcripts
def paraphrase_transcript(paraphrasing_model, tokenizer, transcript):
    prompt = (
        "Here is a transcript that has been corrupted. Can you reconstruct what the original paragraph says "
        "and provide a legible version? "
        "This is the corrupted transcript: " + "\"" + transcript + "\""
    )
    inputs = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=False) # TODO: check here
    inputs = {k: v.to(paraphrasing_model.device) for k, v in inputs.items()}
    outputs = paraphrasing_model.generate(**inputs, max_new_tokens=1000, do_sample=False)
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    match = re.search(r'"(.*?)"', generated_text, re.DOTALL)
    if match:
        paraphrased_text = match.group(1)
    else:
        paraphrased_text = "Error"
    return paraphrased_text

def process_df_with_logit_lens_transcription(
        df, activation_extractor, col_names, col_args):
    model_outputs = []
    translated_thinkings = []
    is_correct_list = []
    logit_lens_transcripts = [[] for _ in col_names]

    for index, row in df.iterrows():
        prompt = row['Prompt']
        answer = row['Answer']
        print(index)
        print(prompt)

        # Format the prompt using the chat template
        formatted_prompt = activation_extractor.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate model response and activations
        generation_results = activation_extractor.generate_with_activations(
            formatted_prompt,
            do_sample=False,
            max_new_tokens=1500,
        )

        # Extract model output
        generated_text = generation_results['response']
        model_outputs.append(generated_text)

        # Translate thinking using rot13_alpha
        thinking_content = generated_text.split('</think>')[0].strip('\n')
        thinking_content = thinking_content.replace("<｜end▁of▁sentence｜>", "")
        thinking_content = thinking_content.replace("<｜begin▁of▁sentence｜>", "")
        translated_thinkings.append(rot13_alpha(thinking_content))

        # Evaluate correctness
        if "</think>" in generated_text:
            content_after_think = generated_text.split("</think>", 1)[1].strip()
            is_correct = answer.lower() in content_after_think.lower()
        else:
            is_correct = False
        is_correct_list.append(is_correct)

        # Generate logit lens transcript
        for col_index, col_name, col_arg in zip(range(len(col_names)), col_names, col_args):
            layers_to_average = col_arg['layers_to_average']
            confidence_threshold = col_arg['confidence_threshold']
            if len(layers_to_average) == 0:  # just use rot13 output, no logit lens transcript
                transcript = thinking_content
            else:
                transcript = generate_logit_lens_transcript(
                    activation_extractor, generation_results['token_activations'], layers_to_average, confidence_threshold)
            logit_lens_transcripts[col_index].append(transcript)

    # Add the new columns to the DataFrame
    df['model_output'] = model_outputs
    df['translated_thinking'] = translated_thinkings
    df['is_correct'] = is_correct_list
    for col_index, col_name in enumerate(col_names):
        df[f'logit_lens_transcript_{col_name}'] = logit_lens_transcripts[col_index]
    return df

def process_df_with_model_paraphrase(df, col_names):
    paraphrasing_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    paraphrasing_tokenizer = AutoTokenizer.from_pretrained(paraphrasing_model_name)
    paraphrasing_model = AutoModelForCausalLM.from_pretrained(
        paraphrasing_model_name, torch_dtype=torch.float16, device_map="auto")
    for col_name in col_names:
        paraphrased_transcripts = []
        for transcript in df[f"logit_lens_transcript_{col_name}"]:
           _paraphrased_transcript = paraphrase_transcript(paraphrasing_model, paraphrasing_tokenizer, transcript)
           paraphrased_transcripts.append(_paraphrased_transcript)

        # Add the new column to the DataFrame
        df[f"paraphrased_transcript_{col_name}"] = paraphrased_transcripts
    return df

if __name__ == "__main__":
    main()