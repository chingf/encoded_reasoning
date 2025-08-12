import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import warnings
from typing import Dict, List, Optional
import pickle
import codecs
import re
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import defaultdict

def rot13_alpha(text):
    # Split text around '<think>' and '</think>'
    segments = re.split(r'(<think>|</think>)', text)
    converted_segments = []

    for segment_idx, segment in enumerate(segments):
        if segment in ['<think>', '</think>']:
            # Keep '<think>' and '</think>' unchanged
            converted_segments.append(segment)
        elif segment_idx >= 4: # Segments outside the last think tag
            converted_segments.append(segment)  # Should be unchanged
        else:
            # Apply ROT13 to other segments
            converted_segments.append(codecs.encode(segment, 'rot_13'))
    
    # Reassemble the text
    reassembled_text = ''.join(converted_segments)
    return reassembled_text

def logit_lens_single_layer(self, 
                           activation: torch.Tensor, 
                           apply_layer_norm: bool = True) -> torch.Tensor:
    """
    Apply logit lens to a single layer's activations.
    
    Args:
        activation: Tensor of shape (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        apply_layer_norm: Whether to apply layer normalization before projection
        
    Returns:
        Logits tensor of shape (seq_len, vocab_size) or (batch, seq_len, vocab_size)
    """
    # Ensure activation is on the correct device
    activation = activation.to(self.device)
    
    # Apply layer normalization if requested (this is typically done in the final layer)
    if apply_layer_norm:
        activation = self.model.model.norm(activation)
    
    # Project to vocabulary space
    logits = self.model.lm_head(activation)

    return logits

class LlamaActivationExtractor:
    def __init__(
            self,
            model_name_or_path: str = "meta-llama/Llama-3.3-70B-Instruct",
            layer_defaults: str = None,
            cache_dir: str = None,
            ):
        """
        Initialize the Llama model for both generation and activation extraction.
        Note: You'll need to request access to the model on Hugging Face.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,  # Use float16 for memory efficiency; also could be float16
            device_map="auto",          # Automatically distribute across available GPUs
            trust_remote_code=True,
            low_cpu_mem_usage=True, 
            cache_dir=cache_dir,
        )
        self.layer_defaults = layer_defaults  # default layers to extract if not specified
        
        # Storage for activations
        self.activations = {}
        self.hooks = []

    def overwrite_chat_template(self, template_path: str='chat_templates/deepseek_distill_llama_template.jinja'):
        """
        Overwrite the chat template used by the model.
        
        Args:
            template_path: Path to the new chat template file.
        """
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")
        with open(template_path, 'r') as f:
            template_content = f.read()
        self.tokenizer.chat_template = template_content
    
    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """
        Register forward hooks to extract activations from specific layers.
        
        Args:
            layer_names: List of layer names to extract from. If None, extracts from all layers.
        """
        def hook_fn(name):
            def hook(module, input, output):
                # Store the output activation
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
                else:
                    self.activations[name] = output.detach().cpu()
            return hook
        
        # Clear previous hooks
        self.clear_hooks()
        
        # Register hooks for specified layers or all layers
        if layer_names is None:
            for i, layer in enumerate(self.model.model.layers):
                if self.layer_defaults is None:
                    pass
                elif self.layer_defaults == 'even':
                    if i % 2 != 0: continue
                else:
                    raise ValueError(f"Unknown layer defaults: {self.layer_defaults}")
                hook_name = f"layer_{i}"
                hook = layer.register_forward_hook(hook_fn(hook_name))
                self.hooks.append(hook)
        else: # Register hooks for specific named modules
            for name, module in self.model.named_modules():
                if any(layer_name in name for layer_name in layer_names):
                    hook = module.register_forward_hook(hook_fn(name))
                    self.hooks.append(hook)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def generate_with_activations(self, 
                                prompt: str, 
                                max_new_tokens: int = 100,
                                do_sample: bool = True,
                                temperature: float = 0.7,
                                top_p: float = 0.9,
                                extract_layers: Optional[List[str]] = None) -> Dict:
        """
        Generate text while extracting activations from specified layers.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            extract_layers: List of layer names to extract activations from
            
        Returns:
            Dictionary containing generated text and activations
        """

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_token_len = len(inputs['input_ids'][0])
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Second pass: Extract activations for the full sequence
        self.register_hooks(extract_layers)
        with torch.no_grad():
            _ = self.model(input_ids=outputs.sequences)
        token_activations = {k: v.squeeze()[input_token_len:] for k, v in self.activations.items()}
        
        # Clip text and tokens to just the assistant's response
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        assistant_response = generated_text.split(prompt)[1].strip('<|eot_id|>')
        assistant_response_tokens = outputs.sequences[0][input_token_len:]
        
        self.clear_hooks()
        
        generation_results = {
            "prompt": prompt,
            "response": assistant_response,
            "response_token_ids": assistant_response_tokens,
            "response_tokens": [self.tokenizer.decode(t) for t in assistant_response_tokens],
            "token_activations": token_activations,
            "generation_scores": outputs.scores if hasattr(outputs, 'scores') else None
        }

        return generation_results

    
    def extract_activations_only(self, 
                               text: str, 
                               extract_layers: Optional[List[str]] = None) -> Dict:
        """
        Extract activations from text without generation.
        
        Args:
            text: Input text
            extract_layers: List of layer names to extract activations from
            
        Returns:
            Dictionary containing activations
        """
        # Register hooks for activation extraction
        self.register_hooks(extract_layers)
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass to extract activations
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Copy activations before clearing hooks
        extracted_activations = {k: v.clone() for k, v in self.activations.items()}

        self.clear_hooks()

        return {
            "text": text,
            "activations": extracted_activations,
            "input_ids": inputs["input_ids"].cpu(),
            "tokens": self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        }
    
    def get_layer_names(self) -> List[str]:
        """Get all available layer names in the model."""
        return [name for name, _ in self.model.named_modules()]
    
    def analyze_activation_shapes(self, activations: Dict) -> Dict:
        """Analyze the shapes of extracted activations."""
        shapes = {}
        for name, activation in activations.items():
            shapes[name] = {
                "shape": list(activation.shape),
                "dtype": str(activation.dtype),
                "size_mb": activation.numel() * activation.element_size() / (1024 * 1024)
            }
        return shapes

# Usage examples
def main():
    # Initialize the extractor
    extractor = LlamaActivationExtractor()
    
    # Example 1: Generate text with activation extraction
    print("=== Generation with Activation Extraction ===")
    prompt = "The future of artificial intelligence is"
    
    # Extract from specific layers (first few layers as example)
    target_layers = ["layer_0", "layer_1", "layer_2"]
    
    result = extractor.generate_with_activations(
        prompt=prompt,
        max_new_tokens=50,
        extract_layers=target_layers
    )
    
    print(f"Generated text: {result['generated_text']}")
    print(f"Extracted activations from: {list(result['activations'].keys())}")
    
    # Analyze activation shapes
    shapes = extractor.analyze_activation_shapes(result['activations'])
    print("Activation shapes:")
    for layer, info in shapes.items():
        print(f"  {layer}: {info['shape']} ({info['size_mb']:.2f} MB)")
    
    # Example 2: Extract activations from existing text
    print("\n=== Activation Extraction Only ===")
    text = "Machine learning is a subset of artificial intelligence."
    
    activation_result = extractor.extract_activations_only(
        text=text,
        extract_layers=["layer_0", "layer_10", "layer_20"]
    )
    
    print(f"Extracted activations for: {activation_result['text']}")
    print(f"Tokens: {activation_result['tokens'][:10]}...")  # Show first 10 tokens
    
    # Example 3: Extract from all layers (memory intensive!)
    print("\n=== All Layer Extraction (Small Example) ===")
    small_result = extractor.extract_activations_only("Hello world!")
    print(f"Total layers extracted: {len(small_result['activations'])}")
    
    # Clean up
    extractor.clear_hooks()