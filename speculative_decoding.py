"""
Speculative Decoding with Matryoshka LittleBit Models

Supports two decode modes:
1. "serial": Standard sequential draft → verify (default)
2. "tree": EAGLE-style tree attention — drafts a tree of candidates,
           verifies all branches in ONE target model forward pass

Supports three target modes:
1. "fp": Draft=0.1-bit, Target=original FP model (pre-Step2)
2. "matryoshka": Draft=0.1-bit, Target=0.1+0.9-bit combined (post-Step2)
3. "littlebit_cpu": Draft+Target both use C++ LittleBit kernel on CPU

Usage (serial, FP target):
    python speculative_decoding.py \
        --draft_model_path outputs/step1_draft_0.1bit/<ts> \
        --target_mode fp --decode_mode serial --draft_length 5

Usage (tree, FP target):
    python speculative_decoding.py \
        --draft_model_path outputs/step1_draft_0.1bit/<ts> \
        --target_mode fp --decode_mode tree --tree_preset default
"""

import argparse
import json
import os
import time
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization.utils.quant_util import load_quantized_model
from quantization.modules import LittleBitLinear
from utils.datautils import load_tokenizer
from utils.misc import setup_logger
from tree_utils import (
    generate_tree_buffers, generate_draft_tree,
    evaluate_posterior, build_tree_attention_mask,
    TREE_PRESETS,
)

logger = setup_logger(__name__)


def str2bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception(f'Boolean value expected: {value}')


class MatryoshkaDraftModel:
    """0.1-bit draft model for fast token generation."""
    
    def __init__(self, model_path, torch_dtype=torch.bfloat16, device="cuda"):
        config_path = os.path.join(model_path, "littlebit_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        quant_args = argparse.Namespace(
            quant_func=config.get("quant_func", "STEBinary"),
            quant_mod=config.get("quant_mod", "LittleBitLinear"),
            eff_bit=config.get("eff_bit", 0.1),
            split_dim=config.get("split_dim", 1024),
            residual=config.get("residual", False),
            kv_factor=config.get("kv_factor", 1.0),
            min_split_dim=config.get("min_split_dim", 8),
            model_id=model_path,
        )
        
        self.model = load_quantized_model(
            model_path=model_path,
            quant_args=quant_args,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        logger.info(f"Draft model loaded from {model_path} on {self.device}")
    
    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
    
    @torch.no_grad()
    def generate_draft_tokens(self, input_ids, attention_mask=None, 
                               draft_length=5, temperature=1.0, greedy=True):
        """Generate K draft tokens autoregressively."""
        draft_tokens = []
        draft_probs = []
        current_ids = input_ids
        current_mask = attention_mask
        past_key_values = None
        
        for k in range(draft_length):
            outputs = self.forward(
                input_ids=current_ids,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[:, -1, :]  # (batch, vocab)
            past_key_values = outputs.past_key_values
            
            if greedy:
                probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            draft_tokens.append(next_token)
            draft_probs.append(probs)
            
            current_ids = next_token
            if current_mask is not None:
                current_mask = torch.cat([
                    current_mask,
                    torch.ones(current_mask.shape[0], 1, device=current_mask.device, dtype=current_mask.dtype)
                ], dim=1)
        
        return draft_tokens, draft_probs


class FPTargetModel:
    """Original FP (full-precision) model as target for speculative decoding.
    
    Use this when benchmarking before Step 2 (residual training).
    Draft = 0.1-bit quantized model, Target = original FP model.
    """
    
    def __init__(self, model_id, torch_dtype=torch.bfloat16, device="cuda"):
        logger.info(f"Loading FP target model from {model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        logger.info(f"FP target model loaded on {self.device}")
    
    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits


class MatryoshkaTargetModel:
    """Combined 0.1-bit + 0.9-bit target model for verification."""
    
    def __init__(self, draft_model_path, residual_model_path, 
                 torch_dtype=torch.bfloat16, device="cuda"):
        # Load draft model
        draft_config_path = os.path.join(draft_model_path, "littlebit_config.json")
        with open(draft_config_path, 'r') as f:
            draft_config = json.load(f)
        
        draft_quant_args = argparse.Namespace(
            quant_func=draft_config.get("quant_func", "STEBinary"),
            quant_mod=draft_config.get("quant_mod", "LittleBitLinear"),
            eff_bit=draft_config.get("eff_bit", 0.1),
            split_dim=draft_config.get("split_dim", 1024),
            residual=draft_config.get("residual", False),
            kv_factor=draft_config.get("kv_factor", 1.0),
            min_split_dim=draft_config.get("min_split_dim", 8),
            model_id=draft_model_path,
        )
        
        self.draft_model = load_quantized_model(
            model_path=draft_model_path,
            quant_args=draft_quant_args,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.draft_model.eval()
        
        # Load residual model
        residual_config_path = os.path.join(residual_model_path, "littlebit_config.json")
        with open(residual_config_path, 'r') as f:
            residual_config = json.load(f)
        
        residual_quant_args = argparse.Namespace(
            quant_func=residual_config.get("quant_func", "STEBinary"),
            quant_mod=residual_config.get("quant_mod", "LittleBitLinear"),
            eff_bit=residual_config.get("eff_bit", 0.9),
            split_dim=residual_config.get("split_dim", 1024),
            residual=residual_config.get("residual", False),
            kv_factor=residual_config.get("kv_factor", 1.0),
            min_split_dim=residual_config.get("min_split_dim", 8),
            model_id=residual_model_path,
        )
        
        self.residual_model = load_quantized_model(
            model_path=residual_model_path,
            quant_args=residual_quant_args,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.residual_model.eval()
        
        self.device = next(self.draft_model.parameters()).device
        logger.info(f"Target model loaded (draft + residual) on {self.device}")
    
    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None):
        """Forward pass: sum logits from draft and residual models."""
        draft_outputs = self.draft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        residual_outputs = self.residual_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        
        combined_logits = draft_outputs.logits + residual_outputs.logits
        return combined_logits


def speculative_decode(
    draft_model: MatryoshkaDraftModel,
    target_model,  # FPTargetModel or MatryoshkaTargetModel
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_new_tokens: int = 256,
    draft_length: int = 5,
    temperature: float = 1.0,
    greedy: bool = True,
    eos_token_id: int = 2,
    verbose: bool = False,
):
    """
    Perform speculative decoding.
    
    Args:
        draft_model: Fast 0.1-bit model for draft generation
        target_model: Target model for verification (FP or Matryoshka)
        input_ids: Input token IDs (batch=1)
        attention_mask: Attention mask
        max_new_tokens: Maximum new tokens to generate
        draft_length: Number of draft tokens per step (K)
        temperature: Sampling temperature
        greedy: Whether to use greedy decoding
        eos_token_id: End of sequence token ID
        verbose: Print step-by-step info
    
    Returns:
        generated_ids: All generated token IDs
        stats: Dictionary with timing and acceptance statistics
    """
    device = input_ids.device
    generated_tokens = []
    total_draft_tokens = 0
    total_accepted_tokens = 0
    num_steps = 0
    
    current_ids = input_ids
    current_mask = attention_mask
    
    start_time = time.time()
    
    while len(generated_tokens) < max_new_tokens:
        num_steps += 1
        remaining = max_new_tokens - len(generated_tokens)
        k = min(draft_length, remaining)
        
        # === Draft Phase: Generate K draft tokens ===
        draft_tokens, draft_probs = draft_model.generate_draft_tokens(
            input_ids=current_ids,
            attention_mask=current_mask,
            draft_length=k,
            temperature=temperature,
            greedy=greedy,
        )
        
        # Build verification sequence: original + K draft tokens
        draft_token_ids = torch.cat(draft_tokens, dim=1).to(device)  # (1, K) — ensure same device as target
        verify_ids = torch.cat([current_ids, draft_token_ids], dim=1)  # (1, seq_len + K)
        if current_mask is not None:
            verify_mask = torch.cat([
                current_mask,
                torch.ones(1, k, device=device, dtype=current_mask.dtype)
            ], dim=1)
        else:
            verify_mask = None
        
        # === Verification Phase: Run target model on full sequence ===
        target_logits = target_model.forward(verify_ids, verify_mask)  # (1, seq_len + K, vocab)
        
        # The target logits at position i correspond to predicting token i+1
        seq_len = current_ids.shape[1]
        
        # === Accept/Reject Phase ===
        accepted_count = 0
        for i in range(k):
            target_logit_i = target_logits[:, seq_len - 1 + i, :]
            draft_token_i = draft_tokens[i]
            
            if greedy:
                target_token = torch.argmax(target_logit_i, dim=-1, keepdim=True)
                if target_token.item() == draft_token_i.item():
                    accepted_count += 1
                    generated_tokens.append(draft_token_i.item())
                    if draft_token_i.item() == eos_token_id:
                        break
                else:
                    generated_tokens.append(target_token.item())
                    if target_token.item() == eos_token_id:
                        break
                    break
            else:
                target_prob_i = F.softmax(target_logit_i / max(temperature, 1e-8), dim=-1)
                draft_prob_i = draft_probs[i].to(device) if hasattr(draft_probs[i], 'to') else draft_probs[i]
                
                token_idx = draft_token_i.item()
                p_target = target_prob_i[0, token_idx].item()
                p_draft = draft_prob_i[0, token_idx].item()
                
                accept_prob = min(1.0, p_target / max(p_draft, 1e-10))
                
                if torch.rand(1).item() < accept_prob:
                    accepted_count += 1
                    generated_tokens.append(token_idx)
                    if token_idx == eos_token_id:
                        break
                else:
                    residual_prob = torch.clamp(target_prob_i - draft_prob_i, min=0)
                    residual_prob = residual_prob / (residual_prob.sum() + 1e-10)
                    corrected_token = torch.multinomial(residual_prob, num_samples=1)
                    generated_tokens.append(corrected_token.item())
                    if corrected_token.item() == eos_token_id:
                        break
                    break
        
        # If all K tokens accepted, sample one bonus token from target
        if accepted_count == k and len(generated_tokens) < max_new_tokens:
            bonus_logit = target_logits[:, seq_len + k - 1, :]
            if greedy:
                bonus_token = torch.argmax(bonus_logit, dim=-1).item()
            else:
                bonus_prob = F.softmax(bonus_logit / max(temperature, 1e-8), dim=-1)
                bonus_token = torch.multinomial(bonus_prob, num_samples=1).item()
            generated_tokens.append(bonus_token)
        
        total_draft_tokens += k
        total_accepted_tokens += accepted_count
        
        if verbose:
            logger.info(f"Step {num_steps}: drafted {k}, accepted {accepted_count}, "
                       f"total generated {len(generated_tokens)}")
        
        # Update current sequence
        all_gen = torch.tensor([generated_tokens], device=device, dtype=torch.long)
        current_ids = torch.cat([input_ids, all_gen], dim=1)
        if attention_mask is not None:
            current_mask = torch.ones_like(current_ids)
        
        # Check for EOS
        if generated_tokens[-1] == eos_token_id:
            break
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    stats = {
        "total_tokens_generated": len(generated_tokens),
        "total_draft_tokens": total_draft_tokens,
        "total_accepted_tokens": total_accepted_tokens,
        "num_steps": num_steps,
        "mean_acceptance_length": total_accepted_tokens / max(num_steps, 1),
        "acceptance_rate": total_accepted_tokens / max(total_draft_tokens, 1),
        "tokens_per_second": len(generated_tokens) / max(elapsed, 1e-6),
        "elapsed_seconds": elapsed,
        "draft_length_k": draft_length,
    }
    
    output_ids = torch.cat([input_ids, torch.tensor([generated_tokens], device=device)], dim=1)
    return output_ids, stats


def autoregressive_generate(
    target_model,  # FPTargetModel or MatryoshkaTargetModel
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    greedy: bool = True,
    eos_token_id: int = 2,
):
    """Standard autoregressive generation with target model (baseline)."""
    device = input_ids.device
    generated_tokens = []
    current_ids = input_ids
    current_mask = attention_mask
    
    start_time = time.time()
    
    for _ in range(max_new_tokens):
        target_logits = target_model.forward(current_ids, current_mask)
        next_logits = target_logits[:, -1, :]
        
        if greedy:
            next_token = torch.argmax(next_logits, dim=-1).item()
        else:
            probs = F.softmax(next_logits / max(temperature, 1e-8), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        
        generated_tokens.append(next_token)
        
        next_token_tensor = torch.tensor([[next_token]], device=device)
        current_ids = torch.cat([current_ids, next_token_tensor], dim=1)
        if current_mask is not None:
            current_mask = torch.cat([
                current_mask,
                torch.ones(1, 1, device=device, dtype=current_mask.dtype)
            ], dim=1)
        
        if next_token == eos_token_id:
            break
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    stats = {
        "total_tokens_generated": len(generated_tokens),
        "tokens_per_second": len(generated_tokens) / max(elapsed, 1e-6),
        "elapsed_seconds": elapsed,
    }
    
    output_ids = torch.cat([input_ids, torch.tensor([generated_tokens], device=device)], dim=1)
    return output_ids, stats


def speculative_decode_tree(
    draft_model: MatryoshkaDraftModel,
    target_model,  # FPTargetModel or MatryoshkaTargetModel
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_new_tokens: int = 256,
    tree_buffers: dict = None,
    top_k: int = 10,
    temperature: float = 1.0,
    greedy: bool = True,
    eos_token_id: int = 2,
    verbose: bool = False,
):
    """
    Tree-based speculative decoding (EAGLE-style).
    
    Draft model generates a tree of candidates, target model verifies
    ALL branches in one forward pass using tree attention mask.
    """
    device = input_ids.device
    generated_tokens = []
    total_draft_tokens = 0
    total_accepted_tokens = 0
    num_steps = 0
    
    tree_attn_mask = tree_buffers["tree_attn_mask"]  # (1,1,T,T)
    tree_position_ids = tree_buffers["tree_position_ids"]  # (T,)
    retrieve_indices = tree_buffers["retrieve_indices"]  # (num_leaves, max_depth+1)
    tree_len = tree_buffers["tree_len"]
    
    current_ids = input_ids
    current_mask = attention_mask
    
    start_time = time.time()
    
    while len(generated_tokens) < max_new_tokens:
        num_steps += 1
        
        # === Draft Phase: Generate tree of candidates ===
        tree_candidates = generate_draft_tree(
            draft_model=draft_model,
            input_ids=current_ids,
            attention_mask=current_mask,
            tree_buffers=tree_buffers,
            top_k=top_k,
            temperature=temperature,
        )
        # tree_candidates: (1, tree_len)
        
        # === Verification Phase: Run target on prefix + tree ===
        prefix_len = current_ids.shape[1]
        verify_ids = torch.cat([current_ids, tree_candidates[:, 1:]], dim=1)  # skip root (duplicate)
        
        # Build tree attention mask for the full sequence
        full_attn_mask = build_tree_attention_mask(
            prefix_len=prefix_len,
            tree_attn_mask=tree_attn_mask,
            device=device,
            dtype=torch.bfloat16,
        )
        
        # Adjust: we skip root in tree_candidates, so tree part is tree_len-1
        actual_tree_len = tree_len - 1
        actual_total = prefix_len + actual_tree_len
        full_attn_mask_trimmed = full_attn_mask[:, :, :actual_total, :actual_total]
        
        # Position IDs: prefix uses sequential, tree uses prefix_len + tree_position_ids
        prefix_position_ids = torch.arange(prefix_len, device=device).unsqueeze(0)
        tree_pos = (prefix_len - 1) + tree_position_ids[1:]  # skip root
        tree_pos = tree_pos.unsqueeze(0)
        full_position_ids = torch.cat([prefix_position_ids, tree_pos], dim=1)
        
        # Target model forward — for FP models we pass attention_mask as 2D
        # For tree attention we need 4D mask
        target_logits = target_model.forward(
            verify_ids,
            full_attn_mask_trimmed,
        )
        # target_logits: (1, prefix_len + actual_tree_len, vocab)
        
        # Extract logits for tree nodes
        tree_logits = target_logits[:, prefix_len - 1:, :]  # from last prefix pos onward
        # tree_logits shape: (1, actual_tree_len + 1, vocab)
        
        # Build candidate paths using retrieve_indices
        # Each row in retrieve_indices is a path from root to leaf
        # Pad tree_candidates with -1 for out-of-range indices
        padded_candidates = torch.cat([
            tree_candidates,
            torch.full((1, 1), -1, device=device, dtype=torch.long)
        ], dim=1)
        
        cart_candidates = padded_candidates[0, retrieve_indices]  # (num_leaves, max_depth+1)
        
        # Get logits for each candidate path
        padded_tree_logits = torch.cat([
            tree_logits,
            torch.zeros(1, 1, tree_logits.shape[-1], device=device, dtype=tree_logits.dtype)
        ], dim=1)
        
        cart_logits = padded_tree_logits[0, retrieve_indices]  # (num_leaves, max_depth+1, vocab)
        
        # === Accept/Reject Phase ===
        best_candidate, accept_length, next_token_logits = evaluate_posterior(
            cart_logits, cart_candidates, greedy=greedy, temperature=temperature
        )
        
        # Get accepted tokens
        accepted_path = cart_candidates[best_candidate, 1:accept_length + 1]  # skip root
        for t in accepted_path.tolist():
            if t == -1:
                break
            generated_tokens.append(t)
            if t == eos_token_id:
                break
        
        # Get bonus token from next_token_logits
        if len(generated_tokens) < max_new_tokens and (not generated_tokens or generated_tokens[-1] != eos_token_id):
            if greedy:
                if next_token_logits.dim() == 0:
                    bonus_token = next_token_logits.item()
                else:
                    bonus_token = torch.argmax(next_token_logits).item()
            else:
                if next_token_logits.dim() > 1:
                    next_token_logits = next_token_logits.squeeze()
                probs = F.softmax(next_token_logits / max(temperature, 1e-8), dim=-1)
                bonus_token = torch.multinomial(probs, 1).item()
            generated_tokens.append(bonus_token)
        
        total_draft_tokens += actual_tree_len
        total_accepted_tokens += accept_length
        
        if verbose:
            logger.info(f"Step {num_steps}: tree_len={actual_tree_len}, accepted={accept_length}, "
                       f"total generated={len(generated_tokens)}")
        
        # Update current sequence
        all_gen = torch.tensor([generated_tokens], device=device, dtype=torch.long)
        current_ids = torch.cat([input_ids, all_gen], dim=1)
        if attention_mask is not None:
            current_mask = torch.ones_like(current_ids)
        
        if generated_tokens[-1] == eos_token_id:
            break
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    stats = {
        "decode_mode": "tree",
        "total_tokens_generated": len(generated_tokens),
        "total_draft_tokens": total_draft_tokens,
        "total_accepted_tokens": total_accepted_tokens,
        "num_steps": num_steps,
        "mean_acceptance_length": total_accepted_tokens / max(num_steps, 1),
        "acceptance_rate": total_accepted_tokens / max(total_draft_tokens, 1),
        "tokens_per_second": len(generated_tokens) / max(elapsed, 1e-6),
        "elapsed_seconds": elapsed,
        "tree_size": tree_len,
    }
    
    output_ids = torch.cat([input_ids, torch.tensor([generated_tokens], device=device)], dim=1)
    return output_ids, stats


def load_target_model(args, device):
    """Load target model based on target_mode."""
    if args.target_mode == "fp":
        logger.info("Loading FP (full-precision) target model...")
        return FPTargetModel(
            args.base_model_id, torch_dtype=torch.bfloat16, device=str(device)
        )
    elif args.target_mode == "matryoshka":
        if not args.residual_model_path:
            raise ValueError("--residual_model_path required for matryoshka target mode")
        logger.info("Loading Matryoshka target model (0.1-bit + 0.9-bit)...")
        return MatryoshkaTargetModel(
            args.draft_model_path, args.residual_model_path,
            torch_dtype=torch.bfloat16, device=str(device)
        )
    elif args.target_mode == "littlebit_cpu":
        from cpu_target_model import CPUTargetModel
        draft_rt = getattr(args, 'draft_runtime_path', None)
        residual_rt = getattr(args, 'residual_runtime_path', None)
        if not draft_rt or not residual_rt:
            raise ValueError(
                "--draft_runtime_path and --residual_runtime_path required "
                "for littlebit_cpu target mode"
            )
        logger.info("Loading CPU LittleBit target model (draft + residual kernel)...")
        return CPUTargetModel(
            draft_runtime_path=draft_rt,
            residual_runtime_path=residual_rt,
            base_model_id=args.base_model_id,
        )
    else:
        raise ValueError(f"Unknown target_mode: {args.target_mode}")


def load_draft_model(args, device):
    """Load draft model based on draft_device."""
    draft_device = getattr(args, 'draft_device', 'cuda')
    
    if draft_device == "cpu_kernel":
        logger.info("Loading draft model with CPU kernel...")
        from cpu_draft_model import CPUDraftModel
        return CPUDraftModel(
            runtime_path=args.draft_model_path,
            base_model_id=args.base_model_id,
        )
    else:
        logger.info("Loading draft model (0.1-bit, GPU)...")
        return MatryoshkaDraftModel(
            args.draft_model_path, torch_dtype=torch.bfloat16, device=str(device)
        )


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding with Matryoshka LittleBit")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model ID (for tokenizer, and FP target if target_mode=fp)")
    parser.add_argument("--draft_model_path", type=str, required=True,
                        help="Path to 0.1-bit draft model")
    parser.add_argument("--residual_model_path", type=str, default=None,
                        help="Path to 0.9-bit residual model (required for target_mode=matryoshka)")
    parser.add_argument("--target_mode", type=str, default="fp",
                        choices=["fp", "matryoshka", "littlebit_cpu"],
                        help="Target model type: 'fp'=original FP, 'matryoshka'=0.1+0.9 combined, 'littlebit_cpu'=CPU kernel")
    parser.add_argument("--draft_runtime_path", type=str, default=None,
                        help="Path to draft model runtime checkpoint (for littlebit_cpu target)")
    parser.add_argument("--residual_runtime_path", type=str, default=None,
                        help="Path to residual model runtime checkpoint (for littlebit_cpu target)")
    parser.add_argument("--decode_mode", type=str, default="serial",
                        choices=["serial", "tree"],
                        help="Decode mode: 'serial'=sequential draft, 'tree'=EAGLE tree attention")
    parser.add_argument("--tree_preset", type=str, default="default",
                        choices=["small", "default", "large"],
                        help="Tree structure preset (for tree mode)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-K for tree draft expansion (tree mode only)")
    parser.add_argument("--prompt", type=str, 
                        default="Write a Python function to compute fibonacci numbers efficiently.",
                        help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--draft_length", type=int, default=5,
                        help="Number of draft tokens per speculative step K (serial mode only)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--compare_baseline", type=str2bool, default=True,
                        help="Also run autoregressive baseline for comparison")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--draft_device", type=str, default="cuda",
                        choices=["cuda", "cpu_kernel"],
                        help="Device for draft model: 'cuda'=GPU PyTorch, 'cpu_kernel'=CPU LittleBit kernel")
    
    args = parser.parse_args()
    
    greedy = args.mode == "greedy"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.base_model_id)
    eos_token_id = tokenizer.eos_token_id or 2
    
    # Load models
    draft_model = load_draft_model(args, device)
    
    target_model = load_target_model(args, device)
    
    target_desc = "FP (original)" if args.target_mode == "fp" else "Matryoshka (0.1+0.9 bit)"
    decode_desc = f"serial (K={args.draft_length})" if args.decode_mode == "serial" else f"tree ({args.tree_preset})"
    
    # Prepare tree buffers if tree mode
    tree_buffers = None
    if args.decode_mode == "tree":
        tree_choices = TREE_PRESETS[args.tree_preset]
        tree_buffers = generate_tree_buffers(tree_choices, device=str(device))
        logger.info(f"Tree buffers generated: {tree_buffers['tree_len']} nodes, "
                    f"{tree_buffers['retrieve_indices'].shape[0]} leaf paths")
    
    # Tokenize prompt
    chat_messages = [{"role": "user", "content": args.prompt}]
    try:
        prompt_text = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt_text = args.prompt
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    
    # === Speculative Decoding ===
    logger.info(f"\n{'='*60}")
    logger.info(f"Speculative Decoding ({decode_desc}, mode={args.mode})")
    logger.info(f"  Draft:  0.1-bit quantized")
    logger.info(f"  Target: {target_desc}")
    logger.info(f"{'='*60}")
    
    if args.decode_mode == "serial":
        output_ids, spec_stats = speculative_decode(
            draft_model=draft_model,
            target_model=target_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            draft_length=args.draft_length,
            temperature=args.temperature,
            greedy=greedy,
            eos_token_id=eos_token_id,
            verbose=True,
        )
    else:  # tree
        output_ids, spec_stats = speculative_decode_tree(
            draft_model=draft_model,
            target_model=target_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            tree_buffers=tree_buffers,
            top_k=args.top_k,
            temperature=args.temperature,
            greedy=greedy,
            eos_token_id=eos_token_id,
            verbose=True,
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"\n{'='*60}")
    print(f"SPECULATIVE DECODING RESULTS ({decode_desc})")
    print(f"  Draft:  0.1-bit | Target: {target_desc}")
    print(f"{'='*60}")
    print(f"Generated text:\n{generated_text}")
    print(f"\nStatistics:")
    for key, value in spec_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # === Baseline comparison ===
    if args.compare_baseline:
        logger.info(f"\n{'='*60}")
        logger.info(f"Autoregressive Baseline ({target_desc})")
        logger.info(f"{'='*60}")
        
        baseline_ids, baseline_stats = autoregressive_generate(
            target_model=target_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            greedy=greedy,
            eos_token_id=eos_token_id,
        )
        
        baseline_text = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)
        
        print(f"\n{'='*60}")
        print(f"AUTOREGRESSIVE BASELINE RESULTS ({target_desc})")
        print(f"{'='*60}")
        print(f"Generated text:\n{baseline_text}")
        print(f"\nStatistics:")
        for key, value in baseline_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Speedup
        if baseline_stats["elapsed_seconds"] > 0:
            speedup = baseline_stats["elapsed_seconds"] / max(spec_stats["elapsed_seconds"], 1e-6)
            print(f"\n  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
