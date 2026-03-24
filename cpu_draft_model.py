"""
CPU Kernel Draft Model for Speculative Decoding.

Wraps lb_kernels/littlebit_kernels_cpu's DummyLlama3LittleBitModel
to provide the same interface as MatryoshkaDraftModel.

Key design: Persistent KV cache for incremental token generation.
Uses C++ ops for embedding, RMSNorm, lm_head, and SiLU to eliminate
Python overhead.

Usage:
    from cpu_draft_model import CPUDraftModel
    model = CPUDraftModel(runtime_path, base_model_id)
    tokens, probs = model.generate_draft_tokens(input_ids, draft_length=5)
"""

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# Add lb_kernels to path
_LB_KERNELS_ROOT = Path(__file__).resolve().parent / "lb_kernels"
if str(_LB_KERNELS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LB_KERNELS_ROOT))

from littlebit_kernels_cpu import (
    DummyLlama3Config,
    load_runtime_checkpoint,
    littlebit_linear,
)
from littlebit_kernels_cpu.dummy_model import (
    load_dummy_llama3_model_from_state,
    DummyLlama3LittleBitModel,
)
from utils.misc import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# C++ op wrappers — fall back to Python if not available
# ============================================================================

def _has_cpp_op(name: str) -> bool:
    """Check if a C++ op is available."""
    return (
        hasattr(torch.ops, "littlebit_cpu_ops")
        and hasattr(torch.ops.littlebit_cpu_ops, name)
    )


def _cpp_embedding(weight: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Embedding lookup via C++ op or Python fallback."""
    if _has_cpp_op("embedding_lookup"):
        return torch.ops.littlebit_cpu_ops.embedding_lookup(weight, token_ids)
    return F.embedding(token_ids.to("cpu"), weight)


def _cpp_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm via C++ op or Python fallback."""
    if _has_cpp_op("rms_norm"):
        return torch.ops.littlebit_cpu_ops.rms_norm(x, weight, eps)
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def _cpp_lm_head(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Dense GEMV via C++ op or Python fallback."""
    if _has_cpp_op("dense_gemv_f32"):
        h = hidden.reshape(1, -1).to(torch.float32).contiguous()
        return torch.ops.littlebit_cpu_ops.dense_gemv_f32(h, weight)
    return F.linear(hidden.to(torch.float32), weight)


def _cpp_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(gate) * up via C++ op or Python fallback."""
    if _has_cpp_op("silu_mul"):
        return torch.ops.littlebit_cpu_ops.silu_mul(gate.contiguous(), up.contiguous())
    return F.silu(gate) * up


class CPUDraftModel:
    """CPU kernel-based draft model for speculative decoding.
    
    Uses persistent KV cache for efficient incremental generation.
    Uses C++ ops for embedding, RMSNorm, lm_head, and SiLU when available.
    """
    
    def __init__(
        self,
        runtime_path: str,
        base_model_id: str,
        torch_dtype=torch.float32,
    ):
        runtime_path = Path(runtime_path)
        
        # Load runtime checkpoint
        logger.info(f"Loading CPU runtime checkpoint from {runtime_path}...")
        state_dict, runtime_config = load_runtime_checkpoint(
            runtime_path, device="cpu"
        )
        
        # Load dummy config
        dummy_config_path = runtime_path / "dummy_llama3_config.json"
        with open(dummy_config_path, "r") as f:
            dummy_config_dict = json.load(f)
        
        # Remove non-config keys
        extra_keys = ["include_lm_head", "eff_bit", "residual"]
        for k in extra_keys:
            dummy_config_dict.pop(k, None)
        
        config = DummyLlama3Config(**dummy_config_dict)
        
        # Build LittleBit model from runtime state
        self.lb_model = load_dummy_llama3_model_from_state(
            state_dict, config, device="cpu"
        )
        logger.info(f"CPU LittleBit model loaded: {config.num_hidden_layers} layers, "
                     f"hidden_size={config.hidden_size}")
        
        # Load embedding and lm_head from base model (dense, kept on CPU)
        logger.info(f"Loading embeddings & lm_head from {base_model_id}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        # Extract embedding and lm_head weights
        self.embed_tokens = base_model.model.embed_tokens.weight.data.to(torch.float32).clone()
        self.lm_head_weight = base_model.lm_head.weight.data.to(torch.float32).clone()
        
        # Check if embed_tokens is in the runtime state
        if "model.embed_tokens.weight" in state_dict:
            self.embed_tokens = state_dict["model.embed_tokens.weight"].to(torch.float32)
            logger.info("Using embed_tokens from runtime checkpoint")
        
        del base_model
        import gc; gc.collect()
        
        self.config = config
        self.device = torch.device("cpu")
        self.max_seq_len = 4096
        
        # Persistent KV cache
        self._cache = None
        self._cache_pos = 0        # Next position to write in cache
        self._cached_seq_len = 0   # How many input tokens have been cached
        self._last_hidden = None   # Last hidden state from transformer

        # Log C++ op availability
        cpp_ops = ["embedding_lookup", "rms_norm", "dense_gemv_f32", "silu_mul"]
        avail = [name for name in cpp_ops if _has_cpp_op(name)]
        logger.info(f"C++ ops available: {avail if avail else 'none (using Python fallback)'}")
        logger.info("CPU draft model ready")
    
    def reset(self):
        """Reset KV cache. Call when prompt changes."""
        self._cache = None
        self._cache_pos = 0
        self._cached_seq_len = 0
        self._last_hidden = None
    
    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings via C++ op."""
        return _cpp_embedding(self.embed_tokens, token_ids)
    
    def _lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden state to logits via C++ dense GEMV."""
        return _cpp_lm_head(hidden, self.lm_head_weight)
    
    def _ensure_cache(self):
        """Allocate KV cache if not yet allocated."""
        if self._cache is None:
            self._cache = self.lb_model.allocate_cache(self.max_seq_len)
            self._cache_pos = 0
            self._cached_seq_len = 0
    
    @torch.no_grad()
    def _forward_token(self, token_id: int) -> torch.Tensor:
        """Process a single token through the transformer, updating KV cache.
        Uses C++ ops for RMSNorm and SiLU where available.
        Returns the hidden state (1, hidden_size).
        """
        self._ensure_cache()
        
        # Embedding lookup (C++ op)
        hidden = self._embed(torch.tensor([token_id], dtype=torch.long))
        hidden = hidden.reshape(1, self.config.hidden_size)
        
        # Run through all layers using C++ RMSNorm and SiLU
        from littlebit_kernels_cpu.runtime import littlebit_linear as lb_linear_fn
        from littlebit_kernels_cpu.dummy_model import (
            _group_query_heads, _cache_write_grouped, _grouped_attention_context
        )
        
        x = hidden.to(torch.float32)
        eps = self.config.rms_norm_eps
        
        for layer, layer_cache in zip(self.lb_model.layers, self._cache):
            residual = x
            
            # RMSNorm (C++ op)
            normed = _cpp_rms_norm(x, layer.input_layernorm_weight, eps)
            
            # Q, K, V projections (LittleBit kernel)
            q = lb_linear_fn(normed, layer.q_proj).to(torch.float32)
            k = lb_linear_fn(normed, layer.k_proj).to(torch.float32)
            v = lb_linear_fn(normed, layer.v_proj).to(torch.float32)
            
            # Attention with KV cache
            q_grouped = _group_query_heads(
                q,
                num_key_value_heads=self.config.num_key_value_heads,
                kv_repeat=self.lb_model.kv_repeat,
                head_dim=self.lb_model.head_dim,
            )
            keys, values = _cache_write_grouped(
                layer_cache, k, v,
                position=self._cache_pos,
                num_key_value_heads=self.config.num_key_value_heads,
                head_dim=self.lb_model.head_dim,
            )
            attn_out = _grouped_attention_context(
                q_grouped, keys, values,
                attn_scale=self.lb_model.attn_scale,
            )
            
            x = residual + lb_linear_fn(attn_out, layer.o_proj).to(torch.float32)
            
            # MLP
            residual = x
            mlp_in = _cpp_rms_norm(x, layer.post_attention_layernorm_weight, eps)
            gate = lb_linear_fn(mlp_in, layer.gate_proj).to(torch.float32)
            up = lb_linear_fn(mlp_in, layer.up_proj).to(torch.float32)
            
            # Fused SiLU * up (C++ op)
            mlp_hidden = _cpp_silu_mul(gate, up)
            x = residual + lb_linear_fn(mlp_hidden, layer.down_proj).to(torch.float32)
        
        # Final RMSNorm (C++ op)
        final_hidden = _cpp_rms_norm(x, self.lb_model.final_norm_weight, eps)
        
        self._cache_pos += 1
        self._last_hidden = final_hidden
        return final_hidden
    
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor):
        """Process input tokens and cache KV states. 
        Only processes tokens not yet cached.
        """
        input_ids = input_ids.to("cpu")
        seq_len = input_ids.shape[1]
        
        # If we already cached some prefix and input extends it, only process new tokens
        if self._cached_seq_len > 0 and seq_len > self._cached_seq_len:
            start = self._cached_seq_len
        elif self._cached_seq_len == 0:
            start = 0
        else:
            # seq_len <= cached: prompt changed, need full reset
            self.reset()
            start = 0
        
        self._ensure_cache()
        
        for pos in range(start, seq_len):
            token_id = input_ids[0, pos].item()
            self._forward_token(token_id)
        
        self._cached_seq_len = seq_len
    
    @torch.no_grad()
    def generate_draft_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        draft_length: int = 5,
        temperature: float = 1.0,
        greedy: bool = True,
    ):
        """Generate K draft tokens autoregressively using CPU kernel.
        
        Uses persistent KV cache — only processes new tokens in input_ids
        that haven't been cached yet, then generates K draft tokens.
        The K draft positions are rolled back after generation so the
        cache stays at the prefix boundary for the next call.
        """
        input_ids = input_ids.to("cpu")
        seq_len = input_ids.shape[1]
        
        # Check if prompt changed (different prefix)
        if self._cached_seq_len > seq_len:
            self.reset()
        
        # Prefill: only process uncached tokens
        self.prefill(input_ids)
        
        # Save cache position before generating drafts
        cache_pos_before_draft = self._cache_pos
        
        # Generate K draft tokens
        draft_tokens = []
        draft_probs = []
        last_hidden = self._last_hidden  # (1, hidden_size)
        
        for k in range(draft_length):
            # LM head via C++ dense GEMV
            logits = self._lm_head(last_hidden)  # (1, vocab_size)
            
            if greedy:
                probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (1, 1)
            else:
                probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            draft_tokens.append(next_token)
            draft_probs.append(probs.unsqueeze(0))
            
            # Forward next token through transformer (updates cache)
            token_id = next_token.reshape(-1)[0].item()
            last_hidden = self._forward_token(token_id)
        
        # Roll back cache position — draft tokens are speculative
        self._cache_pos = cache_pos_before_draft
        self._cached_seq_len = seq_len
        
        return draft_tokens, draft_probs
