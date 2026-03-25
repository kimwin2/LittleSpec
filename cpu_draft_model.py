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

_NEW_OPS_LOADED = False

def _ensure_new_ops():
    """Force-load the C++ extension if new ops are not yet available.
    
    The runtime.py build_extension() may early-return if old ops are cached,
    so we need to explicitly load the freshly-built .so with the new ops.
    """
    global _NEW_OPS_LOADED
    if _NEW_OPS_LOADED:
        return
    
    # Check if new ops are already available
    if (hasattr(torch.ops, "littlebit_cpu_ops") 
        and hasattr(torch.ops.littlebit_cpu_ops, "embedding_lookup")):
        _NEW_OPS_LOADED = True
        return
    
    # Try to force-load from built .so
    try:
        from torch.utils.cpp_extension import load
        import shutil
        
        kernels_dir = _LB_KERNELS_ROOT / "littlebit_kernels_cpu"
        source_file = kernels_dir / "littlebit_cpu.cpp"
        build_dir = kernels_dir / "build"
        
        if source_file.exists():
            # Clear old build to force fresh compilation with new ops
            if build_dir.exists():
                # Check if .so already exists (from script build step)
                so_files = list(build_dir.glob("*.so"))
                if so_files:
                    logger.info(f"Force-loading extension from {so_files[0]}...")
            
            build_dir.mkdir(parents=True, exist_ok=True)
            load(
                name="littlebit_cpu_ops",
                sources=[str(source_file)],
                build_directory=str(build_dir),
                extra_cflags=["-O3", "-march=native"],
                with_cuda=False,
                is_python_module=False,
                verbose=False,
            )
            _NEW_OPS_LOADED = True
            logger.info("New C++ ops force-loaded successfully")
    except Exception as e:
        logger.warning(f"Could not force-load C++ extension: {e}")


def _has_cpp_op(name: str) -> bool:
    """Check if a C++ op is available."""
    _ensure_new_ops()
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
        cpp_ops = ["generate_token", "quantize_lm_head_q4", "full_forward",
                    "repack_signs_to_i2", "embedding_lookup", "rms_norm"]
        avail = [name for name in cpp_ops if _has_cpp_op(name)]
        logger.info(f"C++ ops available: {avail if avail else 'none (using Python fallback)'}")
        
        # Build flat tensor list and dims for monolithic C++ forward
        self._use_full_forward = False
        self._layer_tensors = []  # flat list: 37 tensors per layer
        self._layer_dims = []     # flat list: 28 ints per layer
        
        if (_has_cpp_op("full_forward") and _has_cpp_op("repack_signs_to_i2")):
            logger.info("Preparing monolithic C++ forward pass...")
            import time
            t0 = time.perf_counter()
            
            proj_names = ["q_proj", "k_proj", "v_proj", "o_proj", 
                           "gate_proj", "up_proj", "down_proj"]
            
            for layer_idx, layer in enumerate(self.lb_model.layers):
                # [0] input_layernorm_weight
                self._layer_tensors.append(layer.input_layernorm_weight.contiguous())
                # [1] post_attention_layernorm_weight
                self._layer_tensors.append(layer.post_attention_layernorm_weight.contiguous())
                
                for proj_name in proj_names:
                    rt = getattr(layer, proj_name)
                    branch = rt.main
                    v_cols = int(branch.v_shape[1])
                    rank = int(branch.v_shape[0])  # = u_shape[1]
                    u_cols = int(branch.u_shape[1])
                    out_features = int(branch.u_shape[0])
                    
                    # Repack signs to i2
                    v_sign_i2 = torch.ops.littlebit_cpu_ops.repack_signs_to_i2(
                        branch.v_sign.contiguous(), v_cols)
                    u_sign_i2 = torch.ops.littlebit_cpu_ops.repack_signs_to_i2(
                        branch.u_sign.contiguous(), u_cols)
                    
                    # [2+p*5+0] v2
                    self._layer_tensors.append(branch.v2.to(torch.float32).contiguous().squeeze(0))
                    # [2+p*5+1] v_sign_i2
                    self._layer_tensors.append(v_sign_i2.contiguous())
                    # [2+p*5+2] mid
                    self._layer_tensors.append(branch.mid.to(torch.float32).contiguous().squeeze(0))
                    # [2+p*5+3] u_sign_i2
                    self._layer_tensors.append(u_sign_i2.contiguous())
                    # [2+p*5+4] u1
                    self._layer_tensors.append(branch.u1.to(torch.float32).contiguous().squeeze(0))
                    
                    # Dims: [v_cols, rank, u_cols, out_features]
                    self._layer_dims.extend([v_cols, rank, u_cols, out_features])
            
            self._use_full_forward = True
            logger.info(f"Full forward prep done in {time.perf_counter() - t0:.2f}s "
                        f"({len(self._layer_tensors)} tensors, {len(self._layer_dims)} dims)")
        
        # Q4 quantize lm_head for generate_token_cpu
        self._use_generate_token = False
        self._lm_head_q4 = None
        self._vocab_size = 0
        if (_has_cpp_op("generate_token") and _has_cpp_op("quantize_lm_head_q4")
            and self._use_full_forward):
            logger.info("Quantizing lm_head to Q4_0...")
            import time
            t0 = time.perf_counter()
            self._lm_head_q4 = torch.ops.littlebit_cpu_ops.quantize_lm_head_q4(
                self.lm_head_weight)
            self._vocab_size = self.lm_head_weight.shape[0]
            self._use_generate_token = True
            q4_mb = self._lm_head_q4.numel() / (1024 * 1024)
            orig_mb = self.lm_head_weight.numel() * 4 / (1024 * 1024)
            logger.info(f"lm_head Q4 quantized in {time.perf_counter() - t0:.2f}s "
                        f"({orig_mb:.0f}MB -> {q4_mb:.0f}MB, {orig_mb/q4_mb:.1f}x compression)")
        
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
            # Build flat KV cache tensor list for C++ full_forward
            if self._use_full_forward:
                self._kv_cache_tensors = []
                for layer_cache in self._cache:
                    self._kv_cache_tensors.append(layer_cache.key)
                    self._kv_cache_tensors.append(layer_cache.value)
    
    @torch.no_grad()
    def _forward_token(self, token_id: int, profile: bool = False) -> torch.Tensor:
        """Process a single token through the transformer, updating KV cache.
        Uses monolithic C++ forward pass when available (1 C++ call for all layers).
        Returns the hidden state (1, hidden_size).
        """
        import time
        self._ensure_cache()
        
        if self._use_full_forward:
            # ===== MONOLITHIC C++ PATH =====
            # One C++ call for embedding + 32 layers + final norm
            t0 = time.perf_counter() if profile else 0
            
            hidden = torch.ops.littlebit_cpu_ops.full_forward(
                token_id,
                self.embed_tokens,
                self.lb_model.final_norm_weight.contiguous(),
                self._layer_tensors,
                self._kv_cache_tensors,
                self._layer_dims,
                self.config.num_hidden_layers,
                self.config.hidden_size,
                self.config.num_key_value_heads,
                self.lb_model.kv_repeat,
                self.lb_model.head_dim,
                self.max_seq_len,
                self._cache_pos,
                self.lb_model.attn_scale,
                self.config.rms_norm_eps,
            )
            
            if profile:
                t_fwd = time.perf_counter() - t0
                logger.info(f"Token profile: full_forward={t_fwd*1000:.0f}ms (monolithic C++)")
            
            self._cache_pos += 1
            self._last_hidden = hidden
            return hidden
        
        # ===== FALLBACK: PER-LAYER PYTHON PATH =====
        if profile:
            t_embed = 0; t_rms = 0; t_lb = 0; t_attn = 0; t_silu = 0
        
        t0 = time.perf_counter() if profile else 0
        hidden = self._embed(torch.tensor([token_id], dtype=torch.long))
        hidden = hidden.reshape(1, self.config.hidden_size)
        if profile: t_embed += time.perf_counter() - t0
        
        from littlebit_kernels_cpu.runtime import littlebit_linear as lb_linear_fn
        from littlebit_kernels_cpu.dummy_model import (
            _group_query_heads, _cache_write_grouped, _grouped_attention_context
        )
        
        x = hidden.to(torch.float32)
        eps = self.config.rms_norm_eps
        
        for layer_idx, (layer, layer_cache) in enumerate(zip(self.lb_model.layers, self._cache)):
            residual = x
            t0 = time.perf_counter() if profile else 0
            normed = _cpp_rms_norm(x, layer.input_layernorm_weight, eps)
            if profile: t_rms += time.perf_counter() - t0
            
            t0 = time.perf_counter() if profile else 0
            q = lb_linear_fn(normed, layer.q_proj).to(torch.float32)
            k = lb_linear_fn(normed, layer.k_proj).to(torch.float32)
            v = lb_linear_fn(normed, layer.v_proj).to(torch.float32)
            if profile: t_lb += time.perf_counter() - t0
            
            t0 = time.perf_counter() if profile else 0
            q_grouped = _group_query_heads(q, num_key_value_heads=self.config.num_key_value_heads,
                kv_repeat=self.lb_model.kv_repeat, head_dim=self.lb_model.head_dim)
            keys, values = _cache_write_grouped(layer_cache, k, v,
                position=self._cache_pos, num_key_value_heads=self.config.num_key_value_heads,
                head_dim=self.lb_model.head_dim)
            attn_out = _grouped_attention_context(q_grouped, keys, values,
                attn_scale=self.lb_model.attn_scale)
            if profile: t_attn += time.perf_counter() - t0
            
            t0 = time.perf_counter() if profile else 0
            x = residual + lb_linear_fn(attn_out, layer.o_proj).to(torch.float32)
            if profile: t_lb += time.perf_counter() - t0
            
            residual = x
            t0 = time.perf_counter() if profile else 0
            mlp_in = _cpp_rms_norm(x, layer.post_attention_layernorm_weight, eps)
            if profile: t_rms += time.perf_counter() - t0
            
            t0 = time.perf_counter() if profile else 0
            gate = lb_linear_fn(mlp_in, layer.gate_proj).to(torch.float32)
            up = lb_linear_fn(mlp_in, layer.up_proj).to(torch.float32)
            if profile: t_lb += time.perf_counter() - t0
            
            t0 = time.perf_counter() if profile else 0
            mlp_hidden = _cpp_silu_mul(gate, up)
            if profile: t_silu += time.perf_counter() - t0
            
            t0 = time.perf_counter() if profile else 0
            x = residual + lb_linear_fn(mlp_hidden, layer.down_proj).to(torch.float32)
            if profile: t_lb += time.perf_counter() - t0
        
        t0 = time.perf_counter() if profile else 0
        final_hidden = _cpp_rms_norm(x, self.lb_model.final_norm_weight, eps)
        if profile: t_rms += time.perf_counter() - t0
        
        self._cache_pos += 1
        self._last_hidden = final_hidden
        
        if profile:
            total = t_embed + t_rms + t_lb + t_attn + t_silu
            logger.info(
                f"Token profile (fallback): total={total*1000:.0f}ms | "
                f"LB={t_lb*1000:.0f}ms | Attn={t_attn*1000:.0f}ms | "
                f"RMS={t_rms*1000:.0f}ms | SiLU={t_silu*1000:.0f}ms"
            )
        
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
        
        if self._use_generate_token:
            # ===== PURE C++ PATH =====
            # Each call: full_forward + Q4 lm_head + argmax in ONE C++ call
            last_token_id = input_ids[0, -1].item() if self._last_hidden is None else None
            
            for k in range(draft_length):
                if k == 0 and self._last_hidden is not None:
                    # First draft: we already have last_hidden from prefill
                    # Need lm_head + argmax for first token, then generate_token for rest
                    logits = self._lm_head(self._last_hidden)
                    if greedy:
                        probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    else:
                        probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    draft_tokens.append(next_token)
                    draft_probs.append(probs.unsqueeze(0))
                    token_id = next_token.reshape(-1)[0].item()
                else:
                    if k == 0:
                        token_id = last_token_id
                    # Pure C++ generate: forward + Q4 lm_head + argmax
                    token_id = torch.ops.littlebit_cpu_ops.generate_token(
                        token_id,
                        self.embed_tokens,
                        self.lb_model.final_norm_weight.contiguous(),
                        self._layer_tensors,
                        self._kv_cache_tensors,
                        self._layer_dims,
                        self._lm_head_q4,
                        self._vocab_size,
                        self.config.num_hidden_layers,
                        self.config.hidden_size,
                        self.config.num_key_value_heads,
                        self.lb_model.kv_repeat,
                        self.lb_model.head_dim,
                        self.max_seq_len,
                        self._cache_pos,
                        self.lb_model.attn_scale,
                        self.config.rms_norm_eps,
                    )
                    self._cache_pos += 1
                    next_token = torch.tensor([[token_id]], dtype=torch.long)
                    # For pure C++ path, we don't have logits/probs
                    # Create dummy uniform probs (speculative decoding will use target logits)
                    draft_tokens.append(next_token)
                    dummy_probs = torch.zeros(1, self._vocab_size)
                    dummy_probs[0, token_id] = 1.0
                    draft_probs.append(dummy_probs.unsqueeze(0))
        else:
            # ===== FALLBACK PYTHON PATH =====
            last_hidden = self._last_hidden  # (1, hidden_size)
            
            for k in range(draft_length):
                logits = self._lm_head(last_hidden)
                
                if greedy:
                    probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                draft_tokens.append(next_token)
                draft_probs.append(probs.unsqueeze(0))
                
                token_id = next_token.reshape(-1)[0].item()
                last_hidden = self._forward_token(token_id)
        
        # Roll back cache position — draft tokens are speculative
        self._cache_pos = cache_pos_before_draft
        self._cached_seq_len = seq_len
        
        return draft_tokens, draft_probs
