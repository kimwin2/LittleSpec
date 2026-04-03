"""
CPU Target Model for Speculative Decoding.

Loads TWO LittleBit models (draft + residual) using C++ kernel,
runs them on CPU with persistent KV cache, and sums their logits
for target verification in speculative decoding.

Key optimisation:
    - Persistent KV cache for each model unit
    - Common-prefix detection: only processes NEW tokens each call
    - Hidden-state summation before lm_head (saves one dense GEMV)

Usage:
    from cpu_target_model import CPUTargetModel
    model = CPUTargetModel(
        draft_runtime_path="path/to/draft_runtime",
        residual_runtime_path="path/to/residual_runtime",
        base_model_id="meta-llama/Llama-3.1-8B-Instruct",
    )
    logits = model.forward(input_ids)  # (1, seq_len, vocab)
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List

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
)
from littlebit_kernels_cpu.dummy_model import (
    load_dummy_llama3_model_from_state,
    DummyLlama3LittleBitModel,
)
from utils.misc import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# C++ op helpers (same as cpu_draft_model.py)
# ============================================================================

_NEW_OPS_LOADED = False


def _ensure_new_ops():
    """Force-load the C++ extension if new ops are not yet available."""
    global _NEW_OPS_LOADED
    if _NEW_OPS_LOADED:
        return

    if (hasattr(torch.ops, "littlebit_cpu_ops")
            and hasattr(torch.ops.littlebit_cpu_ops, "embedding_lookup")):
        _NEW_OPS_LOADED = True
        return

    try:
        from torch.utils.cpp_extension import load

        kernels_dir = _LB_KERNELS_ROOT / "littlebit_kernels_cpu"
        source_file = kernels_dir / "littlebit_cpu.cpp"
        build_dir = kernels_dir / "build"

        if source_file.exists():
            build_dir.mkdir(parents=True, exist_ok=True)
            load(
                name="littlebit_cpu_ops",
                sources=[str(source_file)],
                build_directory=str(build_dir),
                extra_cflags=["-O3", "-march=native", "-fopenmp"],
                extra_ldflags=["-lgomp"],
                with_cuda=False,
                is_python_module=False,
                verbose=False,
            )
            _NEW_OPS_LOADED = True
            logger.info("C++ ops loaded successfully")
    except Exception as e:
        logger.warning(f"Could not force-load C++ extension: {e}")


def _has_cpp_op(name: str) -> bool:
    _ensure_new_ops()
    return (
        hasattr(torch.ops, "littlebit_cpu_ops")
        and hasattr(torch.ops.littlebit_cpu_ops, name)
    )


def _cpp_embedding(weight, token_ids):
    if _has_cpp_op("embedding_lookup"):
        return torch.ops.littlebit_cpu_ops.embedding_lookup(weight, token_ids)
    return F.embedding(token_ids.to("cpu"), weight)


def _cpp_rms_norm(x, weight, eps):
    if _has_cpp_op("rms_norm"):
        return torch.ops.littlebit_cpu_ops.rms_norm(x, weight, eps)
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def _cpp_lm_head(hidden, weight):
    if _has_cpp_op("dense_gemv_f32"):
        h = hidden.reshape(1, -1).to(torch.float32).contiguous()
        return torch.ops.littlebit_cpu_ops.dense_gemv_f32(h, weight)
    return F.linear(hidden.to(torch.float32), weight)


def _cpp_silu_mul(gate, up):
    if _has_cpp_op("silu_mul"):
        return torch.ops.littlebit_cpu_ops.silu_mul(
            gate.contiguous(), up.contiguous()
        )
    return F.silu(gate) * up


# ============================================================================
# Internal model unit — one LittleBit model with KV cache
# ============================================================================

class _LBModelUnit:
    """One LittleBit model loaded via C++ kernel, with KV cache management."""

    def __init__(self, runtime_path: str, config: DummyLlama3Config, name: str = ""):
        runtime_path = Path(runtime_path)
        self.name = name
        self.config = config
        self.max_seq_len = 4096

        # Load runtime checkpoint
        state_dict, runtime_config = load_runtime_checkpoint(
            runtime_path, device="cpu"
        )

        # Build LittleBit model
        self.lb_model = load_dummy_llama3_model_from_state(
            state_dict, config, device="cpu"
        )
        logger.info(
            f"[{name}] LittleBit model loaded: "
            f"{config.num_hidden_layers} layers, hidden={config.hidden_size}"
        )

        # KV cache state
        self._cache = None
        self._cache_pos = 0

        # Set up C++ full_forward infrastructure
        self._use_full_forward = False
        self._layer_tensors = []
        self._layer_dims = []

        if _has_cpp_op("full_forward") and _has_cpp_op("repack_signs_to_i2"):
            logger.info(f"[{name}] Preparing monolithic C++ forward pass...")
            t0 = time.perf_counter()

            proj_names = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

            for layer in self.lb_model.layers:
                self._layer_tensors.append(
                    layer.input_layernorm_weight.contiguous()
                )
                self._layer_tensors.append(
                    layer.post_attention_layernorm_weight.contiguous()
                )

                for proj_name in proj_names:
                    rt = getattr(layer, proj_name)
                    branch = rt.main
                    v_cols = int(branch.v_shape[1])
                    rank = int(branch.v_shape[0])
                    u_cols = int(branch.u_shape[1])
                    out_features = int(branch.u_shape[0])

                    v_sign_i2 = torch.ops.littlebit_cpu_ops.repack_signs_to_i2(
                        branch.v_sign.contiguous(), v_cols
                    )
                    u_sign_i2 = torch.ops.littlebit_cpu_ops.repack_signs_to_i2(
                        branch.u_sign.contiguous(), u_cols
                    )

                    self._layer_tensors.append(
                        branch.v2.to(torch.float32).contiguous().squeeze(0)
                    )
                    self._layer_tensors.append(v_sign_i2.contiguous())
                    self._layer_tensors.append(
                        branch.mid.to(torch.float32).contiguous().squeeze(0)
                    )
                    self._layer_tensors.append(u_sign_i2.contiguous())
                    self._layer_tensors.append(
                        branch.u1.to(torch.float32).contiguous().squeeze(0)
                    )

                    self._layer_dims.extend([v_cols, rank, u_cols, out_features])

            self._use_full_forward = True
            elapsed = time.perf_counter() - t0
            logger.info(
                f"[{name}] Full forward prep done in {elapsed:.2f}s "
                f"({len(self._layer_tensors)} tensors, "
                f"{len(self._layer_dims)} dims)"
            )

    def _ensure_cache(self):
        """Allocate KV cache if not yet allocated."""
        if self._cache is None:
            self._cache = self.lb_model.allocate_cache(self.max_seq_len)
            self._cache_pos = 0
            if self._use_full_forward:
                self._kv_cache_tensors = []
                for layer_cache in self._cache:
                    self._kv_cache_tensors.append(layer_cache.key)
                    self._kv_cache_tensors.append(layer_cache.value)

    def reset(self):
        """Reset KV cache.  Call when prompt changes."""
        self._cache = None
        self._cache_pos = 0

    @torch.no_grad()
    def forward_token(self, token_id: int, embed_tokens: torch.Tensor) -> torch.Tensor:
        """Process one token through the transformer, updating KV cache.

        Uses monolithic C++ forward when available.
        Returns the final hidden state (1, hidden_size) after RMSNorm.
        """
        self._ensure_cache()

        if self._use_full_forward:
            hidden = torch.ops.littlebit_cpu_ops.full_forward(
                token_id,
                embed_tokens,
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
            self._cache_pos += 1
            return hidden

        # ===== Fallback: per-layer Python path =====
        from littlebit_kernels_cpu.runtime import littlebit_linear as lb_linear_fn
        from littlebit_kernels_cpu.dummy_model import (
            _group_query_heads, _cache_write_grouped, _grouped_attention_context,
        )

        hidden = _cpp_embedding(
            embed_tokens, torch.tensor([token_id], dtype=torch.long)
        )
        hidden = hidden.reshape(1, self.config.hidden_size)
        x = hidden.to(torch.float32)
        eps = self.config.rms_norm_eps

        for layer, layer_cache in zip(self.lb_model.layers, self._cache):
            residual = x
            normed = _cpp_rms_norm(x, layer.input_layernorm_weight, eps)

            q = lb_linear_fn(normed, layer.q_proj).to(torch.float32)
            k = lb_linear_fn(normed, layer.k_proj).to(torch.float32)
            v = lb_linear_fn(normed, layer.v_proj).to(torch.float32)

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

            residual = x
            mlp_in = _cpp_rms_norm(
                x, layer.post_attention_layernorm_weight, eps
            )
            gate = lb_linear_fn(mlp_in, layer.gate_proj).to(torch.float32)
            up = lb_linear_fn(mlp_in, layer.up_proj).to(torch.float32)
            mlp_hidden = _cpp_silu_mul(gate, up)
            x = residual + lb_linear_fn(mlp_hidden, layer.down_proj).to(torch.float32)

        final_hidden = _cpp_rms_norm(
            x, self.lb_model.final_norm_weight, eps
        )
        self._cache_pos += 1
        return final_hidden


# ============================================================================
# CPUTargetModel — the public class
# ============================================================================

class CPUTargetModel:
    """Combined LittleBit target model (draft + residual) on CPU.

    Loads two LittleBit runtime checkpoints using C++ kernel, runs both
    on CPU, and sums their logits for target verification in speculative
    decoding.

    Uses persistent KV cache with common-prefix detection to avoid
    reprocessing already-cached tokens between successive calls.

    Optimisation:  since both models share the same lm_head weight (from
    the base model), we sum the hidden states first and apply lm_head only
    once:  ``lm_head(h_d + h_r)`` — this halves the lm_head cost.
    """

    def __init__(
        self,
        draft_runtime_path: str,
        residual_runtime_path: str,
        base_model_id: str,
        torch_dtype=torch.float32,
    ):
        draft_runtime_path = Path(draft_runtime_path)
        residual_runtime_path = Path(residual_runtime_path)

        # Load config (both checkpoints share the same architecture)
        draft_config_path = draft_runtime_path / "dummy_llama3_config.json"
        with open(draft_config_path, "r") as f:
            config_dict = json.load(f)
        for key in ("include_lm_head", "eff_bit", "residual"):
            config_dict.pop(key, None)
        config = DummyLlama3Config(**config_dict)

        # Load two LB model units -----------------------------------------------
        logger.info("Loading draft LB model unit...")
        self.draft_unit = _LBModelUnit(
            draft_runtime_path, config, name="draft"
        )

        logger.info("Loading residual LB model unit...")
        self.residual_unit = _LBModelUnit(
            residual_runtime_path, config, name="residual"
        )

        # Load shared embedding & lm_head from base model -----------------------
        logger.info(f"Loading embeddings & lm_head from {base_model_id}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        self.embed_tokens = (
            base_model.model.embed_tokens.weight.data.to(torch.float32).clone()
        )
        self.lm_head_weight = (
            base_model.lm_head.weight.data.to(torch.float32).clone()
        )
        self._vocab_size = self.lm_head_weight.shape[0]

        del base_model
        gc.collect()

        self.device = torch.device("cpu")

        # Track cached input tokens for KV cache reuse --------------------------
        self._cached_tokens: Optional[List[int]] = None

        # Log C++ op availability
        cpp_ops = [
            "full_forward", "repack_signs_to_i2",
            "embedding_lookup", "rms_norm", "dense_gemv_f32",
        ]
        avail = [name for name in cpp_ops if _has_cpp_op(name)]
        logger.info(
            f"C++ ops available: "
            f"{avail if avail else 'none (Python fallback)'}"
        )
        logger.info("CPU target model ready (draft + residual)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_cache(self):
        """Reset KV caches for both model units."""
        self.draft_unit.reset()
        self.residual_unit.reset()
        self._cached_tokens = None

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
    ) -> torch.Tensor:
        """Process full sequence and return combined logits.

        Uses persistent KV cache: only processes tokens beyond the common
        prefix shared with the previous call's input.

        Args:
            input_ids:  (1, seq_len) token IDs
            attention_mask:  ignored (for API compatibility)

        Returns:
            logits:  (1, seq_len, vocab_size) — combined draft + residual
        """
        input_ids = input_ids.to("cpu")
        seq_len = input_ids.shape[1]

        # --- Find common prefix with cached tokens ---------------------------
        common = 0
        if self._cached_tokens is not None:
            max_common = min(seq_len, len(self._cached_tokens))
            for i in range(max_common):
                if input_ids[0, i].item() != self._cached_tokens[i]:
                    break
                common += 1

        # --- Roll back KV caches to the common prefix position ---------------
        self.draft_unit._ensure_cache()
        self.residual_unit._ensure_cache()
        self.draft_unit._cache_pos = common
        self.residual_unit._cache_pos = common

        # --- Process tokens from <common> onward -----------------------------
        new_logits: List[torch.Tensor] = []

        for pos in range(common, seq_len):
            token_id = input_ids[0, pos].item()

            hidden_d = self.draft_unit.forward_token(
                token_id, self.embed_tokens
            )
            hidden_r = self.residual_unit.forward_token(
                token_id, self.embed_tokens
            )

            # Sum hidden states, then one lm_head — mathematically equivalent
            # to lm_head(h_d) + lm_head(h_r) since lm_head is linear.
            combined_hidden = hidden_d + hidden_r
            combined_logits = _cpp_lm_head(combined_hidden, self.lm_head_weight)
            new_logits.append(combined_logits.squeeze(0))  # (vocab_size,)

        # --- Save tokens for next call's prefix comparison -------------------
        self._cached_tokens = input_ids[0].tolist()

        # --- Build full (1, seq_len, vocab) tensor ---------------------------
        if not new_logits:
            # Edge case: all tokens were already cached
            return torch.zeros(1, seq_len, self._vocab_size)

        new_logits_tensor = torch.stack(new_logits, dim=0)  # (N_new, V)

        if common > 0:
            # Pad cached positions with zeros — speculative_decode never
            # accesses logits at positions < current_prefix_len - 1,
            # so the zeros are harmless.
            pad = torch.zeros(common, self._vocab_size)
            full_logits = torch.cat([pad, new_logits_tensor], dim=0)
        else:
            full_logits = new_logits_tensor

        return full_logits.unsqueeze(0)  # (1, seq_len, vocab_size)
