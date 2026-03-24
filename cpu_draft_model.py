"""
CPU Kernel Draft Model for Speculative Decoding.

Wraps lb_kernels/littlebit_kernels_cpu's DummyLlama3LittleBitModel
to provide the same interface as MatryoshkaDraftModel.

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


class CPUDraftModel:
    """CPU kernel-based draft model for speculative decoding.
    
    Loads a converted runtime checkpoint and runs inference
    using the optimized CPU LittleBit kernel.
    
    The model handles:
    - Embedding lookup (dense, from base model)
    - Transformer layers (LittleBit CPU kernel)
    - LM head projection (dense, from base model)
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
        self._cache = None
        self._cache_position = 0
        
        logger.info("CPU draft model ready")
    
    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings."""
        return F.embedding(input_ids.to("cpu"), self.embed_tokens)
    
    def _lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden state to logits."""
        return F.linear(hidden.to(torch.float32), self.lm_head_weight)
    
    @torch.no_grad()
    def forward_single_token(self, hidden: torch.Tensor, position: int):
        """Forward pass for a single token through the transformer + lm_head."""
        if self._cache is None:
            self._cache = self.lb_model.allocate_cache(self.max_seq_len)
        
        output = self.lb_model.forward_token(
            hidden, self._cache, position, compute_logits=False
        )
        logits = self._lm_head(output.hidden)
        return logits
    
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
        
        Note: This processes the ENTIRE prefix first (no KV cache reuse
        across calls), then generates K draft tokens.
        """
        input_ids = input_ids.to("cpu")
        seq_len = input_ids.shape[1]
        
        # Reset cache
        self._cache = self.lb_model.allocate_cache(self.max_seq_len)
        
        # Process prefix: run each token through the model
        for pos in range(seq_len):
            token_id = input_ids[0, pos].item()
            hidden = self._embed(torch.tensor([[token_id]]))  # (1, hidden_size)
            hidden = hidden.squeeze(0)  # (1, hidden_size)
            
            output = self.lb_model.forward_token(
                hidden, self._cache, pos, compute_logits=False
            )
        
        # Now generate K draft tokens
        draft_tokens = []
        draft_probs = []
        last_hidden = output.hidden  # (1, hidden_size)
        
        for k in range(draft_length):
            # Get logits from lm_head
            logits = self._lm_head(last_hidden)  # (1, vocab_size)
            
            if greedy:
                probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (1, 1)
            else:
                probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            draft_tokens.append(next_token)
            draft_probs.append(probs.unsqueeze(0))  # (1, 1, vocab) -> match expected shape
            
            # Forward next token through transformer
            hidden = self._embed(next_token)  # (1, 1, hidden_size)
            hidden = hidden.squeeze(0)  # (1, hidden_size)
            
            pos = seq_len + k
            output = self.lb_model.forward_token(
                hidden, self._cache, pos, compute_logits=False
            )
            last_hidden = output.hidden
        
        return draft_tokens, draft_probs
