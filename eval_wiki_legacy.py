"""
WikiText2 PPL evaluation using legacy (LittleBitITQSpec) checkpoints.

This script loads checkpoints saved in the old format (LittleBitITQSpecLinear with
resume_eff_bit / Matryoshka style) and measures:
  1. Draft-only PPL  (primary path only)
  2. Target PPL      (primary + residual path)

Usage:
    python eval_wiki_legacy.py \
        --model_path /path/to/old/ckpt \
        --model_type llama \
        --eff_bit 1.0 \
        --resume_eff_bit 0.1 \
        --quant_func SmoothSign
"""

import argparse
import gc
import json
import os
import re
import sys
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# --- Import from current project ---
from quantization.utils.binary_packer import binary_unpacker
from quantization.functions import STEBinary, SmoothSign
from utils.datautils import get_eval_loaders


# ==============================================================================
# LittleBitITQSpecLinear (from littlespec_ref/littlebit_itq_spec.py)
# Inlined here so we don't need the 'modeling' module or modify the ref code.
# ==============================================================================

class LittleBitITQSpecLinear(nn.Module):
    """Legacy quantization module supporting Matryoshka (resume_eff_bit) style.

    Handles:
    - Primary path (U, V, u1, u2, v1, v2)
    - Residual path (U_R, V_R, u1_R, u2_R, v1_R, v2_R) when resume_eff_bit > 0
    """

    def __quant_convert__(
        self,
        do_train: bool,
        quant_func,
        *,
        is_po2: bool = False,
        split_dim: int = 1024,
        eff_bit: Optional[float] = None,
        residual: bool = False,
        ratio_factor: float = 1.0,
        min_split_dim: int = 8,
        defer_init: bool = False,
        resume_eff_bit: Optional[float] = None,
        resume_eff_bit_2: Optional[float] = None,
        **kwargs,
    ):
        self.do_train = do_train
        self.quant_func = quant_func
        self.is_po2 = is_po2
        self.residual = residual
        self.defer_init = defer_init
        self.resume_eff_bit = resume_eff_bit
        self.resume_eff_bit_2 = kwargs.get("resume_eff_bit_2", resume_eff_bit_2)
        self._binarized = False
        a, b = self.in_features, self.out_features
        eff_bit_target = eff_bit
        self.ratio_factor = ratio_factor
        is_stage3 = (self.resume_eff_bit_2 is not None and self.resume_eff_bit_2 > 0.0)
        is_matryo = (self.resume_eff_bit is not None and self.resume_eff_bit > 0.0)

        if is_stage3:
            self.residual = True
            p_split = self._estimate_split_dim(a, b, self.resume_eff_bit, False)
            if p_split: p_split *= ratio_factor
            self.split_dim = self._finalize_split_dim(p_split, split_dim, min_split_dim)

            res_eff_bit = (self.resume_eff_bit_2 - self.resume_eff_bit)
            if res_eff_bit < 0: res_eff_bit = 0.0
            r_split = self._estimate_split_dim(a, b, res_eff_bit, False)
            if r_split: r_split *= ratio_factor
            self.split_dim_R = self._finalize_split_dim(r_split, split_dim, min_split_dim)

            res_eff_bit_2 = (eff_bit_target - self.resume_eff_bit_2) if eff_bit_target is not None else 0.0
            if res_eff_bit_2 < 0: res_eff_bit_2 = 0.0
            r2_split = self._estimate_split_dim(a, b, res_eff_bit_2, False)
            if r2_split: r2_split *= ratio_factor
            self.split_dim_R2 = self._finalize_split_dim(r2_split, split_dim, min_split_dim)

            eff_bit_actual = (self._compute_eff_bits(a, b, self.split_dim, False) +
                              self._compute_eff_bits(a, b, self.split_dim_R, False) +
                              self._compute_eff_bits(a, b, self.split_dim_R2, False))
            self.register_buffer("_split_dim_R2_final", torch.tensor(self.split_dim_R2))

        elif is_matryo:
            self.residual = True
            p_split = self._estimate_split_dim(a, b, self.resume_eff_bit, False)
            if p_split: p_split *= ratio_factor
            self.split_dim = self._finalize_split_dim(p_split, split_dim, min_split_dim)

            res_eff_bit = (eff_bit_target - self.resume_eff_bit) if eff_bit_target is not None else 0.0
            if res_eff_bit < 0: res_eff_bit = 0.0

            if ratio_factor > 1.0:
                r_split = p_split
            else:
                r_split = self._estimate_split_dim(a, b, res_eff_bit, False)
                if r_split: r_split *= ratio_factor
            self.split_dim_R = self._finalize_split_dim(r_split, split_dim, min_split_dim)

            eff_bit_actual = (self._compute_eff_bits(a, b, self.split_dim, False) +
                              self._compute_eff_bits(a, b, self.split_dim_R, False))
        else:
            split_calc_float = self._estimate_split_dim(a, b, eff_bit_target, residual)
            if split_calc_float: split_calc_float *= ratio_factor
            self.split_dim = self._finalize_split_dim(split_calc_float, split_dim, min_split_dim)
            self.split_dim_R = self.split_dim
            eff_bit_actual = self._compute_eff_bits(a, b, self.split_dim, residual)

        self.register_buffer("_eff_bit_target", torch.tensor(-1.0 if eff_bit_target is None else float(eff_bit_target)))
        self.register_buffer("_split_dim_final", torch.tensor(self.split_dim))
        self.register_buffer("_split_dim_R_final", torch.tensor(self.split_dim_R))
        self.register_buffer("_eff_bit_actual", torch.tensor(eff_bit_actual))

        # Always initialize empty for inference loading
        self._initialize_empty_parameters()

    @staticmethod
    def _estimate_split_dim(a, b, eff_bit_target, residual) -> Optional[float]:
        if eff_bit_target is None or a * b == 0:
            return None
        base = a + b + 16
        if residual:
            numerator = a * b * eff_bit_target - 32 * (a + b)
            denominator = 2 * base
        else:
            numerator = a * b * eff_bit_target - 16 * (a + b)
            denominator = base
        return numerator / denominator if denominator else None

    @staticmethod
    def _finalize_split_dim(split_float, split_default, min_split_dim):
        cand = split_float if split_float is not None else split_default
        cand = int(cand) if cand is not None else 0
        cand = (cand // 8) * 8
        if cand == 0:
            cand = min_split_dim
        return max(cand, min_split_dim)

    @staticmethod
    def _compute_eff_bits(a, b, s, residual):
        if a * b == 0:
            return float("inf")
        if residual:
            num = s * 2 * (a + b + 16) + 32 * (a + b)
        else:
            num = s * (a + b + 16) + 16 * (a + b)
        return num / (a * b)

    def forward(self, x):
        *seqlen, hidden_dim = x.shape
        seqlen.append(self.out_features)
        hidden_output_dim = tuple(seqlen)
        x = x.view(-1, hidden_dim)

        y = self._compute_forward(x, self.V, self.U, self.v2, self.v1, self.u2, self.u1)

        if self.residual:
            if self.ratio_factor > 1.0:
                res = self._compute_forward(x, self.V, self.U_R, self.v2, self.v1, self.u2_R, self.u1_R)
            else:
                res = self._compute_forward(x, self.V_R, self.U_R, self.v2_R, self.v1_R, self.u2_R, self.u1_R)
            y = y + res

        if getattr(self, "resume_eff_bit_2", 0.0) > 0.0:
            res2 = self._compute_forward(x, self.V_R2, self.U_R2, self.v2_R2, self.v1_R2, self.u2_R2, self.u1_R2)
            y = y + res2

        if self.bias is not None:
            y += self.bias
        y = y.reshape(hidden_output_dim)
        return y

    def forward_draft_only(self, x):
        """Forward using only the primary (draft) path, ignoring residual."""
        *seqlen, hidden_dim = x.shape
        seqlen.append(self.out_features)
        hidden_output_dim = tuple(seqlen)
        x = x.view(-1, hidden_dim)

        y = self._compute_forward(x, self.V, self.U, self.v2, self.v1, self.u2, self.u1)

        if self.bias is not None:
            y += self.bias
        y = y.reshape(hidden_output_dim)
        return y

    def _compute_forward(self, x, V, U, v2, v1, u2, u1):
        Vq = self.quantize(V.to(x.dtype))
        Uq = self.quantize(U.to(x.dtype))
        v1u2 = v1 * u2
        return ((((x * v2) @ Vq.t()) * v1u2) @ Uq.t()) * u1

    def quantize(self, x):
        if self._binarized:
            return x
        return self.quant_func(x)

    def _initialize_empty_parameters(self):
        dtype = torch.bfloat16
        device = "meta"

        def create_param(*shape):
            return nn.Parameter(torch.empty(*shape, device=device, dtype=dtype), requires_grad=False)

        if self.defer_init:
            self.register_buffer("U", torch.empty(self.out_features, self.split_dim, device=device, dtype=dtype))
            self.register_buffer("V", torch.empty(self.split_dim, self.in_features, device=device, dtype=dtype))
            self.register_buffer("u1", torch.empty(1, self.out_features, device=device, dtype=dtype))
            self.register_buffer("u2", torch.empty(1, self.split_dim, device=device, dtype=dtype))
            self.register_buffer("v1", torch.empty(1, self.split_dim, device=device, dtype=dtype))
            self.register_buffer("v2", torch.empty(1, self.in_features, device=device, dtype=dtype))
        else:
            self.U = create_param(self.out_features, self.split_dim)
            self.V = create_param(self.split_dim, self.in_features)
            self.u1 = create_param(1, self.out_features)
            self.u2 = create_param(1, self.split_dim)
            self.v1 = create_param(1, self.split_dim)
            self.v2 = create_param(1, self.in_features)

        if self.residual:
            self.U_R = create_param(self.out_features, self.split_dim_R)
            self.V_R = create_param(self.split_dim_R, self.in_features)
            self.u1_R = create_param(1, self.out_features)
            self.u2_R = create_param(1, self.split_dim_R)
            self.v1_R = create_param(1, self.split_dim_R)
            self.v2_R = create_param(1, self.in_features)

        if getattr(self, "resume_eff_bit_2", 0.0) > 0.0:
            if self.defer_init:
                self.register_buffer("U_R2", torch.empty(self.out_features, self.split_dim_R2, device=device, dtype=dtype))
                self.register_buffer("V_R2", torch.empty(self.split_dim_R2, self.in_features, device=device, dtype=dtype))
                self.register_buffer("u1_R2", torch.empty(1, self.out_features, device=device, dtype=dtype))
                self.register_buffer("u2_R2", torch.empty(1, self.split_dim_R2, device=device, dtype=dtype))
                self.register_buffer("v1_R2", torch.empty(1, self.split_dim_R2, device=device, dtype=dtype))
                self.register_buffer("v2_R2", torch.empty(1, self.in_features, device=device, dtype=dtype))
            else:
                self.U_R2 = create_param(self.out_features, self.split_dim_R2)
                self.V_R2 = create_param(self.split_dim_R2, self.in_features)
                self.u1_R2 = create_param(1, self.out_features)
                self.u2_R2 = create_param(1, self.split_dim_R2)
                self.v1_R2 = create_param(1, self.split_dim_R2)
                self.v2_R2 = create_param(1, self.in_features)

        if hasattr(self, 'weight'):
            del self.weight
        self.register_parameter('weight', None)


# ==============================================================================
# Model loading (based on littlespec_ref/quant.uils.py::load_quantized_model)
# Uses AutoModelForCausalLM.from_config + patching with LittleBitITQSpecLinear
# ==============================================================================

QUANT_FUNC_MAP = {
    "STEBinary": STEBinary,
    "SmoothSign": SmoothSign,
}


def _load_and_process_state_dict(model_path: str, torch_dtype: torch.dtype):
    """Load state dict from safetensors, handling sharded and packed formats."""
    state_dict = {}

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = os.path.join(model_path, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        for shard_file in set(index["weight_map"].values()):
            with safe_open(os.path.join(model_path, shard_file), framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
    elif os.path.exists(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}")

    has_packed_weights = any(key.endswith("_packed") for key in state_dict.keys())
    if not has_packed_weights:
        print("INFO: Unpacked (legacy) format detected. Loading weights as is.")
        return state_dict, False

    print("INFO: Packed format detected. Unpacking binary weights...")
    packed_components = defaultdict(dict)
    final_state_dict = {}
    pattern = re.compile(r"^(.*)\.([^.]+?)_(packed|shape)$")

    for key, value in state_dict.items():
        match = pattern.match(key)
        if match:
            prefix, param_name, suffix_type = match.groups()
            packed_components[prefix][f"{param_name}_{suffix_type}"] = value
        else:
            final_state_dict[key] = value

    for prefix, components in packed_components.items():
        param_names = {
            key.replace("_packed", "")
            for key in components.keys()
            if key.endswith("_packed")
        }

        for name in param_names:
            packed_key = f"{name}_packed"
            shape_key = f"{name}_shape"

            if packed_key in components and shape_key in components:
                shape = tuple(components[shape_key].tolist())
                unpacked_tensor = binary_unpacker(components[packed_key], shape).to(torch_dtype)
                final_state_dict[f"{prefix}.{name}"] = unpacked_tensor

    return final_state_dict, True


def _patch_model_with_itqspec(model, args):
    """Patch nn.Linear modules with LittleBitITQSpecLinear."""
    quant_func_name = getattr(args, "quant_func", "SmoothSign")
    quant_func = QUANT_FUNC_MAP.get(quant_func_name)
    if quant_func is None:
        raise ValueError(f"Unknown quant_func: {quant_func_name}. Available: {list(QUANT_FUNC_MAP.keys())}")

    common_kwargs = {
        "do_train": False,
        "quant_func": quant_func,
        "residual": getattr(args, "residual", False),
        "split_dim": getattr(args, "split_dim", 1024),
        "eff_bit": getattr(args, "eff_bit", 1.0),
        "min_split_dim": getattr(args, "min_split_dim", 8),
        "resume_eff_bit": getattr(args, "resume_eff_bit", None),
        "resume_eff_bit_2": getattr(args, "resume_eff_bit_2", None),
    }

    KV_PATTERN = [re.compile(r'\.k_proj$'), re.compile(r'\.v_proj$')]
    kv_kwargs = {
        "ratio_factor": getattr(args, "kv_factor", 1.0),
    }

    mapping = {nn.Linear: LittleBitITQSpecLinear}

    for name, mod in model.named_modules():
        if "lm_head" in name:
            continue

        # Determine kwargs
        current_kwargs = dict(common_kwargs)
        for pat in KV_PATTERN:
            if pat.search(name):
                current_kwargs.update(kv_kwargs)
                break

        if type(mod) in mapping:
            mod.__class__ = mapping[type(mod)]
            if hasattr(mod, '__quant_convert__'):
                mod.__quant_convert__(**current_kwargs)

    return model


def load_legacy_model(model_path, args, torch_dtype=torch.bfloat16, device="auto"):
    """Load a legacy LittleBitITQSpec checkpoint.

    Args:
        model_path: Path to the checkpoint directory.
        args: Namespace with quant params (eff_bit, resume_eff_bit, quant_func, etc.)
        torch_dtype: Data type for model parameters.
        device: Target device.

    Returns:
        (model, tokenizer)
    """
    print(f"INFO: Loading legacy model from '{model_path}'")

    # 1. Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 2. Create skeleton model on meta device
    print(f"INFO: Creating {config.model_type} skeleton on meta device...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.to_empty(device="cpu")

    for param in model.parameters():
        param.data.zero_()

    # 3. Patch with LittleBitITQSpecLinear
    print("INFO: Patching with LittleBitITQSpecLinear...")
    model = _patch_model_with_itqspec(model, args)

    # 4. Load state dict
    print("INFO: Loading state dictionary...")
    state_dict, was_packed = _load_and_process_state_dict(model_path, torch_dtype)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing:
        # Filter out common expected missing keys
        real_missing = [k for k in missing if not k.startswith("model.layers.") or
                        not any(x in k for x in ["_split_dim", "_eff_bit"])]
        if real_missing:
            print(f"WARNING: Missing keys ({len(real_missing)} shown): {real_missing[:10]}...")
    if unexpected:
        print(f"WARNING: Unexpected keys ({len(unexpected)} shown): {unexpected[:10]}...")

    # 5. Tie weights
    try:
        model.tie_weights()
    except Exception as e:
        print(f"WARN: tie_weights() failed: {e}")

    # Handle lm_head on meta
    if hasattr(model, "lm_head") and getattr(getattr(model.lm_head, "weight", None), "is_meta", False):
        print("WARN: lm_head.weight is still on meta. Manually resolving...")
        if hasattr(model, "get_input_embeddings"):
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight") and not getattr(emb.weight, "is_meta", False):
                model.lm_head.weight = emb.weight

    del state_dict
    gc.collect()

    # 6. Post-process
    if was_packed:
        print("INFO: Setting modules to binarized inference mode.")
        for module in model.modules():
            if isinstance(module, LittleBitITQSpecLinear):
                module._binarized = True
    else:
        print(f"INFO: Legacy format. Casting params to {torch_dtype}...")
        for param in model.parameters():
            if param.dtype != torch_dtype:
                param.data = param.data.to(torch_dtype)

    if hasattr(model, 'lm_head') and model.lm_head is not None:
        model.lm_head.to(torch_dtype)

    # Materialize any remaining meta tensors
    for name, module in model.named_modules():
        for param_name, param in list(module.named_parameters(recurse=False)):
            if param.device.type == 'meta':
                new_param = nn.Parameter(
                    torch.zeros_like(param, device="cpu", dtype=torch_dtype),
                    requires_grad=False
                )
                setattr(module, param_name, new_param)

    # 7. Move to device
    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    print(f"INFO: Moving model to {target_device}...")
    model.to(target_device)
    model.eval()

    print(f"Model ready on {target_device}")
    return model


# ==============================================================================
# PPL evaluation
# ==============================================================================

@torch.no_grad()
def eval_ppl(model, tokenizer, datasets_str="wikitext2", seqlen=2048, mode="target"):
    """Evaluate PPL on the given model.

    Args:
        model: The loaded model.
        tokenizer: Tokenizer.
        datasets_str: Comma-separated dataset names.
        seqlen: Sequence length for chunks.
        mode: "target" (use full forward) or "draft" (use only primary path).
    """
    device = next(model.parameters()).device
    model.eval()

    # If draft mode, monkey-patch all ITQSpec modules to use draft-only forward
    if mode == "draft":
        print("[INFO] Draft mode: patching forward to use primary path only")
        for module in model.modules():
            if isinstance(module, LittleBitITQSpecLinear) and module.residual:
                module._original_forward = module.forward
                module.forward = module.forward_draft_only

    results = {}

    for dataset in datasets_str.split(","):
        dataset = dataset.strip()
        if not dataset:
            continue

        print(f"[INFO] Starting PPL eval for: {dataset} (mode={mode})")
        testloader = get_eval_loaders(dataset, tokenizer)
        testenc = testloader.input_ids
        nsamples = testenc.numel() // seqlen

        if nsamples == 0:
            print(f"Not enough data for {dataset} with seqlen {seqlen}. Skipping.")
            continue

        nlls = []
        for i in tqdm(range(nsamples), desc=f"PPL({dataset}/{mode})"):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(device)
            outputs = model(batch, use_cache=False)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * (seqlen - 1)
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))
        print(f"[{dataset}] {mode.upper()} PPL = {ppl.item():.4f}")
        results[dataset] = ppl.item()

    # Restore forward if we patched it
    if mode == "draft":
        for module in model.modules():
            if isinstance(module, LittleBitITQSpecLinear) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                del module._original_forward

    return results


def main():
    parser = argparse.ArgumentParser(description="Legacy checkpoint WikiText2 PPL evaluation")

    # Model path & type
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the legacy checkpoint directory")
    parser.add_argument("--model_type", type=str, default="llama",
                        help="Model architecture type (llama, gemma2, etc.)")
    parser.add_argument("--base_model_id", type=str, default=None,
                        help="Base model ID for tokenizer (if different from checkpoint)")

    # Quantization parameters (must match how the checkpoint was trained)
    parser.add_argument("--quant_func", type=str, default="SmoothSign",
                        help="Quantization function used during training")
    parser.add_argument("--eff_bit", type=float, default=1.0,
                        help="Total effective bits (draft + residual)")
    parser.add_argument("--resume_eff_bit", type=float, default=0.1,
                        help="Draft effective bits (primary path)")
    parser.add_argument("--resume_eff_bit_2", type=float, default=None,
                        help="Stage 3 effective bits (if applicable)")
    parser.add_argument("--split_dim", type=int, default=1024,
                        help="Default split dimension")
    parser.add_argument("--min_split_dim", type=int, default=8,
                        help="Minimum split dimension")
    parser.add_argument("--kv_factor", type=float, default=1.0,
                        help="KV projection ratio factor")

    # Evaluation settings
    parser.add_argument("--ppl_task", type=str, default="wikitext2",
                        help="Dataset for PPL evaluation (comma-separated)")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length for PPL evaluation")
    parser.add_argument("--eval_draft", action="store_true", default=True,
                        help="Evaluate draft-only PPL")
    parser.add_argument("--eval_target", action="store_true", default=True,
                        help="Evaluate target (draft+residual) PPL")
    parser.add_argument("--no_draft", action="store_true", default=False,
                        help="Skip draft PPL evaluation")
    parser.add_argument("--no_target", action="store_true", default=False,
                        help="Skip target PPL evaluation")

    args = parser.parse_args()

    # Build quant args
    quant_args = argparse.Namespace(
        quant_func=args.quant_func,
        eff_bit=args.eff_bit,
        resume_eff_bit=args.resume_eff_bit,
        resume_eff_bit_2=args.resume_eff_bit_2,
        split_dim=args.split_dim,
        min_split_dim=args.min_split_dim,
        kv_factor=args.kv_factor,
        residual=False,  # Matryoshka mode sets this automatically
    )

    # Load tokenizer
    tokenizer_id = args.base_model_id or args.model_path
    print(f"[INFO] Loading tokenizer from: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=False, trust_remote_code=True)

    # Load model
    model = load_legacy_model(
        model_path=args.model_path,
        args=quant_args,
        torch_dtype=torch.bfloat16,
        device="auto",
    )

    print("\n" + "=" * 60)
    print("Legacy Checkpoint PPL Evaluation")
    print(f"  Checkpoint:       {args.model_path}")
    print(f"  quant_func:       {args.quant_func}")
    print(f"  eff_bit (total):  {args.eff_bit}")
    print(f"  resume_eff_bit:   {args.resume_eff_bit}")
    print(f"  Dataset:          {args.ppl_task}")
    print(f"  Seqlen:           {args.seqlen}")
    print("=" * 60)

    all_results = {}

    # Evaluate target PPL (primary + residual)
    if not args.no_target:
        print("\n>>> Evaluating TARGET PPL (draft + residual)...")
        target_results = eval_ppl(model, tokenizer, args.ppl_task, args.seqlen, mode="target")
        all_results["target"] = target_results

    # Evaluate draft-only PPL (primary path only)
    if not args.no_draft:
        print("\n>>> Evaluating DRAFT PPL (primary path only)...")
        draft_results = eval_ppl(model, tokenizer, args.ppl_task, args.seqlen, mode="draft")
        all_results["draft"] = draft_results

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for mode, results in all_results.items():
        for dataset, ppl in results.items():
            print(f"  [{mode.upper():>6}] {dataset}: PPL = {ppl:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
