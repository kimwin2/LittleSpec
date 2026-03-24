"""
Convert HF LittleBit checkpoint to CPU runtime format.

Our HF format:
    {prefix}.U_packed, {prefix}.U_shape  →  u_sign_rt, u_sign_rt_shape
    {prefix}.V_packed, {prefix}.V_shape  →  v_sign_rt, v_sign_rt_shape
    {prefix}.u1                          →  u1  (same)
    {prefix}.u2  ×  {prefix}.v1          →  mid (fused)
    {prefix}.v2                          →  v2  (same)

Usage:
    python convert_hf_to_runtime.py \
        --input_path outputs/step1_draft_0.1bit/<timestamp> \
        --output_path outputs/step1_draft_0.1bit/<timestamp>_runtime
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def convert_state_dict(hf_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert HF LittleBit state dict keys to CPU runtime format."""
    runtime_state = {}

    # Group keys by LittleBitLinear module prefix
    # Find all unique prefixes that have U_packed
    prefixes = set()
    for key in hf_state:
        if key.endswith(".U_packed"):
            prefixes.add(key[: -len(".U_packed")])
        elif key.endswith(".U_R_packed"):
            prefixes.add(key[: -len(".U_R_packed")])

    converted_keys = set()

    for prefix in sorted(prefixes):
        # === Main branch ===
        if f"{prefix}.U_packed" in hf_state:
            runtime_state[f"{prefix}.u_sign_rt"] = hf_state[f"{prefix}.U_packed"]
            runtime_state[f"{prefix}.u_sign_rt_shape"] = hf_state[f"{prefix}.U_shape"]
            runtime_state[f"{prefix}.v_sign_rt"] = hf_state[f"{prefix}.V_packed"]
            runtime_state[f"{prefix}.v_sign_rt_shape"] = hf_state[f"{prefix}.V_shape"]
            runtime_state[f"{prefix}.u1"] = hf_state[f"{prefix}.u1"]
            runtime_state[f"{prefix}.v2"] = hf_state[f"{prefix}.v2"]

            # Fuse: mid = v1 * u2
            v1 = hf_state[f"{prefix}.v1"].to(torch.float32)
            u2 = hf_state[f"{prefix}.u2"].to(torch.float32)
            runtime_state[f"{prefix}.mid"] = (v1 * u2).contiguous()

            converted_keys.update([
                f"{prefix}.U_packed", f"{prefix}.U_shape",
                f"{prefix}.V_packed", f"{prefix}.V_shape",
                f"{prefix}.u1", f"{prefix}.u2",
                f"{prefix}.v1", f"{prefix}.v2",
            ])

        # === Residual branch ===
        if f"{prefix}.U_R_packed" in hf_state:
            runtime_state[f"{prefix}.u_sign_r_rt"] = hf_state[f"{prefix}.U_R_packed"]
            runtime_state[f"{prefix}.u_sign_r_rt_shape"] = hf_state[f"{prefix}.U_R_shape"]
            runtime_state[f"{prefix}.v_sign_r_rt"] = hf_state[f"{prefix}.V_R_packed"]
            runtime_state[f"{prefix}.v_sign_r_rt_shape"] = hf_state[f"{prefix}.V_R_shape"]
            runtime_state[f"{prefix}.u1_r"] = hf_state[f"{prefix}.u1_R"]
            runtime_state[f"{prefix}.v2_r"] = hf_state[f"{prefix}.v2_R"]

            v1_r = hf_state[f"{prefix}.v1_R"].to(torch.float32)
            u2_r = hf_state[f"{prefix}.u2_R"].to(torch.float32)
            runtime_state[f"{prefix}.mid_r"] = (v1_r * u2_r).contiguous()

            converted_keys.update([
                f"{prefix}.U_R_packed", f"{prefix}.U_R_shape",
                f"{prefix}.V_R_packed", f"{prefix}.V_R_shape",
                f"{prefix}.u1_R", f"{prefix}.u2_R",
                f"{prefix}.v1_R", f"{prefix}.v2_R",
            ])

    # Pass through non-quantized tensors (layernorm, embed_tokens, buffers, etc.)
    for key, value in hf_state.items():
        if key not in converted_keys:
            # Skip internal LittleBit buffers
            if any(key.endswith(s) for s in [
                "._eff_bit_target", "._split_dim_final", "._eff_bit_actual",
                "._binarized",
            ]):
                continue
            runtime_state[key] = value

    return runtime_state


def load_hf_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Load state dict from HF-style safetensors checkpoint."""
    model_path = Path(model_path)
    state_dict = {}

    index_path = model_path / "model.safetensors.index.json"
    single_path = model_path / "model.safetensors"

    if index_path.exists():
        with open(index_path, "r") as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
        for shard_file in shard_files:
            shard_state = load_file(str(model_path / shard_file))
            state_dict.update(shard_state)
    elif single_path.exists():
        state_dict = load_file(str(single_path))
    else:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    return state_dict


def build_dummy_config(hf_config_path: str, littlebit_config_path: str) -> dict:
    """Build dummy_llama3_config.json from HF config.json and littlebit_config.json."""
    with open(hf_config_path, "r") as f:
        hf_config = json.load(f)
    with open(littlebit_config_path, "r") as f:
        lb_config = json.load(f)

    return {
        "name": hf_config.get("_name_or_path", "llama"),
        "hidden_size": hf_config["hidden_size"],
        "intermediate_size": hf_config["intermediate_size"],
        "num_hidden_layers": hf_config["num_hidden_layers"],
        "num_attention_heads": hf_config["num_attention_heads"],
        "num_key_value_heads": hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
        "vocab_size": hf_config["vocab_size"],
        "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
        "include_lm_head": False,
        # Extra info
        "eff_bit": lb_config.get("eff_bit", 0.1),
        "residual": lb_config.get("residual", False),
    }


def build_runtime_config(hf_config_path: str, littlebit_config_path: str) -> dict:
    """Build littlebit_runtime_config.json."""
    with open(hf_config_path, "r") as f:
        hf_config = json.load(f)
    with open(littlebit_config_path, "r") as f:
        lb_config = json.load(f)

    return {
        "format_version": 1,
        "checkpoint_format": "runtime",
        "sign_storage": "row_i1",
        "runtime_layout": "cpu_row_i1",
        "base_model_id": hf_config.get("_name_or_path", None),
        "model_type": hf_config.get("model_type", "llama"),
        "torch_dtype": "bfloat16",
        "quant_func": lb_config.get("quant_func", "STEBinary"),
        "eff_bit": lb_config.get("eff_bit", 0.1),
        "split_dim": lb_config.get("split_dim", None),
        "residual": lb_config.get("residual", False),
    }


def convert_hf_to_runtime(input_path: str, output_path: str):
    """Convert HF LittleBit checkpoint to CPU runtime format."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load HF state dict
    print(f"Loading HF checkpoint from {input_path}...")
    hf_state = load_hf_state_dict(str(input_path))
    print(f"  Loaded {len(hf_state)} tensors")

    # Convert
    print("Converting to runtime format...")
    runtime_state = convert_state_dict(hf_state)
    print(f"  Converted to {len(runtime_state)} tensors")

    # Count converted projections
    rt_keys = [k for k in runtime_state if k.endswith(".u_sign_rt")]
    print(f"  Converted {len(rt_keys)} LittleBit linear projections")

    # Save runtime safetensors
    runtime_safetensors_path = output_path / "littlebit_runtime.safetensors"
    print(f"Saving to {runtime_safetensors_path}...")
    save_file(runtime_state, str(runtime_safetensors_path))

    # Save configs
    hf_config_path = input_path / "config.json"
    lb_config_path = input_path / "littlebit_config.json"

    if hf_config_path.exists() and lb_config_path.exists():
        # Runtime config
        rt_config = build_runtime_config(str(hf_config_path), str(lb_config_path))
        rt_config_path = output_path / "littlebit_runtime_config.json"
        with open(rt_config_path, "w") as f:
            json.dump(rt_config, f, indent=2)
        print(f"Saved runtime config to {rt_config_path}")

        # Dummy model config
        dummy_config = build_dummy_config(str(hf_config_path), str(lb_config_path))
        dummy_config_path = output_path / "dummy_llama3_config.json"
        with open(dummy_config_path, "w") as f:
            json.dump(dummy_config, f, indent=2)
        print(f"Saved dummy config to {dummy_config_path}")
    else:
        print(f"WARNING: config.json or littlebit_config.json not found in {input_path}")

    del hf_state, runtime_state
    print("Conversion complete!")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert HF LittleBit checkpoint to CPU runtime format")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to HF LittleBit checkpoint directory")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for runtime checkpoint")
    args = parser.parse_args()

    convert_hf_to_runtime(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
