"""
Test: Verify HF-to-runtime conversion produces valid tensors.
Run on GPU server after converting a checkpoint.

Usage:
    python test_cpu_draft.py \
        --hf_path outputs/step1_draft_0.1bit/<timestamp> \
        --runtime_path outputs/step1_draft_0.1bit/<timestamp>_runtime
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

# Add lb_kernels to path
_LB_KERNELS_ROOT = Path(__file__).resolve().parent / "lb_kernels"
if str(_LB_KERNELS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LB_KERNELS_ROOT))


def test_conversion_keys(hf_path: str, runtime_path: str):
    """Verify all HF quantized layers were converted to runtime format."""
    hf_path = Path(hf_path)
    runtime_path = Path(runtime_path)

    # Load HF state dict
    index_path = hf_path / "model.safetensors.index.json"
    single_path = hf_path / "model.safetensors"
    
    hf_state = {}
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        for shard in set(index["weight_map"].values()):
            hf_state.update(load_file(str(hf_path / shard)))
    elif single_path.exists():
        hf_state = load_file(str(single_path))

    # Load runtime state dict
    rt_state = load_file(str(runtime_path / "littlebit_runtime.safetensors"))

    # Count HF quantized projections
    hf_prefixes = set()
    for key in hf_state:
        if key.endswith(".U_packed"):
            hf_prefixes.add(key[:-len(".U_packed")])

    # Count runtime projections
    rt_prefixes = set()
    for key in rt_state:
        if key.endswith(".u_sign_rt"):
            rt_prefixes.add(key[:-len(".u_sign_rt")])

    print(f"HF quantized projections: {len(hf_prefixes)}")
    print(f"Runtime projections:      {len(rt_prefixes)}")

    assert len(hf_prefixes) == len(rt_prefixes), \
        f"Mismatch: HF has {len(hf_prefixes)}, runtime has {len(rt_prefixes)}"

    # Verify each projection has all required keys
    for prefix in sorted(rt_prefixes):
        required = [
            f"{prefix}.u_sign_rt", f"{prefix}.u_sign_rt_shape",
            f"{prefix}.v_sign_rt", f"{prefix}.v_sign_rt_shape",
            f"{prefix}.u1", f"{prefix}.mid", f"{prefix}.v2",
        ]
        for key in required:
            assert key in rt_state, f"Missing key: {key}"

    print("✓ All projection keys present")

    # Verify mid = v1 * u2
    for prefix in sorted(list(rt_prefixes)[:3]):  # test first 3
        v1 = hf_state[f"{prefix}.v1"].to(torch.float32)
        u2 = hf_state[f"{prefix}.u2"].to(torch.float32)
        expected_mid = v1 * u2
        actual_mid = rt_state[f"{prefix}.mid"].to(torch.float32)
        assert torch.allclose(expected_mid, actual_mid, atol=1e-6), \
            f"mid mismatch at {prefix}"

    print("✓ mid = v1 * u2 verified")

    # Verify packed weights are identical
    for prefix in sorted(list(rt_prefixes)[:3]):
        assert torch.equal(
            hf_state[f"{prefix}.U_packed"],
            rt_state[f"{prefix}.u_sign_rt"]
        ), f"U_packed != u_sign_rt at {prefix}"
        assert torch.equal(
            hf_state[f"{prefix}.V_packed"],
            rt_state[f"{prefix}.v_sign_rt"]
        ), f"V_packed != v_sign_rt at {prefix}"

    print("✓ Packed weights identical")

    # Verify configs exist
    assert (runtime_path / "littlebit_runtime_config.json").exists(), "Missing runtime config"
    assert (runtime_path / "dummy_llama3_config.json").exists(), "Missing dummy config"
    print("✓ Config files present")

    print("\n✅ All conversion tests passed!")


def test_runtime_linear(runtime_path: str):
    """Test that runtime linear layers can actually execute."""
    runtime_path = Path(runtime_path)
    rt_state = load_file(str(runtime_path / "littlebit_runtime.safetensors"))

    from littlebit_kernels_cpu.runtime import (
        load_runtime_linear,
        littlebit_linear_reference,
    )

    # Find first projection
    prefix = None
    for key in rt_state:
        if key.endswith(".u_sign_rt"):
            prefix = key[:-len(".u_sign_rt")]
            break

    if prefix is None:
        print("No runtime projections found, skipping linear test")
        return

    print(f"\nTesting runtime linear at {prefix}...")
    runtime = load_runtime_linear(rt_state, prefix, device="cpu")

    # Create dummy input
    in_features = int(runtime.main.v_shape[1])
    x = torch.randn(1, in_features, dtype=torch.float32)

    # Run reference forward
    out = littlebit_linear_reference(x, runtime)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
    print("✓ Runtime linear execution successful")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", type=str, required=True)
    parser.add_argument("--runtime_path", type=str, required=True)
    args = parser.parse_args()

    test_conversion_keys(args.hf_path, args.runtime_path)
    test_runtime_linear(args.runtime_path)


if __name__ == "__main__":
    main()
