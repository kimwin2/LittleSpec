"""
Benchmark: Draft vs Target model speed (CPU LittleBit kernel).

Measures autoregressive token generation speed (TPS) for:
  1. Draft model (0.3-bit) — CPUDraftModel
  2. Target model (0.3+1.7-bit combined) — CPUTargetModel

Usage:
    python benchmark_models_cpu.py \
        --base_model_id /path/to/Llama-3.1-8B-Instruct \
        --draft_runtime_path /path/to/draft_runtime \
        --residual_runtime_path /path/to/residual_runtime \
        --num_tokens 64 \
        --warmup_tokens 5
"""

import argparse
import time

import torch
import torch.nn.functional as F

from cpu_draft_model import CPUDraftModel
from cpu_target_model import CPUTargetModel
from utils.datautils import load_tokenizer
from utils.misc import setup_logger

logger = setup_logger(__name__)


def benchmark_draft(draft_model, input_ids, num_tokens, warmup_tokens=5):
    """Benchmark draft model: autoregressive generation one token at a time."""
    draft_model.reset()

    # Prefill prompt
    t0 = time.perf_counter()
    draft_model.prefill(input_ids)
    prefill_time = time.perf_counter() - t0
    prefill_len = input_ids.shape[1]
    logger.info(f"  Prefill: {prefill_len} tokens in {prefill_time:.3f}s "
                f"({prefill_len / prefill_time:.1f} tok/s)")

    # Generate tokens one by one
    generated = []
    last_hidden = draft_model._last_hidden
    token_times = []

    for i in range(num_tokens):
        t0 = time.perf_counter()
        logits = draft_model._lm_head(last_hidden)
        next_token = torch.argmax(logits, dim=-1).item()
        last_hidden = draft_model._forward_token(next_token)
        elapsed = time.perf_counter() - t0

        generated.append(next_token)
        if i >= warmup_tokens:
            token_times.append(elapsed)

    # Stats (excluding warmup)
    if token_times:
        avg_ms = sum(token_times) / len(token_times) * 1000
        tps = len(token_times) / sum(token_times)
    else:
        avg_ms = 0
        tps = 0

    return {
        "model": "draft",
        "num_tokens": num_tokens,
        "warmup_tokens": warmup_tokens,
        "measured_tokens": len(token_times),
        "avg_ms_per_token": avg_ms,
        "tokens_per_second": tps,
        "prefill_time_s": prefill_time,
        "prefill_tokens": prefill_len,
        "total_time_s": sum(token_times),
    }


def benchmark_target(target_model, input_ids, num_tokens, warmup_tokens=5):
    """Benchmark target model: autoregressive generation using forward()."""
    target_model.reset_cache()

    # Prefill prompt
    t0 = time.perf_counter()
    logits = target_model.forward(input_ids)
    prefill_time = time.perf_counter() - t0
    prefill_len = input_ids.shape[1]
    logger.info(f"  Prefill: {prefill_len} tokens in {prefill_time:.3f}s "
                f"({prefill_len / prefill_time:.1f} tok/s)")

    # Generate tokens one by one
    current_ids = input_ids
    generated = []
    token_times = []

    # First token from prefill logits
    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
    generated.append(next_token)

    for i in range(1, num_tokens):
        current_ids = torch.cat([
            current_ids,
            torch.tensor([[next_token]], dtype=torch.long)
        ], dim=1)

        t0 = time.perf_counter()
        logits = target_model.forward(current_ids)
        elapsed = time.perf_counter() - t0

        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        generated.append(next_token)

        if i >= warmup_tokens:
            token_times.append(elapsed)

    # Stats (excluding warmup)
    if token_times:
        avg_ms = sum(token_times) / len(token_times) * 1000
        tps = len(token_times) / sum(token_times)
    else:
        avg_ms = 0
        tps = 0

    return {
        "model": "target",
        "num_tokens": num_tokens,
        "warmup_tokens": warmup_tokens,
        "measured_tokens": len(token_times),
        "avg_ms_per_token": avg_ms,
        "tokens_per_second": tps,
        "prefill_time_s": prefill_time,
        "prefill_tokens": prefill_len,
        "total_time_s": sum(token_times),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark draft vs target model speed on CPU"
    )
    parser.add_argument("--base_model_id", type=str, required=True)
    parser.add_argument("--draft_runtime_path", type=str, required=True)
    parser.add_argument("--residual_runtime_path", type=str, required=True)
    parser.add_argument("--num_tokens", type=int, default=64,
                        help="Number of tokens to generate")
    parser.add_argument("--warmup_tokens", type=int, default=5,
                        help="Warmup tokens to exclude from timing")
    parser.add_argument("--prompt", type=str,
                        default="Explain the concept of quantum entanglement in simple terms.")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = load_tokenizer(args.base_model_id)
    chat = [{"role": "user", "content": args.prompt}]
    try:
        prompt_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_text = args.prompt
    input_ids = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=2048
    )["input_ids"]

    print(f"\n{'='*70}")
    print(f"CPU LittleBit Model Speed Benchmark")
    print(f"  Prompt: {len(input_ids[0])} tokens")
    print(f"  Generate: {args.num_tokens} tokens (warmup: {args.warmup_tokens})")
    print(f"{'='*70}\n")

    # ── Draft Model ──────────────────────────────────────────────────────
    print("Loading draft model...")
    draft_model = CPUDraftModel(
        runtime_path=args.draft_runtime_path,
        base_model_id=args.base_model_id,
    )

    print("\nBenchmarking DRAFT model (0.3-bit)...")
    draft_stats = benchmark_draft(
        draft_model, input_ids,
        num_tokens=args.num_tokens,
        warmup_tokens=args.warmup_tokens,
    )

    # Free memory
    del draft_model
    import gc; gc.collect()

    # ── Target Model ─────────────────────────────────────────────────────
    print("\nLoading target model...")
    target_model = CPUTargetModel(
        draft_runtime_path=args.draft_runtime_path,
        residual_runtime_path=args.residual_runtime_path,
        base_model_id=args.base_model_id,
    )

    print("\nBenchmarking TARGET model (0.3+1.7 = 2.0-bit)...")
    target_stats = benchmark_target(
        target_model, input_ids,
        num_tokens=args.num_tokens,
        warmup_tokens=args.warmup_tokens,
    )

    # ── Results ──────────────────────────────────────────────────────────
    ratio = (draft_stats["tokens_per_second"] /
             target_stats["tokens_per_second"]) if target_stats["tokens_per_second"] > 0 else 0

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"")
    print(f"  {'Model':<20s} {'ms/token':>10s} {'tok/s':>10s} {'Prefill':>10s}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Draft (0.3-bit)':<20s} "
          f"{draft_stats['avg_ms_per_token']:>9.1f}ms "
          f"{draft_stats['tokens_per_second']:>9.1f} "
          f"{draft_stats['prefill_time_s']:>9.2f}s")
    print(f"  {'Target (2.0-bit)':<20s} "
          f"{target_stats['avg_ms_per_token']:>9.1f}ms "
          f"{target_stats['tokens_per_second']:>9.1f} "
          f"{target_stats['prefill_time_s']:>9.2f}s")
    print(f"")
    print(f"  Draft/Target speed ratio: {ratio:.2f}x")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
