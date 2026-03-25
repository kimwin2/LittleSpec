"""
Diagnostic: Compare C++ kernel vs Python fallback vs Target model predictions.
Identifies root cause of low speculative decoding acceptance rate.

Tests:
  1. C++ full_forward hidden state vs Python full_forward hidden state (cosine sim)
  2. C++ generate_token (Q4 lm_head) top-k vs Python FP lm_head top-k
  3. Draft model top-1 vs Target model top-1 (acceptance prediction)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from cpu_draft_model import CPUDraftModel


def cosine_sim(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def main():
    RUNTIME_PATH = os.environ.get(
        "DRAFT_RUNTIME",
        "outputs/step1_draft_0.1bit/2026_03_24_07_56_runtime"
    )
    TARGET_PATH = os.environ.get(
        "TARGET_MODEL",
        os.environ.get("EAGLE_DIR", "/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct")
    )

    print("=" * 70)
    print("DIAGNOSTIC: Draft Model Accuracy Analysis")
    print("=" * 70)
    print(f"  Draft runtime: {RUNTIME_PATH}")
    print(f"  Target model:  {TARGET_PATH}")
    print()

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]
    print(f"Prompt: '{prompt}' ({seq_len} tokens)")
    print()

    # ===================================================================
    # TEST 1: C++ hidden state vs Python hidden state
    # ===================================================================
    print("=" * 70)
    print("TEST 1: C++ full_forward vs Python _forward_token (hidden state)")
    print("=" * 70)

    # Load draft model with C++ kernel
    draft_cpp = CPUDraftModel(RUNTIME_PATH, target_model_path=TARGET_PATH, max_seq_len=256)
    draft_cpp._ensure_cache()

    # Run Python path: process all tokens via _forward_token
    draft_cpp.reset()
    draft_cpp._ensure_cache()
    py_hiddens = []
    for pos in range(seq_len):
        tok = input_ids[0, pos].item()
        h = draft_cpp._forward_token(tok)
        py_hiddens.append(h.clone())
    py_last_hidden = py_hiddens[-1]  # (1, hidden_size)

    # Run C++ path: process all tokens via full_forward
    draft_cpp.reset()
    draft_cpp._ensure_cache()
    cpp_hiddens = []
    for pos in range(seq_len):
        tok = input_ids[0, pos].item()
        hidden = torch.ops.littlebit_cpu_ops.full_forward(
            tok,
            draft_cpp.embed_tokens,
            draft_cpp.lb_model.final_norm_weight.contiguous(),
            draft_cpp._layer_tensors,
            draft_cpp._kv_cache_tensors,
            draft_cpp._layer_dims,
            draft_cpp.config.num_hidden_layers,
            draft_cpp.config.hidden_size,
            draft_cpp.config.num_key_value_heads,
            draft_cpp.lb_model.kv_repeat,
            draft_cpp.lb_model.head_dim,
            draft_cpp.max_seq_len,
            pos,
            draft_cpp.lb_model.attn_scale,
            draft_cpp.config.rms_norm_eps,
        )
        cpp_hiddens.append(hidden.clone())
        draft_cpp._cache_pos = pos + 1
    cpp_last_hidden = cpp_hiddens[-1]

    sim = cosine_sim(py_last_hidden, cpp_last_hidden)
    l2 = (py_last_hidden.float() - cpp_last_hidden.float()).norm().item()
    print(f"  Cosine similarity: {sim:.6f}")
    print(f"  L2 distance:       {l2:.6f}")
    print(f"  Py hidden range:   [{py_last_hidden.min().item():.4f}, {py_last_hidden.max().item():.4f}]")
    print(f"  C++ hidden range:  [{cpp_last_hidden.min().item():.4f}, {cpp_last_hidden.max().item():.4f}]")
    if sim > 0.99:
        print("  ✅ PASS: C++ and Python hidden states match well")
    else:
        print("  ❌ FAIL: C++ and Python hidden states DIVERGE")
    print()

    # ===================================================================
    # TEST 2: Q4 lm_head vs FP lm_head (top-k comparison)
    # ===================================================================
    print("=" * 70)
    print("TEST 2: Q4 lm_head vs FP16 lm_head (token predictions)")
    print("=" * 70)

    # FP lm_head on Python hidden state
    fp_logits = draft_cpp._lm_head(py_last_hidden)
    fp_top10 = torch.topk(fp_logits.squeeze(), 10)
    fp_top1_id = fp_top10.indices[0].item()
    fp_top1_tok = tokenizer.decode([fp_top1_id])

    # Q4 generate_token on C++ path (uses Q4 lm_head)
    draft_cpp.reset()
    draft_cpp._ensure_cache()
    # Prefill all tokens
    for pos in range(seq_len):
        tok = input_ids[0, pos].item()
        draft_cpp._forward_token(tok)

    # Now use generate_token for the last position (it re-processes last token)
    last_tok = input_ids[0, -1].item()
    q4_token_id = torch.ops.littlebit_cpu_ops.generate_token(
        last_tok,
        draft_cpp.embed_tokens,
        draft_cpp.lb_model.final_norm_weight.contiguous(),
        draft_cpp._layer_tensors,
        draft_cpp._kv_cache_tensors,
        draft_cpp._layer_dims,
        draft_cpp._lm_head_q4,
        draft_cpp._vocab_size,
        draft_cpp.config.num_hidden_layers,
        draft_cpp.config.hidden_size,
        draft_cpp.config.num_key_value_heads,
        draft_cpp.lb_model.kv_repeat,
        draft_cpp.lb_model.head_dim,
        draft_cpp.max_seq_len,
        draft_cpp._cache_pos,
        draft_cpp.lb_model.attn_scale,
        draft_cpp.config.rms_norm_eps,
    )
    q4_top1_tok = tokenizer.decode([q4_token_id])

    print(f"  FP lm_head top-1:  id={fp_top1_id} '{fp_top1_tok}'")
    print(f"  Q4 lm_head top-1:  id={q4_token_id} '{q4_top1_tok}'")
    print(f"  FP top-10: {[tokenizer.decode([i.item()]) for i in fp_top10.indices]}")
    if fp_top1_id == q4_token_id:
        print("  ✅ PASS: Q4 and FP lm_head agree on top-1")
    else:
        in_top10 = q4_token_id in fp_top10.indices.tolist()
        print(f"  ⚠️  Q4 top-1 in FP top-10: {in_top10}")
        if not in_top10:
            print("  ❌ FAIL: Q4 lm_head severely distorts predictions")
    print()

    # ===================================================================
    # TEST 3: Draft top-1 vs Target top-1 (multi-position)
    # ===================================================================
    print("=" * 70)
    print("TEST 3: Draft model vs Target model (token-level acceptance)")
    print("=" * 70)

    # Load target model
    print("  Loading target model...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, torch_dtype=torch.float32, device_map="cpu"
    )
    target.eval()

    # Target: get predictions at each position
    with torch.no_grad():
        target_out = target(input_ids)
        target_logits = target_out.logits  # (1, seq_len, vocab)

    # Auto-regressive generation: compare draft vs target for 10 tokens
    print("\n  Position-by-position comparison (10 tokens):")
    print(f"  {'Pos':>4} | {'Target top-1':>15} | {'Draft(FP) top-1':>15} | {'Draft(Q4) top-1':>15} | Match?")
    print("  " + "-" * 80)

    # Reset draft for autoregressive generation
    draft_cpp.reset()
    draft_cpp._ensure_cache()

    # Prefill prompt
    for pos in range(seq_len):
        tok = input_ids[0, pos].item()
        draft_cpp._forward_token(tok)

    current_ids = input_ids.clone()
    matches_fp = 0
    matches_q4 = 0
    total = 0

    for step in range(10):
        with torch.no_grad():
            target_out = target(current_ids)
        target_logits_last = target_out.logits[0, -1, :]
        target_top1 = torch.argmax(target_logits_last).item()
        target_tok = tokenizer.decode([target_top1])

        # Draft FP prediction
        draft_fp_logits = draft_cpp._lm_head(draft_cpp._last_hidden)
        draft_fp_top1 = torch.argmax(draft_fp_logits.squeeze()).item()
        draft_fp_tok = tokenizer.decode([draft_fp_top1])

        # Draft Q4 prediction (generate_token at current pos)
        draft_q4_id = torch.ops.littlebit_cpu_ops.generate_token(
            current_ids[0, -1].item(),
            draft_cpp.embed_tokens,
            draft_cpp.lb_model.final_norm_weight.contiguous(),
            draft_cpp._layer_tensors,
            draft_cpp._kv_cache_tensors,
            draft_cpp._layer_dims,
            draft_cpp._lm_head_q4,
            draft_cpp._vocab_size,
            draft_cpp.config.num_hidden_layers,
            draft_cpp.config.hidden_size,
            draft_cpp.config.num_key_value_heads,
            draft_cpp.lb_model.kv_repeat,
            draft_cpp.lb_model.head_dim,
            draft_cpp.max_seq_len,
            draft_cpp._cache_pos,
            draft_cpp.lb_model.attn_scale,
            draft_cpp.config.rms_norm_eps,
        )
        # Rollback the KV write from generate_token (it wrote at _cache_pos)
        draft_q4_tok = tokenizer.decode([draft_q4_id])

        match_fp = "✅" if draft_fp_top1 == target_top1 else "❌"
        match_q4 = "✅" if draft_q4_id == target_top1 else "❌"
        if draft_fp_top1 == target_top1:
            matches_fp += 1
        if draft_q4_id == target_top1:
            matches_q4 += 1
        total += 1

        pos_label = seq_len + step
        print(f"  {pos_label:>4} | {target_tok:>15} | {draft_fp_tok:>15} {match_fp} | {draft_q4_tok:>15} {match_q4}")

        # Advance with target's token
        next_id = torch.tensor([[target_top1]])
        current_ids = torch.cat([current_ids, next_id], dim=1)
        draft_cpp._forward_token(target_top1)

    print()
    print(f"  Draft(FP) match rate:  {matches_fp}/{total} = {matches_fp/total*100:.0f}%")
    print(f"  Draft(Q4) match rate:  {matches_q4}/{total} = {matches_q4/total*100:.0f}%")
    print()

    if matches_fp / total < 0.1:
        print("  ⚠️  DIAGNOSIS: Model checkpoint quality is low (FP lm_head also fails)")
        print("     → The 0.1-bit draft model itself doesn't predict well")
        print("     → Consider using a better-trained checkpoint or higher bit-rate")
    elif matches_q4 / total < matches_fp / total * 0.5:
        print("  ⚠️  DIAGNOSIS: Q4 lm_head quantization is too lossy")
        print("     → Consider using Q8 or FP16 lm_head instead")
    else:
        print("  ✅ Model and quantization look reasonable")

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
