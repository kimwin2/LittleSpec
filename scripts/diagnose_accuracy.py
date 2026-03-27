"""
Diagnostic: Deep analysis of draft model prediction failures.

Tests (progressive elimination):
  TEST 0: Embedding comparison (draft vs target)
  TEST 1: GPU HF-quantized model vs Target (isolate checkpoint vs runtime issue)
  TEST 2: CPU runtime hidden state vs GPU HF hidden state (conversion issue?)
  TEST 3: Layer-by-layer hidden state divergence (draft vs target)
  TEST 4: Logit distribution analysis (KL divergence, top-k overlap)
  TEST 5: Token-level autoregressive comparison (existing, enhanced)

Usage:
    # Quick (CPU runtime only, no GPU needed):
    python scripts/diagnose_accuracy.py

    # Full (includes GPU HF model comparison - needs GPU):
    python scripts/diagnose_accuracy.py --full

    # Custom paths:
    DRAFT_RUNTIME=path/to/runtime TARGET_MODEL=path/to/fp python scripts/diagnose_accuracy.py
    
    # Custom prompt and more tokens:
    python scripts/diagnose_accuracy.py --prompt "Hello world" --num_tokens 20
    
    # With HF checkpoint (not runtime) for comparison:
    python scripts/diagnose_accuracy.py --full --hf_ckpt path/to/hf_checkpoint
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def cosine_sim(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def kl_divergence(p_logits, q_logits):
    """KL(P || Q) where P=target, Q=draft. Lower = more similar."""
    p = F.softmax(p_logits.float().squeeze(), dim=-1)
    q = F.softmax(q_logits.float().squeeze(), dim=-1)
    # Clamp to avoid log(0)
    q = q.clamp(min=1e-10)
    p = p.clamp(min=1e-10)
    return (p * (p.log() - q.log())).sum().item()


def top_k_overlap(logits_a, logits_b, k=10):
    """Fraction of top-k tokens that overlap between two logit distributions."""
    top_a = set(torch.topk(logits_a.squeeze(), k).indices.tolist())
    top_b = set(torch.topk(logits_b.squeeze(), k).indices.tolist())
    return len(top_a & top_b) / k


def rank_of_target(logits, target_id):
    """Rank of target_id in the sorted logits (0-indexed, 0=top-1)."""
    sorted_ids = torch.argsort(logits.squeeze(), descending=True)
    positions = (sorted_ids == target_id).nonzero(as_tuple=True)[0]
    if len(positions) > 0:
        return positions[0].item()
    return -1


def print_separator(title, char="=", width=80):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def main():
    parser = argparse.ArgumentParser(description="Deep diagnostic for draft model accuracy")
    parser.add_argument("--full", action="store_true",
                        help="Run full diagnostics including GPU HF model comparison")
    parser.add_argument("--hf_ckpt", type=str, default=None,
                        help="Path to HF format checkpoint (pre-conversion) for comparison")
    parser.add_argument("--prompt", type=str, default="The capital of France is",
                        help="Prompt for testing")
    parser.add_argument("--num_tokens", type=int, default=15,
                        help="Number of autoregressive tokens to compare")
    args = parser.parse_args()

    RUNTIME_PATH = os.environ.get(
        "DRAFT_RUNTIME",
        "outputs/step1_draft_0.1bit/2026_03_24_07_56_runtime"
    )
    TARGET_PATH = os.environ.get(
        "TARGET_MODEL",
        os.environ.get("EAGLE_DIR", "/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct")
    )

    print("=" * 80)
    print("  DEEP DIAGNOSTIC: Draft Model Accuracy Analysis")
    print("=" * 80)
    print(f"  Draft runtime:  {RUNTIME_PATH}")
    print(f"  Target model:   {TARGET_PATH}")
    if args.hf_ckpt:
        print(f"  HF checkpoint:  {args.hf_ckpt}")
    print(f"  Mode:           {'full (GPU+CPU)' if args.full else 'quick (CPU only)'}")
    print()

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)
    prompt = args.prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]
    print(f"  Prompt: '{prompt}' ({seq_len} tokens)")
    tokens_str = [tokenizer.decode([input_ids[0, i].item()]) for i in range(seq_len)]
    print(f"  Tokens: {tokens_str}")
    print()

    # ===================================================================
    # LOAD MODELS
    # ===================================================================
    from cpu_draft_model import CPUDraftModel

    print("Loading CPU draft model (runtime)...")
    draft_cpu = CPUDraftModel(RUNTIME_PATH, base_model_id=TARGET_PATH)
    
    print("Loading FP target model...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, torch_dtype=torch.float32, device_map="cpu",
        output_hidden_states=True
    )
    target.eval()
    
    # Optionally load GPU HF quantized model
    draft_gpu = None
    if args.full:
        hf_ckpt = args.hf_ckpt
        if hf_ckpt is None:
            # Try to infer from runtime path
            if RUNTIME_PATH.endswith("_runtime"):
                hf_ckpt = RUNTIME_PATH[:-len("_runtime")]
                if os.path.exists(hf_ckpt):
                    print(f"  Auto-detected HF checkpoint: {hf_ckpt}")
                else:
                    hf_ckpt = None
        
        if hf_ckpt and os.path.exists(hf_ckpt):
            print(f"Loading HF quantized draft model from {hf_ckpt}...")
            try:
                from speculative_decoding import MatryoshkaDraftModel
                device = "cuda" if torch.cuda.is_available() else "cpu"
                draft_gpu = MatryoshkaDraftModel(hf_ckpt, torch_dtype=torch.float32, device=device)
                print(f"  HF draft model loaded on {draft_gpu.device}")
            except Exception as e:
                print(f"  ⚠️  Could not load HF draft model: {e}")
                draft_gpu = None
        else:
            print("  ⚠️  No HF checkpoint found. Skipping GPU comparison.")
            print("     Use --hf_ckpt to specify path, or ensure <runtime>_runtime naming convention.")

    # ===================================================================
    # TEST 0: Embedding Comparison
    # ===================================================================
    print_separator("TEST 0: Embedding Comparison")
    
    # Draft embedding
    draft_embed = draft_cpu.embed_tokens  # (vocab, hidden)
    target_embed = target.model.embed_tokens.weight.data.float()
    
    embed_sim = cosine_sim(draft_embed, target_embed)
    embed_l2 = (draft_embed - target_embed).norm().item()
    embed_max_diff = (draft_embed - target_embed).abs().max().item()
    
    print(f"  Cosine similarity:  {embed_sim:.6f}")
    print(f"  L2 distance:        {embed_l2:.2f}")
    print(f"  Max abs difference: {embed_max_diff:.6f}")
    
    if embed_sim > 0.9999:
        print("  ✅ Embeddings are identical (shared from base model)")
    elif embed_sim > 0.999:
        print("  ⚠️  Embeddings are very close but not identical")
    else:
        print("  ❌ Embeddings DIFFER — draft may have fine-tuned embeddings")
    
    # Check a few specific token embeddings
    test_token_ids = [input_ids[0, 0].item(), 128000, 1]  # first input token, BOS, common token
    for tid in test_token_ids:
        d_emb = draft_embed[tid]
        t_emb = target_embed[tid]
        sim = cosine_sim(d_emb, t_emb)
        print(f"    Token {tid:>6} ('{tokenizer.decode([tid])}'): cos_sim={sim:.8f}")

    # ===================================================================
    # TEST 1: GPU HF Model vs Target (if --full)
    # ===================================================================
    if args.full and draft_gpu is not None:
        print_separator("TEST 1: GPU HF Quantized Model vs Target (Training Accuracy)")
        
        device = draft_gpu.device
        input_ids_gpu = input_ids.to(device)
        
        with torch.no_grad():
            # Target forward
            target_input = input_ids.to("cpu") if str(device) != "cpu" else input_ids
            target_out = target(target_input, output_hidden_states=True)
            target_logits = target_out.logits.float()
            
            # GPU draft forward
            gpu_out = draft_gpu.model(input_ids_gpu, output_hidden_states=True)
            gpu_logits = gpu_out.logits.float().cpu()
        
        # Compare at each position
        print(f"\n  Per-position analysis (prompt positions):")
        print(f"  {'Pos':>4} | {'Token':>12} | {'Target →':>12} | {'GPU Draft →':>12} | {'Match':>5} | {'KL(T||D)':>10} | {'Top10 Ovlp':>10} | {'Rank':>6}")
        print(f"  {'─'*4}-+-{'─'*12}-+-{'─'*12}-+-{'─'*12}-+-{'─'*5}-+-{'─'*10}-+-{'─'*10}-+-{'─'*6}")
        
        gpu_matches = 0
        for pos in range(seq_len):
            t_logits = target_logits[0, pos, :]
            d_logits = gpu_logits[0, pos, :]
            t_top1 = torch.argmax(t_logits).item()
            d_top1 = torch.argmax(d_logits).item()
            
            match = "✅" if t_top1 == d_top1 else "❌"
            if t_top1 == d_top1: gpu_matches += 1
            
            kl = kl_divergence(t_logits, d_logits)
            top10_ovlp = top_k_overlap(t_logits, d_logits, k=10)
            rank = rank_of_target(d_logits, t_top1)
            
            in_tok = tokenizer.decode([input_ids[0, pos].item()])
            t_tok = tokenizer.decode([t_top1])
            d_tok = tokenizer.decode([d_top1])
            
            print(f"  {pos:>4} | {in_tok:>12} | {t_tok:>12} | {d_tok:>12} | {match:>5} | {kl:>10.4f} | {top10_ovlp:>10.1%} | {rank:>6}")
        
        print(f"\n  GPU HF draft match rate: {gpu_matches}/{seq_len} = {gpu_matches/seq_len*100:.1f}%")
        
        if gpu_matches / seq_len > 0.4:
            print("  ✅ GPU HF model accuracy is reasonable (close to training ac0)")
            print("     → Problem is in runtime conversion or CPU kernel, NOT the model itself")
        else:
            print("  ❌ GPU HF model accuracy is also low")
            print("     → Problem is in the model checkpoint itself (training quality)")
    else:
        print_separator("TEST 1: [SKIPPED - use --full to enable GPU HF model comparison]")

    # ===================================================================
    # TEST 2: Layer-by-Layer Hidden State Divergence (Draft vs Target)
    # ===================================================================
    print_separator("TEST 2: Layer-by-Layer Hidden State Comparison (CPU Draft vs Target)")
    
    # Target: get all layer hidden states
    with torch.no_grad():
        target_out = target(input_ids, output_hidden_states=True)
        target_hiddens = [h.float() for h in target_out.hidden_states]  # [embed, layer0, ..., layer31]
        target_logits_full = target_out.logits.float()
    
    # Draft CPU: get hidden state after each layer
    # We can't easily get per-layer from the C++ kernel, so use Python fallback
    draft_cpu.reset()
    draft_cpu._ensure_cache()
    
    # We need per-layer hidden states from draft. Use Python path with hooks.
    from littlebit_kernels_cpu.runtime import littlebit_linear as lb_linear_fn
    from littlebit_kernels_cpu.dummy_model import (
        _group_query_heads, _cache_write_grouped, _grouped_attention_context
    )
    
    eps = draft_cpu.config.rms_norm_eps
    
    # Process all tokens and collect per-layer hidden states at last position
    draft_layer_hiddens = []  # Will store layer outputs at last token position
    
    for pos in range(seq_len):
        tok = input_ids[0, pos].item()
        hidden = draft_cpu._embed(torch.tensor([tok], dtype=torch.long))
        hidden = hidden.reshape(1, draft_cpu.config.hidden_size)
        x = hidden.to(torch.float32)
        
        if pos == seq_len - 1:
            # Store embedding output
            draft_layer_hiddens.append(x.clone())
        
        for layer_idx, (layer, layer_cache) in enumerate(zip(draft_cpu.lb_model.layers, draft_cpu._cache)):
            residual = x
            from cpu_draft_model import _cpp_rms_norm, _cpp_silu_mul
            normed = _cpp_rms_norm(x, layer.input_layernorm_weight, eps)
            
            q = lb_linear_fn(normed, layer.q_proj).to(torch.float32)
            k = lb_linear_fn(normed, layer.k_proj).to(torch.float32)
            v = lb_linear_fn(normed, layer.v_proj).to(torch.float32)
            
            q_grouped = _group_query_heads(q, num_key_value_heads=draft_cpu.config.num_key_value_heads,
                kv_repeat=draft_cpu.lb_model.kv_repeat, head_dim=draft_cpu.lb_model.head_dim)
            keys, values = _cache_write_grouped(layer_cache, k, v,
                position=pos, num_key_value_heads=draft_cpu.config.num_key_value_heads,
                head_dim=draft_cpu.lb_model.head_dim)
            attn_out = _grouped_attention_context(q_grouped, keys, values,
                attn_scale=draft_cpu.lb_model.attn_scale)
            
            x = residual + lb_linear_fn(attn_out, layer.o_proj).to(torch.float32)
            residual = x
            mlp_in = _cpp_rms_norm(x, layer.post_attention_layernorm_weight, eps)
            gate = lb_linear_fn(mlp_in, layer.gate_proj).to(torch.float32)
            up = lb_linear_fn(mlp_in, layer.up_proj).to(torch.float32)
            mlp_hidden = _cpp_silu_mul(gate, up)
            x = residual + lb_linear_fn(mlp_hidden, layer.down_proj).to(torch.float32)
            
            if pos == seq_len - 1:
                draft_layer_hiddens.append(x.clone())
        
        draft_cpu._cache_pos = pos + 1
    
    # Compare layer by layer at the last token position
    num_layers = draft_cpu.config.num_hidden_layers
    print(f"\n  Hidden state comparison at last token position (pos={seq_len-1}):")
    print(f"  {'Layer':>8} | {'Cos Sim':>10} | {'L2 Dist':>10} | {'Draft Norm':>12} | {'Target Norm':>12} | {'Status'}")
    print(f"  {'─'*8}-+-{'─'*10}-+-{'─'*10}-+-{'─'*12}-+-{'─'*12}-+-{'─'*12}")
    
    first_bad_layer = None
    for i in range(min(len(draft_layer_hiddens), len(target_hiddens))):
        draft_h = draft_layer_hiddens[i]
        target_h = target_hiddens[i][0, seq_len - 1, :].unsqueeze(0)  # Extract last token pos
        
        sim = cosine_sim(draft_h, target_h)
        l2 = (draft_h.float() - target_h.float()).norm().item()
        d_norm = draft_h.float().norm().item()
        t_norm = target_h.float().norm().item()
        
        layer_name = "embed" if i == 0 else f"layer {i-1}"
        
        if sim > 0.95:
            status = "✅ OK"
        elif sim > 0.80:
            status = "⚠️  degrading"
            if first_bad_layer is None: first_bad_layer = i
        else:
            status = "❌ DIVERGED"
            if first_bad_layer is None: first_bad_layer = i
        
        print(f"  {layer_name:>8} | {sim:>10.6f} | {l2:>10.2f} | {d_norm:>12.2f} | {t_norm:>12.2f} | {status}")
    
    if first_bad_layer is not None:
        layer_label = "embedding" if first_bad_layer == 0 else f"layer {first_bad_layer - 1}"
        print(f"\n  📍 Divergence starts at: {layer_label}")
        if first_bad_layer == 0:
            print("     → Embeddings differ between draft and target")
        elif first_bad_layer <= 3:
            print("     → Early layer divergence — 0.1-bit quantization is too aggressive for this model")
        else:
            print("     → Divergence accumulates through layers (expected for 0.1-bit)")
    else:
        print(f"\n  ✅ All layers are well-matched")

    # ===================================================================
    # TEST 3: Logit Distribution Analysis
    # ===================================================================
    print_separator("TEST 3: Logit Distribution Analysis (Draft vs Target)")
    
    # Get draft logits at final position using FP lm_head
    draft_final_hidden = draft_layer_hiddens[-1] if draft_layer_hiddens else None
    if draft_final_hidden is not None:
        # Apply final norm
        draft_final_normed = _cpp_rms_norm(
            draft_final_hidden, draft_cpu.lb_model.final_norm_weight, eps
        )
        draft_logits = draft_cpu._lm_head(draft_final_normed)
        target_last_logits = target_logits_full[0, -1, :]
        
        # Overall stats
        kl = kl_divergence(target_last_logits, draft_logits.squeeze())
        top1_ovlp = top_k_overlap(target_last_logits, draft_logits, k=1)
        top5_ovlp = top_k_overlap(target_last_logits, draft_logits, k=5)
        top10_ovlp = top_k_overlap(target_last_logits, draft_logits, k=10)
        top50_ovlp = top_k_overlap(target_last_logits, draft_logits, k=50)
        top100_ovlp = top_k_overlap(target_last_logits, draft_logits, k=100)
        
        t_top1 = torch.argmax(target_last_logits).item()
        d_top1 = torch.argmax(draft_logits.squeeze()).item()
        target_rank_in_draft = rank_of_target(draft_logits, t_top1)
        
        print(f"  KL(Target || Draft): {kl:.4f}")
        print(f"  Top-1 overlap:   {top1_ovlp:.0%}")
        print(f"  Top-5 overlap:   {top5_ovlp:.0%}")
        print(f"  Top-10 overlap:  {top10_ovlp:.0%}")
        print(f"  Top-50 overlap:  {top50_ovlp:.0%}")
        print(f"  Top-100 overlap: {top100_ovlp:.0%}")
        print()
        print(f"  Target top-1: id={t_top1} '{tokenizer.decode([t_top1])}'")
        print(f"  Draft top-1:  id={d_top1} '{tokenizer.decode([d_top1])}'")
        print(f"  Target's top-1 rank in draft: #{target_rank_in_draft + 1}")
        print()
        
        # Show top-10 comparison
        t_top10 = torch.topk(target_last_logits, 10)
        d_top10 = torch.topk(draft_logits.squeeze(), 10)
        
        print(f"  Target top-10:  {[tokenizer.decode([i.item()]) for i in t_top10.indices]}")
        print(f"  Draft  top-10:  {[tokenizer.decode([i.item()]) for i in d_top10.indices]}")
        print()
        
        # Entropy analysis
        t_entropy = -(F.softmax(target_last_logits, dim=-1) * F.log_softmax(target_last_logits, dim=-1)).sum().item()
        d_entropy = -(F.softmax(draft_logits.squeeze(), dim=-1) * F.log_softmax(draft_logits.squeeze(), dim=-1)).sum().item()
        print(f"  Target entropy: {t_entropy:.4f} (higher = more uncertain)")
        print(f"  Draft  entropy: {d_entropy:.4f}")
        
        if kl > 10:
            print("\n  ❌ KL divergence is very high — draft logits are far from target")
        elif kl > 3:
            print("\n  ⚠️  KL divergence is moderate — some useful information preserved")
        else:
            print("\n  ✅ KL divergence is low — draft logits are close to target")
        
        if target_rank_in_draft > 100:
            print(f"  ❌ Target's top-1 is ranked #{target_rank_in_draft+1} in draft — very poor alignment")
        elif target_rank_in_draft > 10:
            print(f"  ⚠️  Target's top-1 is ranked #{target_rank_in_draft+1} in draft — weak alignment")
        elif target_rank_in_draft > 0:
            print(f"  ⚠️  Target's top-1 is in draft's top-{target_rank_in_draft+1} — close but not top-1")

    # ===================================================================
    # TEST 4: lm_head Weight Comparison
    # ===================================================================
    print_separator("TEST 4: lm_head Weight Analysis")
    
    draft_lm = draft_cpu.lm_head_weight  # (vocab, hidden)
    target_lm = target.lm_head.weight.data.float()
    
    lm_sim = cosine_sim(draft_lm, target_lm)
    lm_l2 = (draft_lm - target_lm).norm().item()
    lm_max_diff = (draft_lm - target_lm).abs().max().item()
    
    print(f"  Cosine similarity:  {lm_sim:.6f}")
    print(f"  L2 distance:        {lm_l2:.2f}")
    print(f"  Max abs difference: {lm_max_diff:.6f}")
    print(f"  Draft lm_head norm: {draft_lm.norm().item():.2f}")
    print(f"  Target lm_head norm:{target_lm.norm().item():.2f}")
    
    if lm_sim > 0.9999:
        print("  ✅ lm_head weights are identical (shared from base model)")
    else:
        print("  ❌ lm_head weights DIFFER")
        print("     → Draft model may have fine-tuned lm_head during training")

    # ===================================================================
    # TEST 5: Token-Level Autoregressive Comparison (Enhanced)
    # ===================================================================
    print_separator(f"TEST 5: Autoregressive Token Comparison ({args.num_tokens} tokens)")
    
    # Reset draft for autoregressive generation
    draft_cpu.reset()
    draft_cpu._ensure_cache()
    
    # Prefill prompt
    for pos in range(seq_len):
        tok = input_ids[0, pos].item()
        draft_cpu._forward_token(tok)
    
    current_ids = input_ids.clone()
    matches_fp = 0
    matches_q4 = 0
    total = 0
    kl_sum = 0
    avg_rank_sum = 0
    top5_overlap_sum = 0
    
    print(f"\n  {'Pos':>4} | {'Target':>12} | {'Draft(FP)':>12} | {'Draft(Q4)':>12} | {'KL':>8} | {'T-Rank':>7} | {'Top5':>5}")
    print(f"  {'─'*4}-+-{'─'*12}-+-{'─'*12}-+-{'─'*12}-+-{'─'*8}-+-{'─'*7}-+-{'─'*5}")

    for step in range(args.num_tokens):
        with torch.no_grad():
            target_out = target(current_ids)
        target_logits_last = target_out.logits[0, -1, :].float()
        target_top1 = torch.argmax(target_logits_last).item()
        target_tok = tokenizer.decode([target_top1])

        # Draft FP prediction
        draft_fp_logits = draft_cpu._lm_head(draft_cpu._last_hidden).squeeze()
        draft_fp_top1 = torch.argmax(draft_fp_logits).item()
        draft_fp_tok = tokenizer.decode([draft_fp_top1])

        # Draft Q4 prediction
        if draft_cpu._use_generate_token:
            draft_q4_id = torch.ops.littlebit_cpu_ops.generate_token(
                current_ids[0, -1].item(),
                draft_cpu.embed_tokens,
                draft_cpu.lb_model.final_norm_weight.contiguous(),
                draft_cpu._layer_tensors,
                draft_cpu._kv_cache_tensors,
                draft_cpu._layer_dims,
                draft_cpu._lm_head_q4,
                draft_cpu._vocab_size,
                draft_cpu.config.num_hidden_layers,
                draft_cpu.config.hidden_size,
                draft_cpu.config.num_key_value_heads,
                draft_cpu.lb_model.kv_repeat,
                draft_cpu.lb_model.head_dim,
                draft_cpu.max_seq_len,
                draft_cpu._cache_pos,
                draft_cpu.lb_model.attn_scale,
                draft_cpu.config.rms_norm_eps,
            )
            draft_q4_tok = tokenizer.decode([draft_q4_id])
        else:
            draft_q4_id = draft_fp_top1
            draft_q4_tok = draft_fp_tok

        match_fp = "✅" if draft_fp_top1 == target_top1 else "❌"
        match_q4 = "✅" if draft_q4_id == target_top1 else "❌"
        if draft_fp_top1 == target_top1: matches_fp += 1
        if draft_q4_id == target_top1: matches_q4 += 1
        total += 1

        # Logit analysis
        kl = kl_divergence(target_logits_last, draft_fp_logits)
        kl_sum += kl
        rank = rank_of_target(draft_fp_logits, target_top1)
        avg_rank_sum += rank
        t5_ovlp = top_k_overlap(target_logits_last, draft_fp_logits, k=5)
        top5_overlap_sum += t5_ovlp

        pos_label = seq_len + step
        print(f"  {pos_label:>4} | {target_tok:>12}{match_fp} | {draft_fp_tok:>12} | {draft_q4_tok:>12} | {kl:>8.2f} | {rank:>7} | {t5_ovlp:>5.1%}")

        # Advance with target's token (teacher-forced)
        next_id = torch.tensor([[target_top1]])
        current_ids = torch.cat([current_ids, next_id], dim=1)
        draft_cpu._forward_token(target_top1)

    print()
    print(f"  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │ SUMMARY                                                │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │ Draft(FP) match rate:  {matches_fp:>3}/{total:>3} = {matches_fp/total*100:>5.1f}%{' ':>17}│")
    print(f"  │ Draft(Q4) match rate:  {matches_q4:>3}/{total:>3} = {matches_q4/total*100:>5.1f}%{' ':>17}│")
    print(f"  │ Avg KL divergence:     {kl_sum/total:>9.4f}{' ':>22}│")
    print(f"  │ Avg target rank in draft: #{avg_rank_sum/total:>6.1f}{' ':>19}│")
    print(f"  │ Avg top-5 overlap:     {top5_overlap_sum/total:>9.1%}{' ':>22}│")
    print(f"  └─────────────────────────────────────────────────────────┘")

    # ===================================================================
    # OVERALL DIAGNOSIS
    # ===================================================================
    print_separator("DIAGNOSIS", char="█")
    
    avg_rank = avg_rank_sum / total if total > 0 else 999
    avg_kl = kl_sum / total if total > 0 else 999
    fp_match_rate = matches_fp / total if total > 0 else 0
    
    # Determine root cause
    issues = []
    
    if embed_sim < 0.999:
        issues.append(("Embedding mismatch", "Embeddings differ between draft and target. Check if training modified embed_tokens."))
    
    if first_bad_layer is not None and first_bad_layer <= 2:
        issues.append(("Early layer divergence", f"Hidden states diverge from layer {first_bad_layer}. 0.1-bit quantization may be too aggressive."))
    
    if lm_sim < 0.999:
        issues.append(("lm_head mismatch", "lm_head weights differ. Check if training modified lm_head."))
    
    if fp_match_rate < 0.1:
        if avg_rank > 50:
            issues.append(("Severe model quality issue", f"Target's top-1 is ranked #{avg_rank:.0f} on average in draft logits. The model fundamentally can't predict the right tokens."))
        elif avg_rank > 5:
            issues.append(("Moderate model quality issue", f"Target's top-1 is ranked #{avg_rank:.0f} on average — in the ballpark but not top-1."))
        else:
            issues.append(("Close but not matching", f"Target's top-1 is only ranked #{avg_rank:.1f} in draft — might improve with more training."))
    
    if avg_kl > 10:
        issues.append(("High KL divergence", f"Average KL = {avg_kl:.2f}. Logit distributions are very different."))
    
    if not issues:
        print("\n  ✅ No major issues detected. Model looks reasonable.")
    else:
        print(f"\n  Found {len(issues)} issue(s):\n")
        for i, (title, desc) in enumerate(issues, 1):
            print(f"  {i}. ❌ {title}")
            print(f"     {desc}")
            print()
    
    # Actionable recommendations
    print("  Recommended next steps:")
    if fp_match_rate < 0.1 and avg_rank > 50:
        print("  1. Check if the correct checkpoint was loaded (verify training loss converged)")
        print("  2. Try a checkpoint from a different training step (maybe overfit or underfit)")
        print("  3. Run with --full to compare GPU HF model vs CPU runtime")
        print("  4. Consider higher bit-rate (e.g., 0.5-bit or 1-bit) for this model")
    elif fp_match_rate < 0.3 and avg_rank <= 10:
        print("  1. Model quality is borderline — more training epochs may help")
        print("  2. Use sampling instead of greedy for speculative decoding (higher acceptance)")
        print("  3. Consider knowledge distillation with layer-to-layer loss")
    elif fp_match_rate >= 0.3:
        print("  1. Model quality looks acceptable for speculative decoding")
        print("  2. If runtime acceptance is lower, check CPU kernel / runtime conversion")
    
    print()
    print("=" * 80)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
