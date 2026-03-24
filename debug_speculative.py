"""
Debug script: Compare draft model vs target model token predictions.
Identifies why acceptance rate is 0%.
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from speculative_decoding import MatryoshkaDraftModel, FPTargetModel
from utils.datautils import load_tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str, required=True)
    parser.add_argument("--draft_model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.base_model_id)

    # Load draft model
    print("Loading draft model...")
    draft_model = MatryoshkaDraftModel(
        args.draft_model_path, torch_dtype=torch.bfloat16, device=str(device)
    )

    # Load target model
    print("Loading target model...")
    target_model = FPTargetModel(
        args.base_model_id, torch_dtype=torch.bfloat16, device=str(device)
    )

    # Test prompt
    prompt = "Write a Python function to compute fibonacci numbers."
    chat_messages = [{"role": "user", "content": prompt}]
    try:
        prompt_text = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt_text = prompt

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    seq_len = input_ids.shape[1]
    print(f"\nInput sequence length: {seq_len}")

    # ========== Test 1: Draft model single forward (no KV cache) ==========
    print("\n" + "="*60)
    print("TEST 1: Draft model forward (no KV cache)")
    print("="*60)
    with torch.no_grad():
        draft_out = draft_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    draft_logits_nocache = draft_out.logits[:, -1, :]
    draft_token_nocache = torch.argmax(draft_logits_nocache, dim=-1)
    draft_top5_nocache = torch.topk(draft_logits_nocache, 5, dim=-1)
    print(f"Draft (no cache) next token: {draft_token_nocache.item()} "
          f"= '{tokenizer.decode([draft_token_nocache.item()])}'")
    print(f"Draft top-5 tokens: {draft_top5_nocache.indices[0].tolist()}")
    print(f"Draft top-5 decoded: {[tokenizer.decode([t]) for t in draft_top5_nocache.indices[0].tolist()]}")
    print(f"Draft logits stats: min={draft_logits_nocache.min():.4f}, max={draft_logits_nocache.max():.4f}, "
          f"mean={draft_logits_nocache.mean():.4f}, std={draft_logits_nocache.std():.4f}")

    # ========== Test 2: Draft model with KV cache (generate_draft_tokens) ==========
    print("\n" + "="*60)
    print("TEST 2: Draft model generate_draft_tokens (with KV cache)")
    print("="*60)
    draft_tokens, draft_probs = draft_model.generate_draft_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
        draft_length=4,
        temperature=1.0,
        greedy=True,
    )
    print(f"Draft tokens (K=4): {[t.item() for t in draft_tokens]}")
    print(f"Draft tokens decoded: {[tokenizer.decode([t.item()]) for t in draft_tokens]}")

    # ========== Test 3: Target model forward ==========
    print("\n" + "="*60)
    print("TEST 3: Target model forward")
    print("="*60)
    with torch.no_grad():
        target_logits = target_model.forward(input_ids, attention_mask)
    target_next_logits = target_logits[:, -1, :]
    target_token = torch.argmax(target_next_logits, dim=-1)
    target_top5 = torch.topk(target_next_logits, 5, dim=-1)
    print(f"Target next token: {target_token.item()} "
          f"= '{tokenizer.decode([target_token.item()])}'")
    print(f"Target top-5 tokens: {target_top5.indices[0].tolist()}")
    print(f"Target top-5 decoded: {[tokenizer.decode([t]) for t in target_top5.indices[0].tolist()]}")

    # ========== Test 4: Simulate speculative decode verification ==========
    print("\n" + "="*60)
    print("TEST 4: Simulate speculative verification (like the actual code)")
    print("="*60)
    draft_token_ids = torch.cat(draft_tokens, dim=1)  # (1, K)
    verify_ids = torch.cat([input_ids, draft_token_ids], dim=1)
    if attention_mask is not None:
        verify_mask = torch.cat([
            attention_mask,
            torch.ones(1, 4, device=device, dtype=attention_mask.dtype)
        ], dim=1)
    else:
        verify_mask = None

    with torch.no_grad():
        verify_logits = target_model.forward(verify_ids, verify_mask)

    print(f"\nVerify ids shape: {verify_ids.shape}")
    print(f"Verify logits shape: {verify_logits.shape}")
    print(f"seq_len (current_ids length): {seq_len}")

    for i in range(4):
        # This is the index used in speculative_decode
        logit_idx = seq_len - 1 + i
        target_logit_i = verify_logits[:, logit_idx, :]
        target_token_i = torch.argmax(target_logit_i, dim=-1)
        draft_token_i = draft_tokens[i].item()
        match = "✓ MATCH" if target_token_i.item() == draft_token_i else "✗ MISMATCH"

        target_top3 = torch.topk(target_logit_i, 3, dim=-1)
        print(f"\n  Position {i}: logit_idx={logit_idx}")
        print(f"    Draft token:  {draft_token_i} = '{tokenizer.decode([draft_token_i])}'")
        print(f"    Target token: {target_token_i.item()} = '{tokenizer.decode([target_token_i.item()])}'")
        print(f"    Target top-3: {target_top3.indices[0].tolist()} = {[tokenizer.decode([t]) for t in target_top3.indices[0].tolist()]}")
        print(f"    Result: {match}")

    # ========== Test 5: Check if draft model logits are degenerate ==========
    print("\n" + "="*60)
    print("TEST 5: Draft model logit distribution analysis")
    print("="*60)
    with torch.no_grad():
        draft_full_out = draft_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    draft_full_logits = draft_full_out.logits  # (1, seq_len, vocab)

    # Check if logits are all the same (degenerate)
    for pos in [0, seq_len//2, seq_len-1]:
        logits_at_pos = draft_full_logits[:, pos, :]
        probs_at_pos = F.softmax(logits_at_pos, dim=-1)
        entropy = -(probs_at_pos * torch.log(probs_at_pos + 1e-10)).sum().item()
        top_prob = probs_at_pos.max().item()
        unique_vals = torch.unique(logits_at_pos).numel()
        print(f"  Position {pos}: entropy={entropy:.4f}, top_prob={top_prob:.6f}, "
              f"unique_logit_vals={unique_vals}, "
              f"logits_range=[{logits_at_pos.min():.4f}, {logits_at_pos.max():.4f}]")

    # ========== Test 6: Direct top-1 match rate across all positions ==========
    print("\n" + "="*60)
    print("TEST 6: Token-level top-1 match rate (draft vs target)")
    print("="*60)
    with torch.no_grad():
        target_full_logits = target_model.forward(input_ids, attention_mask)

    draft_top1 = torch.argmax(draft_full_logits[:, :-1, :], dim=-1)  # predictions for pos 1..N
    target_top1 = torch.argmax(target_full_logits[:, :-1, :], dim=-1)
    match_mask = (draft_top1 == target_top1)
    match_rate = match_mask.float().mean().item()
    print(f"  Top-1 match rate across all positions: {match_rate:.4f} ({match_mask.sum().item()}/{match_mask.numel()})")

    # Show first 10 mismatches
    mismatches = (~match_mask).nonzero(as_tuple=True)
    if len(mismatches[1]) > 0:
        print(f"\n  First 10 mismatches:")
        for idx in range(min(10, len(mismatches[1]))):
            pos = mismatches[1][idx].item()
            d_tok = draft_top1[0, pos].item()
            t_tok = target_top1[0, pos].item()
            print(f"    pos={pos}: draft='{tokenizer.decode([d_tok])}' vs target='{tokenizer.decode([t_tok])}'")

    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
