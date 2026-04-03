"""
HumanEval Evaluation for LittleBit Models

Measures:
  1. pass@1 accuracy for FP16, draft (0.3-bit), target (draft+residual)
  2. Speculative decoding speedup (tokens/s, acceptance rate, speedup ratio)

Usage:
    # Accuracy only (fp16 + draft + target)
    python eval_humaneval.py \
        --base_model_id /path/to/Llama-3.1-8B-Instruct \
        --draft_model_path /path/to/step1_draft \
        --residual_model_path /path/to/step2_residual \
        --eval_fp --eval_draft --eval_target

    # Speculative decoding speedup
    python eval_humaneval.py \
        --base_model_id /path/to/Llama-3.1-8B-Instruct \
        --draft_model_path /path/to/step1_draft \
        --residual_model_path /path/to/step2_residual \
        --eval_speculative --draft_lengths 1,3,5,7
"""

import argparse
import json
import os
import signal
import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization.utils.quant_util import load_quantized_model
from utils.datautils import load_tokenizer
from utils.misc import setup_logger

logger = setup_logger(__name__)


# ==============================================================================
# HumanEval Dataset
# ==============================================================================

def load_humaneval_dataset():
    """Load HumanEval dataset from HuggingFace."""
    try:
        dataset = load_dataset("openai_humaneval", split="test")
        problems = []
        for item in dataset:
            problems.append({
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"],
            })
        logger.info(f"Loaded {len(problems)} HumanEval problems")
        return problems
    except Exception as e:
        logger.error(f"Failed to load HumanEval dataset: {e}")
        raise


# ==============================================================================
# Code Execution (sandboxed with timeout)
# ==============================================================================

class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager for execution timeout (Unix only)."""
    if sys.platform == "win32":
        yield
        return

    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def check_correctness(problem: dict, completion: str, timeout: int = 5) -> dict:
    """Execute generated code against test cases.

    Args:
        problem: HumanEval problem dict with 'prompt', 'test', 'entry_point'.
        completion: Generated function body.
        timeout: Max execution time in seconds.

    Returns:
        dict with 'passed' (bool) and 'result' (str).
    """
    # Build executable code: prompt + completion + test harness
    full_code = problem["prompt"] + completion + "\n" + problem["test"] + f"\ncheck({problem['entry_point']})"

    try:
        exec_globals = {}
        with time_limit(timeout):
            exec(full_code, exec_globals)
        return {"passed": True, "result": "passed"}
    except TimeoutError:
        return {"passed": False, "result": "timeout"}
    except Exception as e:
        return {"passed": False, "result": str(e)[:200]}


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model_from_checkpoint(model_path, torch_dtype=torch.bfloat16, device="cuda"):
    """Load a LittleBit quantized model from checkpoint."""
    config_path = os.path.join(model_path, "littlebit_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    quant_args = argparse.Namespace(
        quant_func=config.get("quant_func", "STEBinary"),
        quant_mod=config.get("quant_mod", "LittleBitLinear"),
        eff_bit=config.get("eff_bit", 0.1),
        split_dim=config.get("split_dim", 1024),
        residual=config.get("residual", False),
        kv_factor=config.get("kv_factor", 1.0),
        min_split_dim=config.get("min_split_dim", 8),
        model_id=model_path,
    )

    model = load_quantized_model(
        model_path=model_path,
        quant_args=quant_args,
        torch_dtype=torch_dtype,
        device=device,
    )
    model.eval()
    return model


# ==============================================================================
# Code Generation
# ==============================================================================

@torch.no_grad()
def generate_code(model, tokenizer, prompt, max_new_tokens=512,
                  temperature=0.0, device="cuda"):
    """Generate code completion for a HumanEval prompt.

    Uses stop tokens to cut generation at function boundary.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(device)
    input_ids = inputs["input_ids"]

    eos_token_id = tokenizer.eos_token_id or 2
    stop_token_ids = {eos_token_id}

    # Add common stop tokens
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        for stop_tok in ['<|eot_id|>', '<|end_of_text|>', '</s>']:
            tid = tokenizer.convert_tokens_to_ids(stop_tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                stop_token_ids.add(tid)

    generated_tokens = []
    current_ids = input_ids

    for _ in range(max_new_tokens):
        outputs = model(current_ids, use_cache=False)
        next_logits = outputs.logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(next_logits, dim=-1).item()
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        generated_tokens.append(next_token)

        if next_token in stop_token_ids:
            break

        next_token_tensor = torch.tensor([[next_token]], device=device)
        current_ids = torch.cat([current_ids, next_token_tensor], dim=1)

    completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Post-process: stop at next function definition or class definition
    stop_sequences = ["\ndef ", "\nclass ", "\n#", "\nif __name__"]
    for stop_seq in stop_sequences:
        idx = completion.find(stop_seq)
        if idx > 0:
            completion = completion[:idx]

    return completion


@torch.no_grad()
def generate_code_combined(draft_model, residual_model, tokenizer, prompt,
                           max_new_tokens=512, temperature=0.0, device="cuda"):
    """Generate code using combined draft+residual (target) logits."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(device)
    input_ids = inputs["input_ids"]

    eos_token_id = tokenizer.eos_token_id or 2
    stop_token_ids = {eos_token_id}
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        for stop_tok in ['<|eot_id|>', '<|end_of_text|>', '</s>']:
            tid = tokenizer.convert_tokens_to_ids(stop_tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                stop_token_ids.add(tid)

    generated_tokens = []
    current_ids = input_ids

    for _ in range(max_new_tokens):
        draft_out = draft_model(current_ids, use_cache=False)
        residual_out = residual_model(current_ids, use_cache=False)
        combined_logits = draft_out.logits + residual_out.logits
        next_logits = combined_logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(next_logits, dim=-1).item()
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        generated_tokens.append(next_token)

        if next_token in stop_token_ids:
            break

        next_token_tensor = torch.tensor([[next_token]], device=device)
        current_ids = torch.cat([current_ids, next_token_tensor], dim=1)

    completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    stop_sequences = ["\ndef ", "\nclass ", "\n#", "\nif __name__"]
    for stop_seq in stop_sequences:
        idx = completion.find(stop_seq)
        if idx > 0:
            completion = completion[:idx]

    return completion


# ==============================================================================
# Accuracy Evaluation (pass@1)
# ==============================================================================

def evaluate_accuracy(model_name, generate_fn, tokenizer, problems,
                      max_new_tokens=512, temperature=0.0, device="cuda",
                      timeout=5):
    """Evaluate pass@1 on HumanEval problems.

    Args:
        model_name: Name for logging.
        generate_fn: Function(tokenizer, prompt, max_new_tokens, temperature, device) -> completion.
        tokenizer: Tokenizer.
        problems: List of HumanEval problem dicts.
        max_new_tokens: Max generation length.
        temperature: 0.0 for greedy.
        device: torch device string.
        timeout: Execution timeout per problem.

    Returns:
        dict with pass@1, per-problem results.
    """
    passed = 0
    total = len(problems)
    results = []

    for idx, problem in enumerate(tqdm(problems, desc=f"pass@1({model_name})")):
        start = time.time()
        completion = generate_fn(
            tokenizer=tokenizer,
            prompt=problem["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        gen_time = time.time() - start

        check = check_correctness(problem, completion, timeout=timeout)

        if check["passed"]:
            passed += 1

        results.append({
            "task_id": problem["task_id"],
            "passed": check["passed"],
            "result": check["result"],
            "completion_len": len(completion),
            "gen_time": gen_time,
        })

        if (idx + 1) % 20 == 0:
            logger.info(f"  [{model_name}] {idx+1}/{total}: "
                        f"pass@1 = {passed}/{idx+1} ({passed/(idx+1)*100:.1f}%)")

    pass_at_1 = passed / total if total > 0 else 0
    logger.info(f"  [{model_name}] Final pass@1 = {passed}/{total} ({pass_at_1*100:.1f}%)")

    return {
        "model": model_name,
        "pass_at_1": pass_at_1,
        "passed": passed,
        "total": total,
        "avg_gen_time": sum(r["gen_time"] for r in results) / total if total > 0 else 0,
        "per_problem": results,
    }


# ==============================================================================
# Speculative Decoding Evaluation
# ==============================================================================

def evaluate_speculative(draft_model, target_model, tokenizer, problems,
                         draft_lengths=[1, 3, 5, 7], max_new_tokens=512,
                         device="cuda", timeout=5):
    """Evaluate speculative decoding on HumanEval.

    Measures: tokens/s, acceptance rate, speedup, and pass@1.
    """
    from speculative_decoding import (
        MatryoshkaDraftModel, speculative_decode,
        autoregressive_generate,
    )

    eos_token_id = tokenizer.eos_token_id or 2

    results = {
        "benchmark": "humaneval",
        "num_problems": len(problems),
        "baseline": {},
        "draft_lengths": {},
    }

    # --- Baseline: autoregressive with target model ---
    logger.info("[Speculative] Running autoregressive baseline...")
    baseline_tokens_total = 0
    baseline_time_total = 0
    baseline_completions = []

    for idx, problem in enumerate(tqdm(problems, desc="Baseline")):
        inputs = tokenizer(problem["prompt"], return_tensors="pt",
                           truncation=True, max_length=2048).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        try:
            output_ids, stats = autoregressive_generate(
                target_model=target_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                greedy=True,
                eos_token_id=eos_token_id,
            )
            gen_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:],
                                        skip_special_tokens=True)
            # Post-process
            for stop_seq in ["\ndef ", "\nclass ", "\n#", "\nif __name__"]:
                stop_idx = gen_text.find(stop_seq)
                if stop_idx > 0:
                    gen_text = gen_text[:stop_idx]

            baseline_completions.append(gen_text)
            baseline_tokens_total += stats.get("total_tokens_generated", 0)
            baseline_time_total += stats.get("elapsed_seconds", 0)
        except Exception as e:
            logger.warning(f"  Baseline problem {idx} failed: {e}")
            baseline_completions.append("")

    # Baseline pass@1
    baseline_passed = 0
    for prob, comp in zip(problems, baseline_completions):
        chk = check_correctness(prob, comp, timeout=timeout)
        if chk["passed"]:
            baseline_passed += 1

    n = len(problems)
    avg_baseline_time = baseline_time_total / n if n > 0 else 0
    results["baseline"] = {
        "avg_tokens_per_second": baseline_tokens_total / baseline_time_total if baseline_time_total > 0 else 0,
        "avg_elapsed_seconds": avg_baseline_time,
        "total_tokens": baseline_tokens_total,
        "pass_at_1": baseline_passed / n if n > 0 else 0,
        "passed": baseline_passed,
    }

    # --- Speculative decoding with different K values ---
    for K in draft_lengths:
        logger.info(f"[Speculative] Running K={K}...")
        spec_completions = []
        spec_stats_list = []

        for idx, problem in enumerate(tqdm(problems, desc=f"K={K}")):
            inputs = tokenizer(problem["prompt"], return_tensors="pt",
                               truncation=True, max_length=2048).to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)

            try:
                output_ids, stats = speculative_decode(
                    draft_model=draft_model,
                    target_model=target_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    draft_length=K,
                    greedy=True,
                    eos_token_id=eos_token_id,
                )
                gen_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:],
                                            skip_special_tokens=True)
                for stop_seq in ["\ndef ", "\nclass ", "\n#", "\nif __name__"]:
                    stop_idx = gen_text.find(stop_seq)
                    if stop_idx > 0:
                        gen_text = gen_text[:stop_idx]

                spec_completions.append(gen_text)
                spec_stats_list.append(stats)
            except Exception as e:
                logger.warning(f"  K={K} problem {idx} failed: {e}")
                spec_completions.append("")
                spec_stats_list.append({})

        # Compute pass@1
        spec_passed = 0
        for prob, comp in zip(problems, spec_completions):
            chk = check_correctness(prob, comp, timeout=timeout)
            if chk["passed"]:
                spec_passed += 1

        # Aggregate stats
        valid_stats = [s for s in spec_stats_list if s]
        ns = len(valid_stats) if valid_stats else 1
        avg_acc_len = sum(s.get("mean_acceptance_length", 0) for s in valid_stats) / ns
        avg_acc_rate = sum(s.get("acceptance_rate", 0) for s in valid_stats) / ns
        avg_tps = sum(s.get("tokens_per_second", 0) for s in valid_stats) / ns
        avg_time = sum(s.get("elapsed_seconds", 0) for s in valid_stats) / ns
        total_accepted = sum(s.get("total_accepted_tokens", 0) for s in valid_stats)
        total_drafted = sum(s.get("total_draft_tokens", 0) for s in valid_stats)
        total_steps = sum(s.get("num_steps", 0) for s in valid_stats)

        speedup = avg_baseline_time / max(avg_time, 1e-6) if avg_baseline_time > 0 else 0

        results["draft_lengths"][str(K)] = {
            "mean_acceptance_length": avg_acc_len,
            "global_acceptance_length": total_accepted / max(total_steps, 1),
            "acceptance_rate": avg_acc_rate,
            "global_acceptance_rate": total_accepted / max(total_drafted, 1),
            "avg_tokens_per_second": avg_tps,
            "avg_elapsed_seconds": avg_time,
            "speedup_vs_baseline": speedup,
            "pass_at_1": spec_passed / n if n > 0 else 0,
            "passed": spec_passed,
        }

    return results


# ==============================================================================
# Results Display
# ==============================================================================

def print_accuracy_results(all_results: List[dict]):
    """Print formatted accuracy comparison table."""
    print("\n" + "=" * 70)
    print("HUMANEVAL PASS@1 ACCURACY")
    print("=" * 70)
    print(f"  {'Model':<30s} {'pass@1':>8s} {'Passed':>8s} {'Total':>6s} {'Avg Time':>10s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*6} {'-'*10}")

    for r in all_results:
        print(f"  {r['model']:<30s} {r['pass_at_1']*100:>7.1f}% "
              f"{r['passed']:>8d} {r['total']:>6d} {r['avg_gen_time']:>9.2f}s")

    print("=" * 70 + "\n")


def print_speculative_results(results: dict):
    """Print formatted speculative decoding results."""
    print("\n" + "=" * 90)
    print("HUMANEVAL SPECULATIVE DECODING RESULTS")
    print("=" * 90)

    base = results.get("baseline", {})
    print(f"\n  Baseline (Autoregressive):")
    print(f"    Tokens/sec: {base.get('avg_tokens_per_second', 0):.2f}")
    print(f"    Avg time:   {base.get('avg_elapsed_seconds', 0):.3f}s")
    print(f"    pass@1:     {base.get('pass_at_1', 0)*100:.1f}%")

    print(f"\n  {'K':>3s} | {'Acc Len':>8s} | {'Acc Rate':>9s} | {'TPS':>9s} | "
          f"{'Speedup':>8s} | {'pass@1':>7s}")
    print(f"  {'─'*3}-+-{'─'*8}-+-{'─'*9}-+-{'─'*9}-+-{'─'*8}-+-{'─'*7}")

    for k_str, stats in sorted(results.get("draft_lengths", {}).items(),
                                key=lambda x: int(x[0])):
        acc_len = stats.get("global_acceptance_length",
                            stats.get("mean_acceptance_length", 0))
        acc_rate = stats.get("global_acceptance_rate",
                             stats.get("acceptance_rate", 0))
        tps = stats.get("avg_tokens_per_second", 0)
        speedup = stats.get("speedup_vs_baseline", 0)
        p1 = stats.get("pass_at_1", 0)

        print(f"  {k_str:>3s} | {acc_len:>8.3f} | {acc_rate:>8.1%} | "
              f"{tps:>9.2f} | {speedup:>7.2f}x | {p1*100:>6.1f}%")

    print("=" * 90 + "\n")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HumanEval Evaluation for LittleBit Models"
    )

    # Model paths
    parser.add_argument("--base_model_id", type=str, required=True,
                        help="Base model ID or path (for FP16 model and tokenizer)")
    parser.add_argument("--draft_model_path", type=str, default=None,
                        help="Path to draft model checkpoint")
    parser.add_argument("--residual_model_path", type=str, default=None,
                        help="Path to residual model checkpoint")

    # Evaluation modes
    parser.add_argument("--eval_fp", action="store_true",
                        help="Evaluate FP16 baseline model")
    parser.add_argument("--eval_draft", action="store_true",
                        help="Evaluate draft model only")
    parser.add_argument("--eval_target", action="store_true",
                        help="Evaluate target (draft+residual) model")
    parser.add_argument("--eval_speculative", action="store_true",
                        help="Evaluate speculative decoding speedup")

    # Generation settings
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 for greedy (deterministic)")
    parser.add_argument("--max_problems", type=int, default=164,
                        help="Max number of HumanEval problems (default: all 164)")
    parser.add_argument("--exec_timeout", type=int, default=5,
                        help="Code execution timeout in seconds")

    # Speculative decoding settings
    parser.add_argument("--draft_lengths", type=str, default="1,3,5,7",
                        help="Comma-separated draft lengths for speculative decoding")

    # Output
    parser.add_argument("--output_dir", type=str, default="eval_results/humaneval")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    problems = load_humaneval_dataset()[:args.max_problems]
    logger.info(f"Evaluating {len(problems)} HumanEval problems")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.base_model_id)

    all_accuracy_results = []

    # ====== 1. FP16 Baseline ======
    if args.eval_fp:
        logger.info("=" * 60)
        logger.info("Evaluating FP16 baseline model...")
        logger.info("=" * 60)

        fp_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        fp_model.eval()

        def fp_gen_fn(tokenizer, prompt, max_new_tokens, temperature, device):
            return generate_code(fp_model, tokenizer, prompt,
                                 max_new_tokens, temperature, device)

        fp_result = evaluate_accuracy(
            model_name="fp16_baseline",
            generate_fn=fp_gen_fn,
            tokenizer=tokenizer,
            problems=problems,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
            timeout=args.exec_timeout,
        )
        all_accuracy_results.append(fp_result)

        # Save
        with open(os.path.join(args.output_dir, "fp16_results.json"), "w") as f:
            json.dump(fp_result, f, indent=2, default=str)

        # Free memory
        if args.eval_draft or args.eval_target or args.eval_speculative:
            del fp_model
            torch.cuda.empty_cache()
            import gc; gc.collect()

    # ====== 2. Draft Model (0.3-bit) ======
    if args.eval_draft and args.draft_model_path:
        logger.info("=" * 60)
        logger.info("Evaluating DRAFT model...")
        logger.info("=" * 60)

        draft_model = load_model_from_checkpoint(args.draft_model_path, device=device)

        def draft_gen_fn(tokenizer, prompt, max_new_tokens, temperature, device):
            return generate_code(draft_model, tokenizer, prompt,
                                 max_new_tokens, temperature, device)

        draft_result = evaluate_accuracy(
            model_name="draft_model",
            generate_fn=draft_gen_fn,
            tokenizer=tokenizer,
            problems=problems,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
            timeout=args.exec_timeout,
        )
        all_accuracy_results.append(draft_result)

        with open(os.path.join(args.output_dir, "draft_results.json"), "w") as f:
            json.dump(draft_result, f, indent=2, default=str)

        if args.eval_target or args.eval_speculative:
            if not (args.eval_target and args.residual_model_path):
                del draft_model
                torch.cuda.empty_cache()
                import gc; gc.collect()

    # ====== 3. Target Model (draft + residual) ======
    if args.eval_target and args.draft_model_path and args.residual_model_path:
        logger.info("=" * 60)
        logger.info("Evaluating TARGET model (draft + residual)...")
        logger.info("=" * 60)

        # Load both models (reuse draft if already loaded)
        if 'draft_model' not in dir() or draft_model is None:
            draft_model = load_model_from_checkpoint(args.draft_model_path, device=device)
        residual_model = load_model_from_checkpoint(args.residual_model_path, device=device)

        def target_gen_fn(tokenizer, prompt, max_new_tokens, temperature, device):
            return generate_code_combined(draft_model, residual_model, tokenizer,
                                          prompt, max_new_tokens, temperature, device)

        target_result = evaluate_accuracy(
            model_name="target_model",
            generate_fn=target_gen_fn,
            tokenizer=tokenizer,
            problems=problems,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
            timeout=args.exec_timeout,
        )
        all_accuracy_results.append(target_result)

        with open(os.path.join(args.output_dir, "target_results.json"), "w") as f:
            json.dump(target_result, f, indent=2, default=str)

        if not args.eval_speculative:
            del draft_model, residual_model
            torch.cuda.empty_cache()
            import gc; gc.collect()

    # Print accuracy comparison
    if all_accuracy_results:
        print_accuracy_results(all_accuracy_results)

    # ====== 4. Speculative Decoding Speedup ======
    if args.eval_speculative and args.draft_model_path and args.residual_model_path:
        logger.info("=" * 60)
        logger.info("Evaluating SPECULATIVE DECODING speedup...")
        logger.info("=" * 60)

        from speculative_decoding import (
            MatryoshkaDraftModel, MatryoshkaTargetModel,
            load_target_model, load_draft_model,
        )

        # Create a minimal args namespace for speculative_decoding loaders
        spec_args = argparse.Namespace(
            base_model_id=args.base_model_id,
            draft_model_path=args.draft_model_path,
            residual_model_path=args.residual_model_path,
            target_mode="matryoshka",
            draft_device="cuda",
            draft_runtime_path=None,
            residual_runtime_path=None,
        )

        draft_lengths = [int(x.strip()) for x in args.draft_lengths.split(",")]

        sd_draft_model = load_draft_model(spec_args, torch.device(device))
        sd_target_model = load_target_model(spec_args, torch.device(device))

        spec_result = evaluate_speculative(
            draft_model=sd_draft_model,
            target_model=sd_target_model,
            tokenizer=tokenizer,
            problems=problems,
            draft_lengths=draft_lengths,
            max_new_tokens=args.max_new_tokens,
            device=torch.device(device),
            timeout=args.exec_timeout,
        )

        print_speculative_results(spec_result)

        with open(os.path.join(args.output_dir, "speculative_results.json"), "w") as f:
            json.dump(spec_result, f, indent=2, default=str)

    # Save combined summary
    summary = {
        "accuracy": [
            {"model": r["model"], "pass_at_1": r["pass_at_1"],
             "passed": r["passed"], "total": r["total"]}
            for r in all_accuracy_results
        ],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("HumanEval evaluation complete!")


if __name__ == "__main__":
    main()
