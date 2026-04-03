"""
Comprehensive Evaluation for Matryoshka Speculative Decoding

Benchmarks:
- MT-Bench-style prompts (diverse instruction following)
- GSM8K (math reasoning)
- HumanEval-style (code generation)
- CNN/DailyMail-style (summarization)

Metrics:
- Mean acceptance length (α)
- Token acceptance rate
- Speedup ratio vs autoregressive baseline
- Tokens per second
- Quality comparison (output match rate for greedy)

Usage:
    python eval_speculative.py \
        --base_model_id meta-llama/Llama-3.1-8B-Instruct \
        --draft_model_path outputs/step1_draft_0.1bit/<timestamp> \
        --residual_model_path outputs/step2_residual_0.9bit/<timestamp> \
        --benchmark all \
        --max_samples 50 \
        --draft_lengths 1,3,5,7
"""

import argparse
import json
import os
import time
from typing import Optional, List, Dict
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from speculative_decoding import (
    MatryoshkaDraftModel, MatryoshkaTargetModel, FPTargetModel,
    speculative_decode, speculative_decode_tree,
    autoregressive_generate, str2bool,
    load_target_model, load_draft_model,
)
from tree_utils import generate_tree_buffers, TREE_PRESETS
from utils.datautils import load_tokenizer
from utils.misc import setup_logger

logger = setup_logger(__name__)


# ==============================================================================
# Benchmark Datasets
# ==============================================================================

def load_mt_bench_prompts(max_samples=50):
    """Load MT-Bench-style diverse instruction prompts."""
    prompts = [
        "Explain the concept of quantum entanglement in simple terms.",
        "Write a Python function to find the longest common subsequence of two strings.",
        "Summarize the key differences between TCP and UDP protocols.",
        "Create a haiku about machine learning.",
        "What are the pros and cons of microservices architecture?",
        "Write a SQL query to find the top 5 customers by total spending.",
        "Explain how a transformer neural network processes text input step by step.",
        "Design a REST API for a simple todo application. Include endpoints and data models.",
        "What is the time complexity of quicksort and why?",
        "Write a bash script that finds all Python files modified in the last 24 hours.",
        "Explain the CAP theorem and its implications for distributed databases.",
        "Write a recursive function to generate all permutations of a string.",
        "What are the SOLID principles in software engineering? Give examples.",
        "Create a simple neural network from scratch using only NumPy.",
        "Explain the difference between symmetric and asymmetric encryption.",
        "Write a Python decorator that caches function results.",
        "What is gradient descent and how does it work?",
        "Design a database schema for an e-commerce platform.",
        "Explain how garbage collection works in Java.",
        "Write a function to detect if a linked list has a cycle.",
        "What are the advantages of using Docker containers?",
        "Implement binary search in Python with error handling.",
        "Explain the concept of eventual consistency in distributed systems.",
        "Write a regular expression to validate email addresses.",
        "What is the difference between processes and threads?",
        "Create a simple HTTP server in Python.",
        "Explain how attention mechanisms work in transformers.",
        "Write a Python function to merge two sorted arrays.",
        "What are the key features of Kubernetes?",
        "Implement a LRU cache in Python.",
        "Explain the difference between supervised and unsupervised learning.",
        "Write a function to serialize and deserialize a binary tree.",
        "What is CI/CD and why is it important?",
        "Create a Python class for a min-heap data structure.",
        "Explain how TLS/SSL handshake works.",
        "Write a function to find all prime numbers up to N using the Sieve of Eratosthenes.",
        "What are design patterns? Describe the Observer pattern.",
        "Implement a simple rate limiter in Python.",
        "Explain the difference between REST and GraphQL.",
        "Write a Python script to parse CSV files and generate statistics.",
        "What is MapReduce and how does it work?",
        "Create a function to validate balanced parentheses.",
        "Explain the concept of database indexing and its trade-offs.",
        "Write a Python generator function for the Fibonacci sequence.",
        "What are the different types of database joins? Give examples.",
        "Implement a trie data structure for autocomplete functionality.",
        "Explain how HTTPS ensures secure communication.",
        "Write a function to find the shortest path in a graph using BFS.",
        "What is the difference between stack and queue data structures?",
        "Create a simple command-line calculator in Python.",
    ]
    return prompts[:max_samples]


def load_gsm8k_prompts(max_samples=50):
    """Load GSM8K math reasoning prompts."""
    try:
        dataset = load_dataset("gsm8k", "main", split="test")
        prompts = [item["question"] for item in dataset]
        return prompts[:max_samples]
    except Exception as e:
        logger.warning(f"Failed to load GSM8K: {e}. Using fallback prompts.")
        fallback = [
            "A store sold 120 apples on Monday and twice as many on Tuesday. How many apples were sold in total?",
            "If a train travels at 60 mph for 2.5 hours, how far does it travel?",
            "Sarah has 3 boxes with 12 cookies each. She gives away 1/4 of all cookies. How many does she have left?",
            "A rectangle has a perimeter of 30cm. If the length is twice the width, what are the dimensions?",
            "John earns $15/hour and works 8 hours a day, 5 days a week. What is his monthly income?",
        ]
        return fallback[:max_samples]


def load_humaneval_prompts(max_samples=50):
    """Load HumanEval-style code generation prompts."""
    try:
        dataset = load_dataset("openai_humaneval", split="test")
        prompts = [item["prompt"] for item in dataset]
        return prompts[:max_samples]
    except Exception as e:
        logger.warning(f"Failed to load HumanEval: {e}. Using fallback prompts.")
        fallback = [
            "Write a Python function that takes a list of integers and returns the second largest element.",
            "Implement a function to check if a string is a valid palindrome, ignoring spaces and punctuation.",
            "Write a function that converts a Roman numeral string to an integer.",
            "Implement a function to find the longest palindromic substring.",
            "Write a function that groups anagrams together from a list of strings.",
        ]
        return prompts[:max_samples] if 'prompts' in dir() else fallback[:max_samples]


def load_summarization_prompts(max_samples=50):
    """Load CNN/DailyMail summarization prompts."""
    try:
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
        prompts = [f"Summarize the following article:\n\n{item['article'][:1000]}" for item in dataset]
        return prompts[:max_samples]
    except Exception as e:
        logger.warning(f"Failed to load CNN/DailyMail: {e}. Using fallback prompts.")
        fallback = [
            "Summarize the following: Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. These systems can make predictions or decisions without being explicitly programmed.",
            "Summarize the following: The Internet of Things (IoT) refers to the network of physical devices embedded with sensors, software, and connectivity. IoT enables objects to collect and exchange data, transforming industries.",
            "Summarize the following: Climate change is the long-term alteration of temperature and typical weather patterns. The main cause is the burning of fossil fuels, which releases greenhouse gases.",
        ]
        return fallback[:max_samples]


BENCHMARK_LOADERS = {
    "mt_bench": load_mt_bench_prompts,
    "gsm8k": load_gsm8k_prompts,
    "humaneval": load_humaneval_prompts,
    "summarization": load_summarization_prompts,
}


# ==============================================================================
# Evaluation Engine
# ==============================================================================

def evaluate_benchmark(
    draft_model: MatryoshkaDraftModel,
    target_model,  # FPTargetModel or MatryoshkaTargetModel
    tokenizer,
    prompts: List[str],
    benchmark_name: str,
    draft_lengths: List[int] = [1, 3, 5, 7],
    max_new_tokens: int = 128,
    greedy: bool = True,
    eos_token_id: int = 2,
    device: torch.device = torch.device("cuda"),
    decode_mode: str = "serial",
    tree_buffers: dict = None,
    top_k: int = 10,
    temperature: float = 1.0,
) -> Dict:
    """Run evaluation on a set of prompts with multiple draft lengths."""
    
    results = {
        "benchmark": benchmark_name,
        "num_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "mode": "greedy" if greedy else "sampling",
        "draft_lengths": {},
        "baseline": {},
    }
    
    # === Baseline: Autoregressive ===
    logger.info(f"[{benchmark_name}] Running autoregressive baseline...")
    baseline_stats_list = []
    baseline_outputs = []
    
    for idx, prompt in enumerate(prompts):
        try:
            chat_messages = [{"role": "user", "content": prompt}]
            try:
                prompt_text = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt_text = prompt
            
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
            
            output_ids, stats = autoregressive_generate(
                target_model=target_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                greedy=greedy,
                eos_token_id=eos_token_id,
            )
            
            baseline_stats_list.append(stats)
            output_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            baseline_outputs.append(output_text)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"  Baseline: {idx + 1}/{len(prompts)} done")
        except Exception as e:
            logger.warning(f"  Baseline prompt {idx} failed: {e}")
            baseline_stats_list.append({"tokens_per_second": 0, "elapsed_seconds": 0, "total_tokens_generated": 0})
            baseline_outputs.append("")
    
    # Aggregate baseline stats
    if baseline_stats_list:
        avg_tps = sum(s.get("tokens_per_second", 0) for s in baseline_stats_list) / len(baseline_stats_list)
        avg_time = sum(s.get("elapsed_seconds", 0) for s in baseline_stats_list) / len(baseline_stats_list)
        total_tokens = sum(s.get("total_tokens_generated", 0) for s in baseline_stats_list)
        results["baseline"] = {
            "avg_tokens_per_second": avg_tps,
            "avg_elapsed_seconds": avg_time,
            "total_tokens_generated": total_tokens,
        }
    
    # === Speculative Decoding with different K values ===
    for K in draft_lengths:
        logger.info(f"[{benchmark_name}] Running speculative decoding with K={K}...")
        spec_stats_list = []
        spec_outputs = []
        
        for idx, prompt in enumerate(prompts):
            try:
                chat_messages = [{"role": "user", "content": prompt}]
                try:
                    prompt_text = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    prompt_text = prompt
                
                inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", None)
                
                if decode_mode == "serial":
                    output_ids, stats = speculative_decode(
                        draft_model=draft_model,
                        target_model=target_model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        draft_length=K,
                        greedy=greedy,
                        eos_token_id=eos_token_id,
                    )
                else:  # tree
                    output_ids, stats = speculative_decode_tree(
                        draft_model=draft_model,
                        target_model=target_model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        tree_buffers=tree_buffers,
                        top_k=top_k,
                        temperature=temperature,
                        greedy=greedy,
                        eos_token_id=eos_token_id,
                    )
                
                spec_stats_list.append(stats)
                output_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                spec_outputs.append(output_text)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"  K={K}: {idx + 1}/{len(prompts)} done")
            except Exception as e:
                logger.warning(f"  K={K} prompt {idx} failed: {e}")
                spec_stats_list.append({
                    "tokens_per_second": 0, "elapsed_seconds": 0,
                    "total_tokens_generated": 0, "mean_acceptance_length": 0,
                    "acceptance_rate": 0, "total_accepted_tokens": 0,
                    "total_draft_tokens": 0, "num_steps": 0,
                })
                spec_outputs.append("")
        
        # Aggregate speculative stats
        if spec_stats_list:
            n = len(spec_stats_list)
            avg_acceptance_length = sum(s.get("mean_acceptance_length", 0) for s in spec_stats_list) / n
            avg_acceptance_rate = sum(s.get("acceptance_rate", 0) for s in spec_stats_list) / n
            avg_tps = sum(s.get("tokens_per_second", 0) for s in spec_stats_list) / n
            avg_time = sum(s.get("elapsed_seconds", 0) for s in spec_stats_list) / n
            total_accepted = sum(s.get("total_accepted_tokens", 0) for s in spec_stats_list)
            total_drafted = sum(s.get("total_draft_tokens", 0) for s in spec_stats_list)
            total_tokens = sum(s.get("total_tokens_generated", 0) for s in spec_stats_list)
            total_steps = sum(s.get("num_steps", 0) for s in spec_stats_list)
            
            # Compute output match rate (greedy only)
            match_count = 0
            if greedy and baseline_outputs:
                for spec_out, base_out in zip(spec_outputs, baseline_outputs):
                    if spec_out.strip() == base_out.strip():
                        match_count += 1
            
            speedup = (results["baseline"].get("avg_elapsed_seconds", 1) / max(avg_time, 1e-6)) if results["baseline"] else 0
            
            results["draft_lengths"][str(K)] = {
                "mean_acceptance_length": avg_acceptance_length,
                "global_acceptance_length": total_accepted / max(total_steps, 1),
                "acceptance_rate": avg_acceptance_rate,
                "global_acceptance_rate": total_accepted / max(total_drafted, 1),
                "avg_tokens_per_second": avg_tps,
                "avg_elapsed_seconds": avg_time,
                "total_tokens_generated": total_tokens,
                "total_accepted_tokens": total_accepted,
                "total_draft_tokens": total_drafted,
                "total_steps": total_steps,
                "speedup_vs_baseline": speedup,
                "output_match_rate": match_count / max(len(prompts), 1) if greedy else "N/A",
            }
    
    return results


def print_results(all_results: List[Dict]):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 100)
    print("MATRYOSHKA SPECULATIVE DECODING - EVALUATION RESULTS")
    print("=" * 100)
    
    for result in all_results:
        print(f"\n{'─' * 80}")
        print(f"Benchmark: {result['benchmark'].upper()}")
        print(f"Prompts: {result['num_prompts']} | Max tokens: {result['max_new_tokens']} | Mode: {result['mode']}")
        print(f"{'─' * 80}")
        
        # Baseline
        base = result.get("baseline", {})
        if base:
            print(f"\n  Baseline (Autoregressive):")
            print(f"    Tokens/sec: {base.get('avg_tokens_per_second', 0):.2f}")
            print(f"    Avg time:   {base.get('avg_elapsed_seconds', 0):.3f}s")
        
        # Speculative with different K
        print(f"\n  {'K':>3s} | {'Acc Length':>10s} | {'Acc Rate':>9s} | {'Tokens/s':>9s} | {'Speedup':>8s} | {'Match%':>7s}")
        print(f"  {'─'*3}-+-{'─'*10}-+-{'─'*9}-+-{'─'*9}-+-{'─'*8}-+-{'─'*7}")
        
        for k_str, stats in sorted(result.get("draft_lengths", {}).items(), key=lambda x: int(x[0])):
            acc_len = stats.get("global_acceptance_length", stats.get("mean_acceptance_length", 0))
            acc_rate = stats.get("global_acceptance_rate", stats.get("acceptance_rate", 0))
            tps = stats.get("avg_tokens_per_second", 0)
            speedup = stats.get("speedup_vs_baseline", 0)
            match = stats.get("output_match_rate", "N/A")
            
            match_str = f"{match*100:.1f}%" if isinstance(match, (int, float)) else match
            
            print(f"  {k_str:>3s} | {acc_len:>10.3f} | {acc_rate:>8.1%} | {tps:>9.2f} | {speedup:>7.2f}x | {match_str:>7s}")
    
    print(f"\n{'='*100}\n")


def save_results(all_results: List[Dict], output_path: str):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Matryoshka Speculative Decoding")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model ID (for tokenizer, and FP target if target_mode=fp)")
    parser.add_argument("--draft_model_path", type=str, required=True,
                        help="Path to 0.1-bit draft model")
    parser.add_argument("--residual_model_path", type=str, default=None,
                        help="Path to 0.9-bit residual model (required for target_mode=matryoshka)")
    parser.add_argument("--target_mode", type=str, default="fp",
                        choices=["fp", "matryoshka", "littlebit_cpu"],
                        help="Target model type: 'fp'=original FP, 'matryoshka'=0.1+0.9 combined, 'littlebit_cpu'=CPU kernel")
    parser.add_argument("--draft_runtime_path", type=str, default=None,
                        help="Path to draft model runtime checkpoint (for littlebit_cpu target)")
    parser.add_argument("--residual_runtime_path", type=str, default=None,
                        help="Path to residual model runtime checkpoint (for littlebit_cpu target)")
    parser.add_argument("--decode_mode", type=str, default="serial",
                        choices=["serial", "tree"],
                        help="Decode mode: 'serial'=sequential draft, 'tree'=EAGLE tree attention")
    parser.add_argument("--tree_preset", type=str, default="default",
                        choices=["small", "default", "large"],
                        help="Tree structure preset (for tree mode)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-K for tree draft expansion (tree mode only)")
    parser.add_argument("--benchmark", type=str, default="all",
                        choices=["all", "mt_bench", "gsm8k", "humaneval", "summarization"],
                        help="Which benchmark to run")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Maximum samples per benchmark")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--draft_lengths", type=str, default="1,3,5,7",
                        help="Comma-separated draft lengths to evaluate")
    parser.add_argument("--mode", type=str, default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--output_file", type=str, default="eval_results/speculative_eval.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--draft_device", type=str, default="cuda",
                        choices=["cuda", "cpu_kernel"],
                        help="Device for draft model: 'cuda'=GPU PyTorch, 'cpu_kernel'=CPU LittleBit kernel")
    
    args = parser.parse_args()
    
    draft_lengths = [int(x.strip()) for x in args.draft_lengths.split(",")]
    greedy = args.mode == "greedy"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    target_desc = (
        "FP (original)" if args.target_mode == "fp"
        else "CPU LittleBit (draft+residual kernel)" if args.target_mode == "littlebit_cpu"
        else "Matryoshka (0.1+0.9 bit)"
    )
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.base_model_id)
    eos_token_id = tokenizer.eos_token_id or 2
    
    # Load models
    logger.info("Loading draft model (0.1-bit)...")
    draft_model = load_draft_model(args, device)
    
    logger.info(f"Loading target model ({target_desc})...")
    target_model = load_target_model(args, device)
    
    # Prepare tree buffers if tree mode
    tree_buffers = None
    if args.decode_mode == "tree":
        tree_choices = TREE_PRESETS[args.tree_preset]
        tree_buffers = generate_tree_buffers(tree_choices, device=str(device))
        logger.info(f"Tree buffers: {tree_buffers['tree_len']} nodes")
    
    # Determine benchmarks to run
    if args.benchmark == "all":
        benchmarks = list(BENCHMARK_LOADERS.keys())
    else:
        benchmarks = [args.benchmark]
    
    all_results = []
    
    for bench_name in benchmarks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running benchmark: {bench_name} (decode_mode={args.decode_mode})")
        logger.info(f"{'='*60}")
        
        prompts = BENCHMARK_LOADERS[bench_name](max_samples=args.max_samples)
        
        result = evaluate_benchmark(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            prompts=prompts,
            benchmark_name=bench_name,
            draft_lengths=draft_lengths,
            max_new_tokens=args.max_new_tokens,
            greedy=greedy,
            eos_token_id=eos_token_id,
            device=device,
            decode_mode=args.decode_mode,
            tree_buffers=tree_buffers,
            top_k=args.top_k,
        )
        
        all_results.append(result)
    
    # Print and save results
    print_results(all_results)
    save_results(all_results, args.output_file)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
