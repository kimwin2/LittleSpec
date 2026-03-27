"""
Dataset Preparation Script for Matryoshka Speculative Decoding

Downloads and caches datasets for QAT training:
1. OpenHermes 2.5 (default) - teknium/OpenHermes-2.5 from HuggingFace
2. Wikitext-2 (legacy) - wikitext-2-raw-v1 from HuggingFace
3. ShareGPT (legacy) - from HuggingFace or local file

Usage:
    # Default: download OpenHermes 2.5 (recommended)
    python prepare_datasets.py --output_dir ./data

    # Legacy: wikitext2 + ShareGPT
    python prepare_datasets.py --output_dir ./data --mode legacy
    python prepare_datasets.py --output_dir ./data --mode legacy --sharegpt_path /path/to/sharegpt.jsonl
"""

import argparse
import json
import os

from datasets import load_dataset
from utils.misc import setup_logger

logger = setup_logger(__name__)


def download_openhermes(num_samples=50000, seed=42):
    """Download OpenHermes 2.5 dataset and return as a HuggingFace Dataset.

    Returns the raw dataset (no tokenization) so it can be cached to disk.
    Tokenization happens at training time via `get_openhermes_train()` in datautils.py.
    """
    logger.info(f"Downloading OpenHermes 2.5 from HuggingFace (sampling {num_samples})...")
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
    logger.info(f"  Full dataset size: {len(dataset)}")

    # Random sampling
    dataset = dataset.shuffle(seed=seed).select(range(min(num_samples, len(dataset))))
    logger.info(f"  Sampled {len(dataset)} conversations")

    return dataset


def download_wikitext2():
    """Download and return wikitext-2-raw-v1 train split as text."""
    logger.info("Downloading Wikitext-2 dataset from HuggingFace...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    logger.info(f"  Wikitext-2: {len(dataset)} rows, {len(text):,} characters")
    return text


def download_sharegpt(sharegpt_path=None):
    """Download/load ShareGPT data and return combined text."""
    sharegpt_texts = []

    if sharegpt_path and os.path.exists(sharegpt_path):
        logger.info(f"Loading ShareGPT from local file: {sharegpt_path}")
        with open(sharegpt_path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
                for item in data:
                    _extract_conversations(item, sharegpt_texts)
            else:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        _extract_conversations(item, sharegpt_texts)
                    except json.JSONDecodeError:
                        continue
        logger.info(f"  Loaded {len(sharegpt_texts)} conversation turns from local file")
    else:
        logger.info("Downloading ShareGPT JSON files directly from HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.error("huggingface_hub not installed. pip install huggingface_hub")
            return ""

        sources = [
            {
                "repo_id": "anon8231489123/ShareGPT_Vicuna_unfiltered",
                "files": [
                    "ShareGPT_V3_unfiltered_cleaned_split.json",
                ],
                "repo_type": "dataset",
            },
            {
                "repo_id": "RyokoAI/ShareGPT52K",
                "files": [
                    "old/sg_52k.json",
                ],
                "repo_type": "dataset",
            },
        ]

        for source in sources:
            if sharegpt_texts:
                break
            repo_id = source["repo_id"]
            logger.info(f"  Trying source: {repo_id}")
            for filename in source["files"]:
                try:
                    logger.info(f"    Downloading {filename}...")
                    local_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type=source["repo_type"],
                    )
                    logger.info(f"    Parsing {filename}...")
                    with open(local_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        for item in data:
                            _extract_conversations(item, sharegpt_texts)

                    logger.info(f"    Extracted {len(sharegpt_texts)} turns from {filename}")
                except Exception as e:
                    logger.warning(f"    Failed to load {filename}: {e}")
                    continue

        if not sharegpt_texts:
            logger.error("All ShareGPT sources failed.")
            logger.error("Please provide a local ShareGPT file via --sharegpt_path")
            return ""

    combined = "\n\n".join(sharegpt_texts)
    logger.info(f"  ShareGPT total: {len(sharegpt_texts)} turns, {len(combined):,} characters")
    return combined


def _extract_conversations(item, output_list):
    """Extract conversation text from a ShareGPT item (various formats)."""
    if not isinstance(item, dict):
        return
    if "conversations" in item and isinstance(item["conversations"], list):
        for turn in item["conversations"]:
            if isinstance(turn, dict):
                content = turn.get("value", turn.get("content", turn.get("text", "")))
                if content and isinstance(content, str) and len(content.strip()) > 0:
                    output_list.append(content.strip())
    elif "text" in item and isinstance(item["text"], str):
        if len(item["text"].strip()) > 0:
            output_list.append(item["text"].strip())


def main():
    parser = argparse.ArgumentParser(description="Download datasets for QAT training")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Directory to save dataset cache")
    parser.add_argument("--mode", type=str, default="openhermes",
                        choices=["openhermes", "legacy"],
                        help="Dataset mode: 'openhermes' (default) or 'legacy' (wikitext2+ShareGPT)")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of samples to use from OpenHermes 2.5 (default: 50000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--sharegpt_path", type=str, default=None,
                        help="(Legacy mode) Path to local ShareGPT .jsonl/.json file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "openhermes":
        # ============================
        # OpenHermes 2.5 (recommended)
        # ============================
        logger.info("\n" + "=" * 60)
        logger.info("Downloading OpenHermes 2.5 dataset")
        logger.info("=" * 60)

        dataset = download_openhermes(
            num_samples=args.num_samples,
            seed=args.seed,
        )

        # Save as HuggingFace dataset (Arrow format) for fast loading
        save_path = os.path.join(args.output_dir, "openhermes_raw")
        dataset.save_to_disk(save_path)
        logger.info(f"  Saved to: {save_path}")

        # Print stats
        num_convs = len(dataset)
        total_turns = sum(
            len(ex.get("conversations", [])) for ex in dataset
        )

        print("\n" + "=" * 60)
        print("DATASET PREPARATION COMPLETE")
        print("=" * 60)
        print(f"  Dataset:       OpenHermes 2.5")
        print(f"  Conversations: {num_convs:,}")
        print(f"  Total turns:   {total_turns:,}")
        print(f"  Saved to:      {save_path}")
        print()
        print("  Tokenization will happen automatically at training time")
        print("  using the model's chat template (e.g. Llama 3.1 format).")
        print()
        print("  To train with this data:")
        print("    --dataset openhermes")
        print("=" * 60)

    else:
        # ============================
        # Legacy: Wikitext-2 + ShareGPT
        # ============================
        logger.info("\n" + "=" * 60)
        logger.info("Step 1: Wikitext-2")
        logger.info("=" * 60)
        wiki_text = download_wikitext2()

        wiki_path = os.path.join(args.output_dir, "wikitext2_raw.txt")
        with open(wiki_path, "w", encoding="utf-8") as f:
            f.write(wiki_text)
        wiki_size_mb = os.path.getsize(wiki_path) / (1024 * 1024)
        logger.info(f"  Saved to: {wiki_path} ({wiki_size_mb:.1f} MB)")

        logger.info("\n" + "=" * 60)
        logger.info("Step 2: ShareGPT")
        logger.info("=" * 60)
        sharegpt_text = download_sharegpt(args.sharegpt_path)

        sharegpt_save_path = os.path.join(args.output_dir, "sharegpt_raw.txt")
        if sharegpt_text:
            with open(sharegpt_save_path, "w", encoding="utf-8") as f:
                f.write(sharegpt_text)
            sg_size_mb = os.path.getsize(sharegpt_save_path) / (1024 * 1024)
            logger.info(f"  Saved to: {sharegpt_save_path} ({sg_size_mb:.1f} MB)")
        else:
            logger.warning("  ShareGPT download failed - check logs above")

        logger.info("\n" + "=" * 60)
        logger.info("Step 3: Combined (wikitext2 + ShareGPT)")
        logger.info("=" * 60)

        combined_text = wiki_text
        if sharegpt_text:
            combined_text = wiki_text + "\n\n" + sharegpt_text

        combined_path = os.path.join(args.output_dir, "wikitext2_sharegpt_raw.txt")
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write(combined_text)
        combined_size_mb = os.path.getsize(combined_path) / (1024 * 1024)
        logger.info(f"  Saved to: {combined_path} ({combined_size_mb:.1f} MB)")

        print("\n" + "=" * 60)
        print("DATASET PREPARATION COMPLETE (legacy mode)")
        print("=" * 60)
        print(f"  wikitext2:          {wiki_path}  ({wiki_size_mb:.1f} MB)")
        if sharegpt_text:
            print(f"  sharegpt:           {sharegpt_save_path}  ({sg_size_mb:.1f} MB)")
        print(f"  combined:           {combined_path}  ({combined_size_mb:.1f} MB)")
        print()
        print("  To use this data with training scripts:")
        print(f"    --sharegpt_path {sharegpt_save_path}")
        print("    --dataset wikitext2_sharegpt")
        print("=" * 60)


if __name__ == "__main__":
    main()
