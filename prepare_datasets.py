"""
Dataset Preparation Script for Matryoshka Speculative Decoding

Downloads and saves RAW TEXT only (no tokenization):
1. Wikitext-2 (from HuggingFace)
2. ShareGPT (from HuggingFace, multiple sources)

Tokenization is handled at training time by datautils.py,
so changing the model later does NOT require re-downloading.

Usage:
    python prepare_datasets.py --output_dir ./data
    python prepare_datasets.py --sharegpt_path /path/to/sharegpt.jsonl --output_dir ./data
"""

import argparse
import json
import os

from datasets import load_dataset
from utils.misc import setup_logger

logger = setup_logger(__name__)


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
    parser = argparse.ArgumentParser(description="Download raw datasets for QAT training")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Directory to save raw text files")
    parser.add_argument("--sharegpt_path", type=str, default=None,
                        help="Path to local ShareGPT .jsonl/.json file (downloads from HF if not set)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ============================
    # 1. Wikitext-2
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

    # ============================
    # 2. ShareGPT
    # ============================
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

    # ============================
    # 3. Combined
    # ============================
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

    # ============================
    # Summary
    # ============================
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE (raw text only)")
    print("=" * 60)
    print(f"  wikitext2:          {wiki_path}  ({wiki_size_mb:.1f} MB)")
    if sharegpt_text:
        print(f"  sharegpt:           {sharegpt_save_path}  ({sg_size_mb:.1f} MB)")
    print(f"  combined:           {combined_path}  ({combined_size_mb:.1f} MB)")
    print()
    print("  Tokenization will happen automatically at training time.")
    print("  You can use any model - just change --model_id in the training script.")
    print()
    print("  To use this data with training scripts:")
    print(f"    --sharegpt_path {sharegpt_save_path}")
    print("    --dataset wikitext2_sharegpt")
    print("=" * 60)


if __name__ == "__main__":
    main()
