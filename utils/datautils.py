# -----------------------------------------------------------------------------
# The following code is derived from a repository licensed under the MIT License.
#
# Copyright (c) 2024 Dongwon Jo
#
# Modified by Dongkyu Kim in 2026 (Samsung Electronics Co., Ltd.)
#   - Added prepare_dataset() with tokenizer hash-based caching
#   - Added load_tokenizer() with Fast/Slow tokenizer fallback
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

import os
import random
import re
import hashlib
from itertools import chain

import datasets
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer

from utils.misc import setup_logger

logger = setup_logger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_dataset(args, tokenizer):
    try:
        if tokenizer.is_fast:
            hash_tokenizer = AutoTokenizer.from_pretrained(tokenizer.name_or_path, use_fast=False,
                                                           trust_remote_code=True)
            text = hash_tokenizer.__repr__()
        else:
            text = tokenizer.__repr__()

    except Exception as e:
        logger.warning(f"Failed to load Slow tokenizer for hashing ({e}). Using current repr with regex hack.")
        text = tokenizer.__repr__()
        text = re.sub(r"TokenizerFast", "Tokenizer", text)
        text = re.sub(r"use_fast=True", "use_fast=False", text)

    hash_key = re.sub(r"name_or_path=[^,]+,?\s*", "", text)

    hash_value = hashlib.sha256(hash_key.encode()).hexdigest()[:7]
    dataset = os.path.join(args.data_root, args.dataset, hash_value)

    logger.info(f"Attempting to load dataset from disk at '{dataset}'")
    need_regenerate = False
    try:
        datasets = load_from_disk(dataset)
        if len(datasets) == 0:
            logger.warning(f"Cached dataset at '{dataset}' has 0 samples! Will regenerate.")
            need_regenerate = True
        else:
            logger.info(f"Successfully loaded dataset from disk at '{dataset}' ({len(datasets)} samples)")
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Failed to load dataset from disk at '{dataset}': {e}")
        need_regenerate = True

    if need_regenerate:
        logger.info("Generating new dataset using get_qat_dataset")
        num_samples = getattr(args, 'num_samples', 50000)
        datasets = get_qat_dataset(args.dataset, tokenizer, sharegpt_path=getattr(args, 'sharegpt_path', None), data_root=args.data_root, num_samples=num_samples)
        if len(datasets) == 0:
            raise ValueError(
                f"Generated dataset for '{args.dataset}' has 0 samples. "
                f"Check your data source and tokenizer configuration."
            )
        datasets.save_to_disk(dataset)
        logger.info(f"Dataset saved to disk at '{dataset}'")
        with open(os.path.join(dataset, "tokenizer_info"), "w") as f:
            f.write(hash_key)
    return datasets


def load_tokenizer(model_id):
    try:
        print(f"Loading Fast Tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    except (OSError, TypeError, ValueError):
        print(f"Fast Tokenizer not found. Falling back to Slow Tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_qat_dataset(name, tokenizer, sharegpt_path=None, data_root="./", num_samples=50000):
    if name == "wikitext2":
        data = get_wikitext2_train(tokenizer=tokenizer)
    elif name == "c4":
        data = get_c4_train(tokenizer=tokenizer)
    elif name == "c4_wiki":
        data = get_c4_wiki_train(tokenizer=tokenizer)
    elif name == "wikitext2_sharegpt":
        data = get_wikitext2_sharegpt_train(tokenizer=tokenizer, sharegpt_path=sharegpt_path)
    elif name == "openhermes":
        data = get_openhermes_train(tokenizer=tokenizer, data_root=data_root, num_samples=num_samples)
    return data


def _extract_sharegpt_turns(item, output_list):
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


def get_wikitext2_sharegpt_train(tokenizer, sharegpt_path=None, seed=0, seqlen=2048):
    """Mixed wikitext2 + ShareGPT dataset for QAT training."""
    # --- Part 1: Wikitext2 ---
    wiki_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    wiki_text = "\n\n".join(wiki_dataset["text"])

    # --- Part 2: ShareGPT ---
    sharegpt_texts = []
    if sharegpt_path and os.path.exists(sharegpt_path):
        logger.info(f"Loading ShareGPT data from local path: {sharegpt_path}")

        # Detect file type: raw text (.txt) vs JSON/JSONL
        if sharegpt_path.endswith('.txt'):
            # Raw text file - read directly
            with open(sharegpt_path, "r", encoding="utf-8") as f:
                sharegpt_text_raw = f.read()
            logger.info(f"  Loaded raw text: {len(sharegpt_text_raw):,} characters")
            # Use directly instead of building from turns
            sharegpt_texts = None  # signal to use sharegpt_text_raw directly
        else:
            # JSON or JSONL file
            import json as _json
            with open(sharegpt_path, "r", encoding="utf-8") as f:
                first_char = f.read(1)
                f.seek(0)
                if first_char == '[':
                    data = _json.load(f)
                    for item in data:
                        _extract_sharegpt_turns(item, sharegpt_texts)
                else:
                    for line in f:
                        try:
                            item = _json.loads(line.strip())
                            _extract_sharegpt_turns(item, sharegpt_texts)
                        except _json.JSONDecodeError:
                            continue
            logger.info(f"  Loaded {len(sharegpt_texts)} conversation turns")
    else:
        logger.info("Downloading ShareGPT JSON directly from HuggingFace...")
        try:
            import json as _json
            from huggingface_hub import hf_hub_download
            sources = [
                ("anon8231489123/ShareGPT_Vicuna_unfiltered", [
                    "ShareGPT_V3_unfiltered_cleaned_split.json",
                ]),
                ("RyokoAI/ShareGPT52K", [
                    "old/sg_52k.json",
                ]),
            ]
            for repo_id, files in sources:
                if sharegpt_texts:
                    break
                for fname in files:
                    try:
                        local_file = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
                        with open(local_file, "r", encoding="utf-8") as f:
                            data = _json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                _extract_sharegpt_turns(item, sharegpt_texts)
                        logger.info(f"ShareGPT loaded from {repo_id}/{fname}: {len(sharegpt_texts)} turns")
                        break
                    except Exception as e:
                        logger.warning(f"Failed {repo_id}/{fname}: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Failed to load ShareGPT from HuggingFace: {e}. Using wikitext2 only.")

    # Build sharegpt_text
    if sharegpt_texts is None:
        # Raw text was loaded directly
        sharegpt_text = sharegpt_text_raw
    else:
        sharegpt_text = "\n\n".join(sharegpt_texts) if sharegpt_texts else ""

    # --- Combine ---
    combined_text = wiki_text + "\n\n" + sharegpt_text if sharegpt_text else wiki_text
    logger.info(f"Combined text: {len(combined_text):,} characters")

    # Split into manageable chunks (~1MB each) to avoid tokenization bottleneck
    CHUNK_SIZE = 1_000_000  # 1MB per chunk
    text_chunks = []
    for i in range(0, len(combined_text), CHUNK_SIZE):
        chunk = combined_text[i:i + CHUNK_SIZE]
        if chunk.strip():
            text_chunks.append(chunk)
    logger.info(f"Split into {len(text_chunks)} chunks for tokenization")

    combined_dataset = datasets.Dataset.from_dict({"text": text_chunks})

    column_names = list(combined_dataset.features)
    text_column_name = "text"

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = combined_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=4,
        desc="Tokenizing",
    )

    block_size = seqlen

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = tokenized_datasets.map(group_texts, batched=True, desc="Chunking")
    logger.info(f"wikitext2_sharegpt dataset prepared: {len(processed_dataset)} samples")
    return processed_dataset


def _convert_openhermes_to_chat_messages(conversations):
    """Convert OpenHermes conversation format to chat messages list.

    OpenHermes format: [{"from": "system", "value": "..."}, {"from": "human", "value": "..."}, ...]
    Output format:     [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
    """
    role_map = {
        "system": "system",
        "human": "user",
        "gpt": "assistant",
    }
    messages = []
    for turn in conversations:
        role = role_map.get(turn.get("from", ""), None)
        content = turn.get("value", "").strip()
        if role and content:
            messages.append({"role": role, "content": content})
    return messages


def get_openhermes_train(tokenizer, num_samples=50000, seed=42, seqlen=2048, data_root="./"):
    """OpenHermes 2.5 dataset with Llama 3.1 chat template applied.

    - First tries to load from pre-downloaded cache at {data_root}/data/openhermes_raw
    - Falls back to downloading teknium/OpenHermes-2.5 from HuggingFace
    - Randomly samples `num_samples` conversations
    - Applies tokenizer.apply_chat_template() for Llama 3.1 format
    - Truncates to `seqlen` tokens, filters out very short samples
    """
    # Try loading from pre-downloaded cache first (created by prepare_datasets.sh)
    local_cache = os.path.join(data_root, "data", "openhermes_raw")
    dataset = None
    if os.path.exists(local_cache):
        try:
            logger.info(f"Loading OpenHermes from local cache: {local_cache}")
            dataset = load_from_disk(local_cache)
            logger.info(f"  Loaded {len(dataset)} conversations from local cache")
        except Exception as e:
            logger.warning(f"  Failed to load local cache: {e}")
            dataset = None

    if dataset is None:
        logger.info(f"Loading OpenHermes 2.5 dataset from HuggingFace (sampling {num_samples} from ~1M)...")
        dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
        logger.info(f"  Full dataset size: {len(dataset)}")
        # Random sampling
        dataset = dataset.shuffle(seed=seed).select(range(min(num_samples, len(dataset))))
        logger.info(f"  Sampled {len(dataset)} conversations")
    else:
        # Local cache may already be sampled; if larger than num_samples, re-sample
        if len(dataset) > num_samples:
            dataset = dataset.shuffle(seed=seed).select(range(num_samples))
            logger.info(f"  Re-sampled to {len(dataset)} conversations")

    # Convert each conversation using chat template
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    skipped = 0
    min_length = 32  # Skip conversations shorter than this

    for i, example in enumerate(dataset):
        conversations = example.get("conversations", [])
        if not conversations:
            skipped += 1
            continue

        messages = _convert_openhermes_to_chat_messages(conversations)
        if len(messages) < 2:  # Need at least user + assistant
            skipped += 1
            continue

        # Use tokenizer's built-in chat template (Llama 3.1 format)
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors=None,  # Return plain list
            )
        except Exception as e:
            if i < 3:
                logger.warning(f"  Failed to apply chat template to sample {i}: {e}")
            skipped += 1
            continue

        # Filter too short
        if len(input_ids) < min_length:
            skipped += 1
            continue

        # Truncate to seqlen
        input_ids = input_ids[:seqlen]
        attention_mask = [1] * len(input_ids)
        labels = input_ids.copy()

        # Pad to seqlen
        pad_len = seqlen - len(input_ids)
        if pad_len > 0:
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            input_ids = input_ids + [pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

        if (i + 1) % 10000 == 0:
            logger.info(f"  Processed {i + 1}/{len(dataset)} conversations...")

    logger.info(f"  Skipped {skipped} conversations (empty/short/error)")
    logger.info(f"  Final dataset: {len(all_input_ids)} samples, seqlen={seqlen}")

    processed_dataset = datasets.Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    })

    return processed_dataset


def get_eval_loaders(name, tokenizer):
    if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
        try:
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
            print(f"bos/eos tokens updated: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
        except AttributeError:
            pass
            print(f"bos/eos tokens unchanged: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(tokenizer)
        return get_ptb(tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(tokenizer)
        return get_c4(tokenizer)


def get_wikitext2_train(tokenizer, seed=0, seqlen=2048):
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
    )

    wikitext_dataset = datasets.Dataset.from_dict(
        {
            "text": [
                # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                "\n\n".join(dataset["text"])
            ],
        }, )

    # Hacks to get around the `remove_columns` to be used later.
    wikitext_dataset = (
        wikitext_dataset  # type: ignore
        .add_column(name="timestamp", column=wikitext_dataset["text"]).add_column(name="url",
                                                                                  column=wikitext_dataset["text"]))
    column_names = list(wikitext_dataset.features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = wikitext_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

    block_size = 2048

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
    )
    return processed_dataset


def get_c4_train(tokenizer, seed=0, seqlen=2048):
    raw_datasets = load_dataset(
        "allenai/c4",
        #"allenai--c4",
        data_files={
            "train": "en/c4-train.00000-of-01024.json.gz",
            "validation": "en/c4-validation.00000-of-00008.json.gz",
        },
    )
    _wikitext_dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )
    # Hacks to be consistent with other works' preprocessing.
    wikitext_dataset = datasets.Dataset.from_dict(
        {
            "text": [
                # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                "\n\n".join(_wikitext_dataset["text"])
            ],
        }, )

    wikitext_dataset = (
        wikitext_dataset  # type: ignore
        .add_column(name="timestamp", column=wikitext_dataset["text"]).add_column(name="url",
                                                                                  column=wikitext_dataset["text"]))

    raw_datasets["wikitext"] = wikitext_dataset

    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

    block_size = 2048

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
    )
    return processed_dataset["train"]


def get_c4_wiki_train(tokenizer, seed=0, seqlen=2048):
    raw_datasets = load_dataset(
        "allenai/c4",
        #"allenai--c4",
        data_files={
            "train": "en/c4-train.00000-of-01024.json.gz",
            "validation": "en/c4-validation.00000-of-00008.json.gz",
        },
    )
    _wikitext_dataset_train = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
    )
    _wikitext_dataset_eval = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )
    # Hacks to be consistent with other works' preprocessing.
    wikitext_dataset_train = datasets.Dataset.from_dict(
        {
            "text": [
                # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                "\n\n".join(_wikitext_dataset_train["text"])
            ],
        }, )
    wikitext_dataset_eval = datasets.Dataset.from_dict(
        {
            "text": [
                # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                "\n\n".join(_wikitext_dataset_eval["text"])
            ],
        }, )

    wikitext_dataset_train = (
        wikitext_dataset_train  # type: ignore
        .add_column(name="timestamp", column=[None for _ in range(len(wikitext_dataset_train["text"]))
                                              ]).add_column(name="url", column=wikitext_dataset_train["text"]))
    wikitext_dataset_eval = (
        wikitext_dataset_eval  # type: ignore
        .add_column(name="timestamp",
                    column=wikitext_dataset_eval["text"]).add_column(name="url", column=wikitext_dataset_eval["text"]))

    raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], wikitext_dataset_train])
    raw_datasets["wikitext"] = wikitext_dataset_eval

    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

    block_size = 2048

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
    )
    return processed_dataset["train"]


def get_wikitext2(tokenizer, seqlen=2048):
    testdata = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )

    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    return testenc


def get_ptb(tokenizer, seqlen=2048):
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    return testenc


def get_c4(tokenizer, seqlen=2048):
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                           split='validation')
    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return valenc


def get_ptb_new(tokenizer, seqlen=2048):
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    return testenc


def get_c4_new(tokenizer, seqlen=2048):
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                           split='validation')

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return valenc
