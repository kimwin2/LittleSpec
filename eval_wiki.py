import argparse
import json
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer

from quantization.utils.quant_util import load_quantized_model
from utils.datautils import get_eval_loaders


def str2bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception(f'Boolean value expected: {value}')


def _make_eval_lm_class(BaseLM):
    """Create the EvalLM class dynamically with the given BaseLM parent.

    This avoids a top-level dependency on lm_eval which may not be installed.
    """

    class EvalLM(BaseLM):
        def __init__(self, model, tokenizer, batch_size=1, accelerator=None):
            super().__init__()
            self.batch_size_per_gpu = batch_size
            self.seqlen = 2048
            self.tokenizer = tokenizer

            if accelerator is not None:
                self.accelerator = accelerator
                self._device = accelerator.device
                self.model = model
            else:
                self.accelerator = None
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = model.to(self._device)

            self.model.eval()
            self.vocab_size = self.tokenizer.vocab_size

        @property
        def eot_token_id(self):
            return self.tokenizer.eos_token_id

        @property
        def max_length(self):
            actual_model = self.model.module if hasattr(self.model, "module") else self.model
            return getattr(actual_model.config, "n_ctx", actual_model.config.max_position_embeddings)

        @property
        def max_gen_toks(self):
            return 256

        @property
        def batch_size(self):
            return self.batch_size_per_gpu

        def tok_encode(self, string: str):
            return self.tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens):
            return self.tokenizer.decode(tokens)

        def _model_call(self, inps):
            with torch.no_grad():
                outputs = self.model(inps, use_cache=False)
                if hasattr(outputs, "logits"):
                    return outputs.logits
                elif isinstance(outputs, dict) and "logits" in outputs:
                    return outputs["logits"]
                else:
                    return outputs

        def _model_generate(self, context, max_length, eos_token_id):
            with torch.no_grad():
                actual_model = self.model.module if hasattr(self.model, "module") else self.model
                return actual_model.generate(
                    context,
                    max_length=max_length,
                    eos_token_id=eos_token_id,
                    do_sample=False,
                )

        @property
        def device(self):
            # TODO: fix multi-gpu
            return self._device

    return EvalLM


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    tasks,
    eval_ppl="",
    num_fewshot=0,
    limit=-1,
    batch_size=1,
    accelerator=None,
):
    from lm_eval import evaluator
    from lm_eval.base import BaseLM

    EvalLM = _make_eval_lm_class(BaseLM)

    lm = EvalLM(model, tokenizer, batch_size=batch_size, accelerator=accelerator)
    results = {}
    if eval_ppl:
        datasets = eval_ppl.split(",")
        for dataset in datasets:
            msg = f"[INFO] Starting PPL eval for: {dataset}"
            if accelerator is not None:
                accelerator.print(msg)
            else:
                print(msg)

            testloader = get_eval_loaders(dataset, tokenizer)
            testenc = testloader.input_ids
            nsamples = testenc.numel() // lm.seqlen

            lm.model.eval()
            nlls = []
            for i in tqdm(range(nsamples), disable=(accelerator is None
                                                    or not getattr(accelerator, "is_local_main_process", True))):
                batch = testenc[:, (i * lm.seqlen):(i + 1) * lm.seqlen].to(lm.device, dtype=torch.long)
                outputs = lm.model(batch, use_cache=False)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

                shift_logits = logits[:, :-1, :]
                shift_labels = batch[:, 1:]

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            out_msg = f"[{dataset}] PPL = {ppl.item()}"
            if accelerator is not None:
                accelerator.print(out_msg)
            else:
                print(out_msg)

            results[dataset] = ppl.item()

    if tasks:
        harness_results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks.split(","),
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=None if limit == -1 else limit,
            no_cache=True,
        )
        harness_results = harness_results["results"]
        results.update(harness_results)
        msg = f"Zero-shot tasks results: {harness_results}"
        if accelerator is not None:
            accelerator.print(msg)
        else:
            print(msg)

    return results


def load_draft_model_for_eval(draft_model_path, base_model_id=None, device="auto"):
    """Load a draft model checkpoint using littlebit_config.json + load_quantized_model.

    Args:
        draft_model_path: Path to the draft model checkpoint directory.
        base_model_id: Base model ID for tokenizer. If None, reads from config.
        device: Target device.

    Returns:
        (model, tokenizer)
    """
    config_path = os.path.join(draft_model_path, "littlebit_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"littlebit_config.json not found in {draft_model_path}. "
            f"Is this a valid LittleBit checkpoint?"
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"[INFO] Draft model config: {json.dumps(config, indent=2)}")

    quant_args = argparse.Namespace(
        quant_func=config.get("quant_func", "STEBinary"),
        quant_mod=config.get("quant_mod", "LittleBitLinear"),
        eff_bit=config.get("eff_bit", 0.1),
        split_dim=config.get("split_dim", 1024),
        residual=config.get("residual", False),
        kv_factor=config.get("kv_factor", 1.0),
        min_split_dim=config.get("min_split_dim", 8),
        is_po2=config.get("is_po2", False),
        num_expert=config.get("num_expert", 4),
        model_id=draft_model_path,
    )

    model = load_quantized_model(
        model_path=draft_model_path,
        quant_args=quant_args,
        torch_dtype=torch.bfloat16,
        device=device,
    )
    model.eval()

    # Load tokenizer from base_model_id (stored in config) or explicit arg
    tokenizer_id = base_model_id or config.get("base_model_id") or draft_model_path
    print(f"[INFO] Loading tokenizer from: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, use_fast=False, trust_remote_code=True
    )

    return model, tokenizer


@torch.no_grad()
def eval_ppl_standalone(model, tokenizer, datasets="wikitext2", seqlen=None):
    """Standalone PPL evaluation — no lm_eval / BaseLM needed.

    Matches eval.py's evaluate_model PPL logic exactly.

    Args:
        model: HuggingFace CausalLM model (already on device).
        tokenizer: Tokenizer.
        datasets: Comma-separated dataset names (e.g. "wikitext2" or "wikitext2,c4").
        seqlen: Sequence length for evaluation chunks. If None, auto-detect from model config.
    """
    device = next(model.parameters()).device
    model.eval()

    # Auto-detect seqlen from model config (same as eval.py)
    if seqlen is None:
        actual_model = model.module if hasattr(model, "module") else model
        seqlen = getattr(actual_model.config, "n_ctx",
                         getattr(actual_model.config, "max_position_embeddings", 2048))
        # Cap at 2048 for practical evaluation (matches standard practice)
        if seqlen > 2048:
            seqlen = 2048
    print(f"[INFO] Using seqlen = {seqlen}")

    results = {}

    for dataset in datasets.split(","):
        dataset = dataset.strip()
        if not dataset:
            continue

        print(f"[INFO] Starting PPL eval for: {dataset}")
        testloader = get_eval_loaders(dataset, tokenizer)
        testenc = testloader.input_ids
        nsamples = testenc.numel() // seqlen

        if nsamples == 0:
            print(f"Not enough data for PPL evaluation on {dataset} with seqlen {seqlen}. Skipping.")
            continue

        nlls = []
        for i in tqdm(range(nsamples), desc=f"PPL({dataset})"):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(device)
            outputs = model(batch, use_cache=False)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * (seqlen - 1)
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))
        print(f"[{dataset}] PPL = {ppl.item():.4f}")
        results[dataset] = ppl.item()

    return results


def main(args):
    # === Draft model path: load via load_quantized_model ===
    if args.draft_model_path:
        print(f"[INFO] Loading draft model from: {args.draft_model_path}")
        model, tokenizer = load_draft_model_for_eval(
            draft_model_path=args.draft_model_path,
            base_model_id=args.model_id,
            device="auto",
        )

        # Standalone PPL eval (no lm_eval / BaseLM dependency)
        eval_ppl_standalone(
            model=model,
            tokenizer=tokenizer,
            datasets=args.ppl_task,
        )

    # === Standard path: model_type + LM.from_pretrained (requires modeling + lm_eval) ===
    else:
        from modeling import (LittleBitGemma2ForCausalLM, LittleBitGemma3ForCausalLM, LittleBitLlamaForCausalLM,
                              LittleBitOPTForCausalLM, LittleBitPhi4ForCausalLM, LittleBitQwQForCausalLM)

        lm_dict = {
            "llama": LittleBitLlamaForCausalLM,
            "gemma2": LittleBitGemma2ForCausalLM,
            "gemma3": LittleBitGemma3ForCausalLM,
            "phi4": LittleBitPhi4ForCausalLM,
            "qwq": LittleBitQwQForCausalLM,
            "opt": LittleBitOPTForCausalLM,
        }

        if args.model_type not in lm_dict:
            raise KeyError(f"Invalid model type: {args.model_type}. Available: {list(lm_dict.keys())}")

        LM = lm_dict[args.model_type]

        if args.use_accelerator:
            from accelerate import Accelerator

            accelerator = Accelerator()
            model = LM.from_pretrained(
                args.model_id,
                device_map=None,
                torch_dtype=torch.bfloat16,
                extra_config=args,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False, trust_remote_code=True)

            from torch.utils.data import TensorDataset, DataLoader
            dummy_data = torch.zeros((8, 1), dtype=torch.float32)
            dummy_dataset = TensorDataset(dummy_data)
            dummy_loader = DataLoader(dummy_dataset, batch_size=1)
            model, dummy_loader = accelerator.prepare(model, dummy_loader)
        else:
            accelerator = None
            model = LM.from_pretrained(
                args.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                extra_config=args,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_id, device_map="auto", use_fast=False,
                                                      trust_remote_code=True)

        _ = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            tasks=args.zeroshot_task,
            eval_ppl=args.ppl_task,
            batch_size=args.batch_size,
            accelerator=accelerator,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation Script with optional Accelerator/DeepSpeed support")
    parser.add_argument("--local_rank", type=int, default=-1, help="(Accelerator/DeepSpeed related) local rank")
    parser.add_argument("--use_accelerator", type=str2bool, default=False, help="Whether to use Accelerator/DeepSpeed")
    parser.add_argument("--model_type", type=str, default=None,
                        help="Model type (llama, gemma2, gemma3, phi4, qwq, opt). Not needed with --draft_model_path.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Base model ID (for tokenizer). Also used as model path if --draft_model_path is not set.")
    parser.add_argument("--draft_model_path", type=str, default=None,
                        help="Path to a LittleBit draft model checkpoint (with littlebit_config.json). "
                             "When set, loads the model via load_quantized_model instead of model_type.")
    parser.add_argument("--ppl_task", type=str, default="wikitext2,c4",
                        help="Perplexity evaluation dataset (comma-separated)")
    parser.add_argument("--zeroshot_task", type=str, default="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge",
                        help="Zero-shot evaluation tasks (comma-separated)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    parser.add_argument("--quant_func", type=str, default="STEBinary")
    parser.add_argument("--quant_mod", type=str, default="BinaryMoSLinear")
    parser.add_argument("--num_expert", type=int, default=4)
    parser.add_argument("--is_po2", type=str2bool, default=False)

    parser.add_argument("--split_dim", type=int, default=1024)
    parser.add_argument("--eff_bit", type=float, default=1.0)
    parser.add_argument("--residual", type=str2bool, default=False)

    args = parser.parse_args()

    main(args)
