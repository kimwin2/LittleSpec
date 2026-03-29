"""
Full Pipeline: Train 0.1-bit Draft + 0.9-bit Residual (Matryoshka Style)

Runs Step 1 (draft model QAT) and Step 2 (residual model QAT) in sequence.
The draft model checkpoint from Step 1 is automatically used as input 
for Step 2, eliminating the need to manually set DRAFT_MODEL_PATH.

Usage:
    deepspeed --num_gpus=4 train_full_pipeline.py \
        --model_id meta-llama/Llama-3.1-8B-Instruct \
        --dataset wikitext2_sharegpt \
        --ds_config_path configs/zero3.json \
        --step1_save_dir outputs/step1_draft_0.1bit \
        --step2_save_dir outputs/step2_residual_0.9bit
"""

import re
import hashlib
import argparse
import datetime
import json
import os
import gc
from pathlib import Path
from copy import deepcopy

import deepspeed
import GPUtil
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import default_data_collator
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    Trainer,
)
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from quantization.utils import apply_littlebit_patch
from quantization.utils.quant_util import load_quantized_model, get_quant_func_and_mod
from quantization.modules import LittleBitLinear
from utils.datautils import prepare_dataset, load_tokenizer
from utils.kd_utils import KDTrainer, TrainTimeTestKDTrainer
from utils.misc import setup_logger
from utils.utils import prepare_model_for_training, print_trainable_parameters

logger = setup_logger(__name__)


# ==============================================================================
# Shared utilities
# ==============================================================================

def get_device_config():
    gpus = GPUtil.getGPUs()
    if not gpus:
        return None, None

    device_map = "auto"
    local_rank_str = os.environ.get('LOCAL_RANK')
    if local_rank_str is not None:
        try:
            local_rank = int(local_rank_str)
            device_map = {'': local_rank}
        except ValueError:
            pass

    return len(gpus), device_map


def str2bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception(f'Boolean value expected: {value}')


def make_save_dir(base_dir, f_name=None):
    """Create a timestamped save directory under base_dir."""
    if f_name is None:
        f_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    save_dir = os.path.join(base_dir, f_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return save_dir


def get_training_arguments(args, save_dir, run_name):
    return TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        save_steps=10000,
        output_dir=save_dir,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        deepspeed=args.ds_config_path,
        report_to=args.report,
        run_name=run_name,
    )


def load_teacher_model(args, num_gpus, torch_dtype, config_path="configs/zero3_inference.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)

    _ = HfDeepSpeedConfig(config)

    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.config.use_cache = False

    teacher_model, _, _, _ = deepspeed.initialize(
        model=teacher_model,
        model_parameters=teacher_model.parameters(),
        config=config,
    )

    return teacher_model


# ==============================================================================
# Step 1 logic
# ==============================================================================

def load_student_model(args, device_map, torch_dtype, eff_bit):
    """Load the base model and apply 0.1-bit quantization patch."""
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    prepare_model_for_training(model)

    # Temporarily override eff_bit for step 1
    step1_args = argparse.Namespace(**vars(args))
    step1_args.eff_bit = eff_bit

    logger.info(f"Applying quantization patch with eff_bit={eff_bit}...")
    model = apply_littlebit_patch(model, step1_args, do_train=True)

    if device_map:
        model.to(device_map if isinstance(device_map, (str, torch.device)) else list(device_map.values())[0])

    print_trainable_parameters(model)
    return model


def setup_step1_trainer(model, teacher_model, tokenizer, datasets, training_args, args):
    if args.train_time_test:
        logger.info(f"Using TrainTimeTestKDTrainer (steps={args.ttt_steps}, decay={args.ttt_decay})")
        trainer = TrainTimeTestKDTrainer(
            model=model,
            teacher_model=teacher_model,
            l2l_loss_scale=args.l2l_loss_scale,
            train_time_test_steps=args.ttt_steps,
            train_time_test_decay=args.ttt_decay,
            processing_class=tokenizer,
            train_dataset=datasets,
            args=training_args,
            data_collator=default_data_collator,
        )
    else:
        logger.info("Using standard KDTrainer (single-step KD)")
        trainer = KDTrainer(
            model=model,
            teacher_model=teacher_model,
            l2l_loss_scale=args.l2l_loss_scale,
            processing_class=tokenizer,
            train_dataset=datasets,
            args=training_args,
            data_collator=default_data_collator,
        )
    return trainer


def save_step1_artifacts(trainer, model, tokenizer, save_dir, args):
    """Save the 0.1-bit draft model checkpoint."""
    try:
        logger.info("Saving Step 1 draft model artifacts (Grouped Chunk Strategy)...")

        if hasattr(trainer, 'accelerator'):
            unwrapped_model = trainer.accelerator.unwrap_model(model)
        else:
            unwrapped_model = model
            while hasattr(unwrapped_model, 'module'):
                unwrapped_model = unwrapped_model.module

        use_ds = (args.ds_config_path is not None)
        final_cpu_state_dict = {}

        if use_ds:
            logger.info("DeepSpeed ZeRO-3 enabled. Gathering parameters in groups...")

            LAYER_CHUNK_SIZE = 4
            for name, module in unwrapped_model.named_children():
                if isinstance(module, torch.nn.ModuleList):
                    num_layers = len(module)
                    for i in range(0, num_layers, LAYER_CHUNK_SIZE):
                        end_idx = min(i + LAYER_CHUNK_SIZE, num_layers)
                        layer_group = module[i:end_idx]

                        logger.info(f"Gathering layers {i} to {end_idx-1}...")

                        with deepspeed.zero.GatheredParameters(layer_group.parameters(), modifier_rank=0):
                            if args.local_rank == 0 or args.local_rank == -1:
                                for idx, layer in enumerate(layer_group):
                                    layer_global_idx = i + idx
                                    layer_state_dict = layer.state_dict()
                                    for k, v in layer_state_dict.items():
                                        final_cpu_state_dict[f"{name}.{layer_global_idx}.{k}"] = v.cpu()

                else:
                    logger.info(f"Processing module: {name}")
                    with deepspeed.zero.GatheredParameters(module.parameters(), modifier_rank=0):
                        if args.local_rank == 0 or args.local_rank == -1:
                            module_state_dict = module.state_dict()
                            for k, v in module_state_dict.items():
                                final_cpu_state_dict[f"{name}.{k}"] = v.cpu()

            remaining_params = [p for n, p in unwrapped_model.named_parameters() if '.' not in n]
            if remaining_params:
                with deepspeed.zero.GatheredParameters(remaining_params, modifier_rank=0):
                    if args.local_rank == 0 or args.local_rank == -1:
                        for n, p in unwrapped_model.named_parameters():
                            if '.' not in n:
                                final_cpu_state_dict[n] = p.cpu()
        else:
            final_cpu_state_dict = {k: v.cpu() for k, v in unwrapped_model.state_dict().items()}

        if args.local_rank == 0 or args.local_rank == -1:
            logger.info("Saving to disk...")

            quant_params = {
                "quant_func": getattr(args, "quant_func", "STEBinary"),
                "eff_bit": args.step1_eff_bit,
                "split_dim": getattr(args, "split_dim", 1024),
                "residual": getattr(args, "residual", False),
                "kv_factor": getattr(args, "kv_factor", 1.0),
                "min_split_dim": getattr(args, "min_split_dim", 8),
                "quant_mod": getattr(args, "quant_mod", "LittleBitLinear"),
                "matryoshka_stage": "draft",
                "matryoshka_bit": 0.1,
            }

            littlebit_config_path = os.path.join(save_dir, "littlebit_config.json")
            with open(littlebit_config_path, "w", encoding="utf-8") as f:
                json.dump(quant_params, f, indent=2)
            logger.info(f"Saved LittleBit config to {littlebit_config_path}")

            for key, value in quant_params.items():
                setattr(unwrapped_model.config, key, value)

            unwrapped_model.config.use_cache = True

            for k, v in final_cpu_state_dict.items():
                if "packed" not in k and "shape" not in k and v.dtype == torch.float32:
                    final_cpu_state_dict[k] = v.to(torch.bfloat16)

            unwrapped_model.save_pretrained(save_dir, state_dict=final_cpu_state_dict, safe_serialization=True)
            tokenizer.save_pretrained(save_dir)

            logger.info(f"Step 1 Draft model (0.1-bit) saved to {save_dir}")
            del final_cpu_state_dict
            gc.collect()

    except Exception as save_err:
        logger.error(f"Failed during Step 1 save: {save_err}", exc_info=True)
        raise


def run_step1(args, tokenizer, datasets, num_gpus, device_map, step1_save_dir):
    """Execute Step 1: Train 0.1-bit draft model."""
    logger.info("=" * 60)
    logger.info("STEP 1: Training 0.1-bit Draft Model")
    logger.info("=" * 60)

    # Load student model with 0.1-bit quantization
    model = load_student_model(args, device_map, torch.bfloat16, eff_bit=args.step1_eff_bit)

    # Load teacher model
    logger.info("Loading teacher model (FP)...")
    teacher_model = load_teacher_model(args, num_gpus, torch.bfloat16)

    training_args = get_training_arguments(args, step1_save_dir, run_name=args.step1_run_name)

    # Setup trainer
    trainer = setup_step1_trainer(model, teacher_model, tokenizer, datasets, training_args, args)

    # Train
    logger.info("Starting Step 1 QAT training (0.1-bit draft model)...")
    trainer.train()

    # Save
    save_step1_artifacts(trainer, model, tokenizer, step1_save_dir, args)

    logger.info("=" * 60)
    logger.info(f"Step 1 complete! Draft model saved to: {step1_save_dir}")
    logger.info("=" * 60)

    # Cleanup Step 1 models to free memory for Step 2
    del model, teacher_model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clear the HfDeepSpeedConfig global state left by Step 1's teacher model.
    # Without this, Step 2's from_pretrained(device_map="cpu") will fail with
    # "DeepSpeed Zero-3 is not compatible with passing a device_map".
    try:
        from transformers.integrations.deepspeed import unset_hf_deepspeed_config
        unset_hf_deepspeed_config()
        logger.info("Cleared HfDeepSpeedConfig after Step 1 cleanup")
    except (ImportError, Exception) as e:
        logger.warning(f"Could not unset HfDeepSpeedConfig: {e}")

    return step1_save_dir


# ==============================================================================
# Step 2 logic
# ==============================================================================

class MatryoshkaResidualModel(nn.Module):
    """
    Wrapper model that combines a frozen 0.1-bit draft model with a trainable 0.9-bit residual model.
    
    Forward pass: logits = draft_logits + residual_logits
    Only the residual model parameters are trainable.
    """
    def __init__(self, draft_model, residual_model, config):
        super().__init__()
        self.draft_model = draft_model
        self.residual_model = residual_model
        self._config = config
        
        # Freeze draft model completely
        for param in self.draft_model.parameters():
            param.requires_grad = False
        self.draft_model.eval()
    
    @property
    def config(self):
        return self._config
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, 
                output_hidden_states=False, **kwargs):
        # Draft forward (frozen, no grad)
        with torch.no_grad():
            draft_outputs = self.draft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )
        
        # Residual forward (trainable)
        residual_outputs = self.residual_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        
        # Combine logits
        combined_logits = draft_outputs.logits + residual_outputs.logits
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = combined_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Build output dict compatible with HuggingFace
        from transformers.modeling_outputs import CausalLMOutputWithPast
        
        combined_hidden_states = None
        if output_hidden_states:
            draft_hs = draft_outputs.hidden_states
            residual_hs = residual_outputs.hidden_states
            combined_hidden_states = tuple(
                d + r for d, r in zip(draft_hs, residual_hs)
            )
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=combined_logits,
            past_key_values=None,
            hidden_states=combined_hidden_states,
            attentions=None,
        )
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.residual_model.gradient_checkpointing_enable(**kwargs)
    
    def get_input_embeddings(self):
        return self.residual_model.get_input_embeddings()
    
    def parameters(self, recurse=True):
        return self.residual_model.parameters(recurse)
    
    def named_parameters(self, prefix='', recurse=True):
        return self.residual_model.named_parameters(prefix=prefix + 'residual_model.', recurse=recurse)


class MatryoshkaKDTrainer(Trainer):
    """
    Knowledge distillation trainer for the Matryoshka residual model.
    The combined model (draft + residual) is trained to match the FP teacher.
    """
    def __init__(self, teacher_model, l2l_loss_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.l2l_loss_scale = l2l_loss_scale

    def ce_loss(self, student_logits, teacher_logits):
        model_output_log_prob = F.log_softmax(student_logits, dim=-1)
        real_output_soft = F.softmax(teacher_logits, dim=-1)
        return F.kl_div(model_output_log_prob, real_output_soft, reduction="batchmean")

    def mse_loss(self, student_logits, teacher_logits):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            return F.mse_loss(student_logits, teacher_logits)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs["output_hidden_states"] = True

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        teacher_logits = teacher_outputs.get("logits")
        teacher_reps = teacher_outputs.hidden_states[1:]
        del teacher_outputs

        outputs = model(**inputs)

        student_logits = outputs.logits
        student_reps = outputs.hidden_states[1:] if outputs.hidden_states else []

        if not return_outputs:
            del_outputs = outputs

        kd_loss = self.ce_loss(student_logits, teacher_logits)

        l2l_loss = torch.tensor(0.0, device=student_logits.device)
        if student_reps and teacher_reps:
            for student_rep, teacher_rep in zip(student_reps, teacher_reps):
                tmp_loss = self.mse_loss(student_rep, teacher_rep)
                l2l_loss = l2l_loss + tmp_loss
            l2l_loss = self.l2l_loss_scale * l2l_loss

        loss = kd_loss + l2l_loss

        self.log({
            "l2l_loss": l2l_loss.item(),
            "kd_loss": kd_loss.item(),
        })

        return (loss, outputs) if return_outputs else loss


def compute_residual_weights(original_model, draft_model, device='cpu'):
    """
    Compute residual weights: W_residual = W_original - W_draft_approx
    """
    residual_weights = {}
    original_state_dict = original_model.state_dict()

    quant_func_name = "STEBinary"
    from quantization.functions import STEBinary

    draft_calc_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    for name, module in draft_model.named_modules():
        if isinstance(module, LittleBitLinear):
            U = module.U.data.float().to(draft_calc_device)
            V = module.V.data.float().to(draft_calc_device)
            u1 = module.u1.data.float().to(draft_calc_device)
            u2 = module.u2.data.float().to(draft_calc_device)
            v1 = module.v1.data.float().to(draft_calc_device)
            v2 = module.v2.data.float().to(draft_calc_device)

            if module._binarized:
                Uq = U
                Vq = V
            else:
                Uq = STEBinary(U)
                Vq = STEBinary(V)

            W_approx = (Uq * (u1.t() @ u2)) @ (Vq * (v1.t() @ v2))

            weight_key = name + ".weight"
            if weight_key in original_state_dict:
                W_original = original_state_dict[weight_key].float().to(draft_calc_device)
                W_residual = W_original - W_approx
                residual_weights[weight_key] = W_residual.cpu()
                logger.info(f"Computed residual for {weight_key}: "
                           f"original_norm={W_original.norm():.4f}, "
                           f"approx_norm={W_approx.norm():.4f}, "
                           f"residual_norm={W_residual.norm():.4f}")
                del W_original, W_approx
            else:
                logger.warning(f"Original weight key {weight_key} not found, skipping")

            del U, V, u1, u2, v1, v2, Uq, Vq
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return residual_weights


def initialize_residual_model_weights(residual_model, residual_weights):
    """
    Initialize the residual model's LittleBitLinear layers from pre-computed residual weights.
    """
    from quantization.functions import STEBinary

    calc_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    for name, module in residual_model.named_modules():
        if isinstance(module, LittleBitLinear):
            weight_key = name + ".weight"
            if weight_key in residual_weights:
                W_residual = residual_weights[weight_key].float()

                logger.info(f"Re-initializing {name} from residual weights (shape={W_residual.shape})")

                W_calc = W_residual.to(calc_device)
                split_dim = module.split_dim

                U_t, S_t, V_t = torch.svd_lowrank(W_calc, q=split_dim)
                Vh_t = V_t.t()

                sqrt_S = torch.sqrt(torch.diag(S_t))[:, :split_dim]

                U_new = (U_t @ sqrt_S).contiguous()
                V_new = (sqrt_S.t() @ Vh_t).contiguous()

                def rank_one_decompose(X):
                    U_r, S_r, V_r = torch.svd_lowrank(X.to(calc_device), q=1)
                    Vh_r = V_r.t()
                    sqrt_S0 = torch.sqrt(S_r[0])
                    u_comp = (U_r[:, :1] * sqrt_S0).t().contiguous()
                    v_comp = (sqrt_S0 * Vh_r[:1, :]).contiguous()
                    return u_comp, v_comp

                v1_new, v2_new = rank_one_decompose(torch.abs(V_new))
                u1_new, u2_new = rank_one_decompose(torch.abs(U_new))

                dtype = module.U.dtype if hasattr(module, 'U') and module.U is not None else torch.bfloat16
                device = 'cpu'

                module.U = nn.Parameter(U_new.to(device=device, dtype=dtype), requires_grad=True)
                module.V = nn.Parameter(V_new.to(device=device, dtype=dtype), requires_grad=True)
                module.u1 = nn.Parameter(u1_new.to(device=device, dtype=dtype), requires_grad=True)
                module.u2 = nn.Parameter(u2_new.to(device=device, dtype=dtype), requires_grad=True)
                module.v1 = nn.Parameter(v1_new.to(device=device, dtype=dtype), requires_grad=True)
                module.v2 = nn.Parameter(v2_new.to(device=device, dtype=dtype), requires_grad=True)

                del W_calc, U_t, S_t, V_t, Vh_t, U_new, V_new
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info(f"  -> Initialized {name} with split_dim={split_dim}")
            else:
                logger.warning(f"No residual weight found for {name}, keeping default init")


def save_step2_artifacts(trainer, residual_model, tokenizer, save_dir, args, draft_model_path):
    """Save only the residual (0.9-bit) model weights."""
    try:
        logger.info("Saving Step 2 residual model artifacts...")

        if hasattr(trainer, 'accelerator'):
            unwrapped = trainer.accelerator.unwrap_model(residual_model)
        else:
            unwrapped = residual_model
            while hasattr(unwrapped, 'module'):
                unwrapped = unwrapped.module

        # If it's a MatryoshkaResidualModel, extract just the residual part
        if isinstance(unwrapped, MatryoshkaResidualModel):
            actual_residual = unwrapped.residual_model
        else:
            actual_residual = unwrapped

        use_ds = (args.ds_config_path is not None)
        final_cpu_state_dict = {}

        if use_ds:
            LAYER_CHUNK_SIZE = 4
            for name, module in actual_residual.named_children():
                if isinstance(module, torch.nn.ModuleList):
                    num_layers = len(module)
                    for i in range(0, num_layers, LAYER_CHUNK_SIZE):
                        end_idx = min(i + LAYER_CHUNK_SIZE, num_layers)
                        layer_group = module[i:end_idx]
                        logger.info(f"Gathering layers {i} to {end_idx-1}...")
                        with deepspeed.zero.GatheredParameters(layer_group.parameters(), modifier_rank=0):
                            if args.local_rank == 0 or args.local_rank == -1:
                                for idx, layer in enumerate(layer_group):
                                    layer_global_idx = i + idx
                                    layer_state_dict = layer.state_dict()
                                    for k, v in layer_state_dict.items():
                                        final_cpu_state_dict[f"{name}.{layer_global_idx}.{k}"] = v.cpu()
                else:
                    logger.info(f"Processing module: {name}")
                    with deepspeed.zero.GatheredParameters(module.parameters(), modifier_rank=0):
                        if args.local_rank == 0 or args.local_rank == -1:
                            module_state_dict = module.state_dict()
                            for k, v in module_state_dict.items():
                                final_cpu_state_dict[f"{name}.{k}"] = v.cpu()

            remaining_params = [p for n, p in actual_residual.named_parameters() if '.' not in n]
            if remaining_params:
                with deepspeed.zero.GatheredParameters(remaining_params, modifier_rank=0):
                    if args.local_rank == 0 or args.local_rank == -1:
                        for n, p in actual_residual.named_parameters():
                            if '.' not in n:
                                final_cpu_state_dict[n] = p.cpu()
        else:
            final_cpu_state_dict = {k: v.cpu() for k, v in actual_residual.state_dict().items()}

        if args.local_rank == 0 or args.local_rank == -1:
            quant_params = {
                "quant_func": args.quant_func,
                "eff_bit": args.step2_eff_bit,
                "split_dim": args.split_dim,
                "residual": args.residual,
                "kv_factor": args.kv_factor,
                "min_split_dim": args.min_split_dim,
                "quant_mod": args.quant_mod,
                "matryoshka_stage": "residual",
                "matryoshka_bit": 0.9,
                "draft_model_path": draft_model_path,
            }

            littlebit_config_path = os.path.join(save_dir, "littlebit_config.json")
            with open(littlebit_config_path, "w", encoding="utf-8") as f:
                json.dump(quant_params, f, indent=2)

            for key, value in quant_params.items():
                if key not in ("draft_model_path", "matryoshka_stage", "matryoshka_bit"):
                    setattr(actual_residual.config, key, value)

            actual_residual.config.use_cache = True

            for k, v in final_cpu_state_dict.items():
                if "packed" not in k and "shape" not in k and v.dtype == torch.float32:
                    final_cpu_state_dict[k] = v.to(torch.bfloat16)

            actual_residual.save_pretrained(save_dir, state_dict=final_cpu_state_dict, safe_serialization=True)
            tokenizer.save_pretrained(save_dir)

            logger.info(f"Residual model (0.9-bit) saved to {save_dir}")
            del final_cpu_state_dict
            gc.collect()

    except Exception as save_err:
        logger.error(f"Failed during Step 2 save: {save_err}", exc_info=True)
        raise


def run_step2(args, tokenizer, datasets, num_gpus, device_map, draft_model_path, step2_save_dir):
    """Execute Step 2: Train 0.9-bit residual model using the draft model from Step 1."""
    logger.info("=" * 60)
    logger.info("STEP 2: Training 0.9-bit Residual Model")
    logger.info(f"  Draft model path: {draft_model_path}")
    logger.info("=" * 60)

    # ===== Step A: Load the original FP model =====
    logger.info("Loading original FP model for residual computation...")
    original_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float32, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )

    # ===== Step B: Load the trained 0.1-bit draft model =====
    logger.info(f"Loading trained 0.1-bit draft model from {draft_model_path}...")

    draft_config_path = os.path.join(draft_model_path, "littlebit_config.json")
    with open(draft_config_path, 'r') as f:
        draft_config = json.load(f)

    draft_args = argparse.Namespace(
        quant_func=draft_config.get("quant_func", "STEBinary"),
        quant_mod=draft_config.get("quant_mod", "LittleBitLinear"),
        eff_bit=draft_config.get("eff_bit", 0.1),
        split_dim=draft_config.get("split_dim", 1024),
        residual=draft_config.get("residual", False),
        kv_factor=draft_config.get("kv_factor", 1.0),
        min_split_dim=draft_config.get("min_split_dim", 8),
        model_id=draft_model_path,
    )

    draft_model = load_quantized_model(
        model_path=draft_model_path,
        quant_args=draft_args,
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    # ===== Step C: Compute residual weights =====
    logger.info("Computing residual weights (W_original - W_draft_approx)...")
    residual_weights = compute_residual_weights(original_model, draft_model, device='cpu')

    del original_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===== Step D: Create and initialize residual model (0.9-bit) =====
    logger.info("Creating 0.9-bit residual model...")
    residual_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    residual_model.config.use_cache = False

    # Apply LittleBit patch with 0.9-bit configuration
    step2_args = argparse.Namespace(**vars(args))
    step2_args.eff_bit = args.step2_eff_bit

    logger.info(f"Applying LittleBit patch with eff_bit={args.step2_eff_bit}...")
    residual_model = apply_littlebit_patch(residual_model, step2_args, do_train=True)

    # Re-initialize from residual weights
    logger.info("Initializing residual model from computed residual weights...")
    initialize_residual_model_weights(residual_model, residual_weights)
    del residual_weights
    gc.collect()

    # ===== Step E: Create combined Matryoshka model =====
    logger.info("Creating Matryoshka combined model (draft frozen + residual trainable)...")
    combined_model = MatryoshkaResidualModel(
        draft_model=draft_model,
        residual_model=residual_model,
        config=residual_model.config,
    )

    for name, param in combined_model.residual_model.named_parameters():
        if any(key in name for key in ("lm_head", "embed")):
            param.requires_grad = False
        else:
            param.requires_grad = True

    if hasattr(combined_model.residual_model, "enable_input_require_grads"):
        combined_model.residual_model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        combined_model.residual_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    combined_model.residual_model.gradient_checkpointing_enable()

    if device_map:
        target = device_map if isinstance(device_map, (str, torch.device)) else list(device_map.values())[0]
        combined_model.to(target)

    print_trainable_parameters(combined_model.residual_model)

    # ===== Step F: Train =====
    logger.info("Loading teacher model for KD...")
    teacher_model = load_teacher_model(args, num_gpus, torch.bfloat16)

    training_args = get_training_arguments(args, step2_save_dir, run_name=args.step2_run_name)

    trainer = MatryoshkaKDTrainer(
        teacher_model=teacher_model,
        l2l_loss_scale=args.l2l_loss_scale,
        model=combined_model,
        processing_class=tokenizer,
        train_dataset=datasets,
        args=training_args,
        data_collator=default_data_collator,
    )

    logger.info("Starting Step 2 QAT training (0.9-bit residual model)...")
    trainer.train()

    # ===== Step G: Save =====
    save_step2_artifacts(trainer, combined_model, tokenizer, step2_save_dir, args, draft_model_path)

    logger.info("=" * 60)
    logger.info("Step 2 training complete!")
    logger.info(f"Residual model saved to: {step2_save_dir}")
    logger.info(f"Draft model path: {draft_model_path}")
    logger.info("=" * 60)

    del combined_model, teacher_model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==============================================================================
# Argument parser
# ==============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Full Pipeline: Train 0.1-bit Draft + 0.9-bit Residual Model"
    )
    # Model
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")

    # Dataset
    parser.add_argument("--data_root", type=str, default="./")
    parser.add_argument("--dataset", type=str, default="wikitext2_sharegpt",
                        choices=['c4', 'wikitext2', 'c4_wiki', 'wikitext2_sharegpt', 'openhermes'])
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of samples to use from OpenHermes 2.5 (default: 50000)")
    parser.add_argument("--sharegpt_path", type=str, default=None,
                        help="Path to local ShareGPT .jsonl file (optional)")

    # Output directories
    parser.add_argument("--step1_save_dir", type=str, default='outputs/step1_draft_0.1bit')
    parser.add_argument("--step2_save_dir", type=str, default='outputs/step2_residual_0.9bit')
    parser.add_argument("--f_name", type=str, default=None,
                        help="Shared timestamp folder name for both steps (auto-generated if not set)")

    # Common training params
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--l2l_loss_scale", type=float, default=10.0)
    parser.add_argument("--dataset_prepared", type=str2bool, default=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--ds_config_path", type=str, default="configs/zero3.json")
    parser.add_argument("--report", nargs="+", default=["wandb"], choices=["wandb", "tensorboard"])

    # Quantization params
    parser.add_argument("--quant_func", type=str, default="STEBinary")
    parser.add_argument("--quant_mod", type=str, default="LittleBitLinear")
    parser.add_argument("--residual", type=str2bool, default=False)
    parser.add_argument("--split_dim", type=int, default=1024)
    parser.add_argument("--kv_factor", type=float, default=1.0)
    parser.add_argument("--min_split_dim", type=int, default=8)

    # Step-specific bit widths
    parser.add_argument("--step1_eff_bit", type=float, default=0.1,
                        help="Effective bit width for Step 1 draft model")
    parser.add_argument("--step2_eff_bit", type=float, default=0.9,
                        help="Effective bit width for Step 2 residual model")

    # Step-specific run names (for wandb/tensorboard)
    parser.add_argument("--step1_run_name", type=str, default="step1_draft_0.1bit")
    parser.add_argument("--step2_run_name", type=str, default="step2_residual_0.9bit")

    # Training-time test (Step 1 only, EAGLE-style multi-step rollout)
    parser.add_argument("--train_time_test", type=str2bool, default=False,
                        help="Enable training-time test for Step 1 (multi-step rollout)")
    parser.add_argument("--ttt_steps", type=int, default=7,
                        help="Number of rollout steps for training-time test")
    parser.add_argument("--ttt_decay", type=float, default=0.8,
                        help="Exponential decay factor for rollout loss weighting")

    # Pipeline control
    parser.add_argument("--skip_step1", action="store_true",
                        help="Skip Step 1 and use --draft_model_path for Step 2")
    parser.add_argument("--skip_step2", action="store_true",
                        help="Only run Step 1 (same as train_step1_draft.py)")
    parser.add_argument("--draft_model_path", type=str, default=None,
                        help="Path to existing draft model (only used with --skip_step1)")

    args = parser.parse_args()
    return args


# ==============================================================================
# Main pipeline
# ==============================================================================

def main():
    args = get_args()
    set_seed(args.seed)

    # Generate a shared timestamp for both steps
    f_name = args.f_name or datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')

    num_gpus, device_map = get_device_config()

    # Load tokenizer and dataset (shared between both steps)
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_id)

    logger.info(f"Preparing training data ({args.dataset})...")
    datasets = prepare_dataset(args, tokenizer)

    draft_model_path = None

    # ============================
    # Step 1: Train draft model
    # ============================
    if not args.skip_step1:
        step1_save_dir = make_save_dir(args.step1_save_dir, f_name)
        draft_model_path = run_step1(
            args, tokenizer, datasets, num_gpus, device_map, step1_save_dir
        )
    else:
        if not args.draft_model_path:
            raise ValueError("--draft_model_path is required when --skip_step1 is set")
        if not os.path.isdir(args.draft_model_path):
            raise ValueError(f"Draft model not found: {args.draft_model_path}")
        draft_model_path = args.draft_model_path
        logger.info(f"Skipping Step 1. Using existing draft model: {draft_model_path}")

    # ============================
    # Step 2: Train residual model
    # ============================
    if not args.skip_step2:
        step2_save_dir = make_save_dir(args.step2_save_dir, f_name)
        run_step2(
            args, tokenizer, datasets, num_gpus, device_map,
            draft_model_path, step2_save_dir
        )
    else:
        logger.info("Skipping Step 2 (--skip_step2 was set).")

    # ============================
    # Final summary
    # ============================
    logger.info("")
    logger.info("=" * 60)
    logger.info("FULL PIPELINE COMPLETE!")
    logger.info("=" * 60)
    if not args.skip_step1:
        logger.info(f"  Step 1 (0.1-bit draft) -> {draft_model_path}")
    if not args.skip_step2:
        logger.info(f"  Step 2 (0.9-bit residual) -> {step2_save_dir}")
    logger.info("")
    logger.info("Next: Run speculative decoding!")
    logger.info("  scripts/run_speculative_decoding.sh")
    logger.info("  scripts/eval_speculative.sh")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
