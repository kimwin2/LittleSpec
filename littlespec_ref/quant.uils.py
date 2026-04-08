import importlib
import os
import re
import json

from collections import ChainMap, defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig

from .binary_packer import binary_unpacker
from .int4_packer import int4_unpacker

__all__ = ['patch_inst', 'load_quantized_model']

# =================================================================================
# Section 1: Model Patching Utilities
# =================================================================================


# Global member of the file that contains the mapping of quantized modules
# Using a function to lazily load the default mapping to avoid circular imports
def _get_default_mapping():
    from quantization.modules import LittleBitLinear
    return {
        nn.Linear: LittleBitLinear,
    }


def _match_pattern(patterns: list, model: nn.Module, name: str, mod: nn.Module) -> bool:
    """
    Match a module against a list of patterns.

    Args:
        patterns (list): List of patterns to match against.
        model (nn.Module): Model to search for the module.
        name (str): Name of the module.
        mod (nn.Module): Module to match.

    Returns:
        bool: True if the module matches any of the patterns, False otherwise.
    """
    for _pattern in patterns:
        if isinstance(_pattern, str):
            _cond = partial(lambda _module_name, _prefix: is_module_include(_module_name, _prefix), _prefix=_pattern)
        elif isinstance(_pattern, type):
            _cond = partial(lambda _module_name, _cls_type: isinstance(mod, _cls_type), _cls_type=_pattern)
        elif isinstance(_pattern, re.Pattern):
            _cond = partial(lambda _module_name, _regex: bool(_regex.search(_module_name)), _regex=_pattern)
        else:
            raise ValueError(f'Invalid pattern: {_pattern}')
        if _cond(name):
            return True
    return False


def patch_inst(
    model: nn.Module,
    mapping: Optional[Dict[type, type]] = None,
    convert_kwargs: Optional[List[Tuple[list, dict]]] = None,
    exclude_names: Optional[List[str]] = None,
    device_map: dict = None,
    do_hessian = False,
    hessian_path=""
):
    """
    Patch instances of a model with quantized modules.

    Args:
        model (nn.Module): Model to patch.
        mapping (Optional[Dict[type, type]]): Mapping of original module types to quantized module types.
        convert_kwargs (Optional[List[Tuple[list, dict]]]): List of tuples containing patterns and keyword
            arguments for converting modules to quantized modules.
        exclude_names (Optional[List[str]]): List of layer names that should not be converted.
        device_map (dict): A dictionary mapping device names to devices.
    """
    # Use lazy loading for default mapping to avoid circular imports
    if mapping is None:
        mapping = _get_default_mapping()
    convert_kwargs = convert_kwargs or []
    exclude_names = exclude_names or []
    device_map = device_map or {}
    mapping = mapping or _DEFAULT_MAPPING

    if device_map == "auto":
        from accelerate import infer_auto_device_map
        device_map = infer_auto_device_map(model)
    default_device = device_map.get("", "cpu")

    mapping_chained = ChainMap({}, mapping)

    if do_hessian:
        print("Do hessian is True, Load Hessian")
        hessian_mat = torch.load(hessian_path)

    # Assume model is a tree structure
    # HACK: depends on the implementation detail of pytorch that .named_modules() yields modules in preordering
    for name, mod in model.named_modules():
        if name in exclude_names:
            continue

        convert_kwargs_ = {}
        for pattern, d in convert_kwargs:
            if _match_pattern(pattern, model, name, mod):
                convert_kwargs_.update(d)

        this_hessian = None        
        if do_hessian:
            name_lst = name.split(".")
            if len(name_lst) >=5 and name_lst[-1].endswith("proj"):
                try:
                    layer_num = int(name_lst[2])
                    this_hessian = hessian_mat[layer_num][name_lst[3] + "." + name_lst[4]]
                except Exception as e:
                    print(e)
                    pass

            if name == "lm_head":
                try:
                    this_hessian = hessian_mat[name]    
                except Exception as e:
                    print(e)
                    pass
        convert_kwargs_["this_hessian"] = this_hessian

        if type(mod) in mapping_chained:
            mod.__class__ = mapping_chained[type(mod)]

        if hasattr(mod, '__quant_convert__'):
            if not device_map:
                mod.__quant_convert__(**convert_kwargs_)
            else:
                mod.to(default_device)
                mod.__quant_convert__(**convert_kwargs_)
                mod.to("cpu") if default_device != "cpu" else None
                print(f"Layer: {name} Converted")
    use_lax = False
    for _, kwargs_dict in convert_kwargs:
        if kwargs_dict.get("use_lax", False):
            use_lax = True
            break

    if use_lax:
        print("[LAX] Applying LAX Block Replacement & Forward Patching...")
        from quantization.modules.lax_llama import LaxLlamaDecoderLayer, LaxLlamaAttention, LaxLlamaMLP
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention, LlamaMLP
        
        # 2-1. DecoderLayer를 LaxLlamaDecoderLayer로 "기능" 변경
        # 객체를 새로 생성하기보다 __class__를 변경하고 forward 메서드를 오버라이드 하는 것이 안전함
        # (이미 weight가 로드되어 있을 수 있으므로)
        for name, module in model.named_modules():
            if isinstance(module, LlamaDecoderLayer):
                # 클래스 변경 (LaxLlamaDecoderLayer의 forward를 쓰기 위함)
                module.__class__ = LaxLlamaDecoderLayer
                
                # 내부 Attention, MLP도 변경
                if isinstance(module.self_attn, LlamaAttention):
                    module.self_attn.__class__ = LaxLlamaAttention
                if isinstance(module.mlp, LlamaMLP):
                    module.mlp.__class__ = LaxLlamaMLP
                    
        # 2-2. LlamaModel의 Forward Loop 패치
        # model.model (LlamaModel)의 forward를 재정의하여 prev_block_latents를 전달하도록 함
        if hasattr(model, "model"):
             model.model.forward = types.MethodType(lax_llama_model_forward, model.model)
             print("[LAX] LlamaModel forward patched successfully.")


def lax_llama_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # [LAX LOOP START]
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    
    prev_block_latents = None # 첫 레이어는 None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            # Checkpoint 함수가 Gradient를 추적하려면 입력이 Tensor여야 합니다.
            # prev_block_latents는 Tensor List이므로, use_reentrant=False를 써야 안전합니다.
            
            def create_custom_forward(module):
                def custom_forward(*args):
                    # args의 마지막 인자가 prev_block_latents임
                    h_states = args[0]
                    p_latents = args[-1] 
                    
                    # 나머지 인자들은 위치에 맞춰 언패킹 (kwargs는 checkpoint통과시 위치인자로 바뀜)
                    # args: (hidden_states, attention_mask, position_ids, past_key_values, 
                    #        output_attentions, use_cache, cache_position, position_embeddings, 
                    #        prev_block_latents)
                    
                    return module(
                        h_states,
                        attention_mask=args[1],
                        position_ids=args[2],
                        past_key_value=args[3],
                        output_attentions=args[4],
                        use_cache=args[5],
                        cache_position=args[6],
                        position_embeddings=args[7],
                        prev_block_latents=p_latents, # 명시적 전달
                        **flash_attn_kwargs,
                    )
                return custom_forward

            # checkpoint 함수 호출
            # 모든 인자를 위치 인자(Positional Args)로 넘겨야 합니다.
            layer_outputs_tuple = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                prev_block_latents, # <--- Tensor List지만 use_reentrant=False면 통과 가능
                use_reentrant=False 
            )
            
        # ------------------------------------------------------------------
        # Case B: 일반 Forward (Inference 또는 Checkpointing Off)
        # ------------------------------------------------------------------
        else:
            layer_outputs_tuple = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                prev_block_latents=prev_block_latents, # [LAX]
                **flash_attn_kwargs,
            )

        # 3. 결과 언패킹 및 다음 단계 준비
        # LaxLlamaDecoderLayer는 (outputs, curr_latents)를 리턴함
        # outputs = (hidden_states, attn_weights(opt))
        
        layer_outputs = layer_outputs_tuple[0] # (hidden_states, ...)
        curr_block_latents = layer_outputs_tuple[1] # [q, k, v, ...]

        hidden_states = layer_outputs[0]
        
        # [LAX] Update latents for next layer
        prev_block_latents = curr_block_latents

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    from transformers.modeling_outputs import BaseModelOutputWithPast
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def is_module_include(name, target):
    """
    Check if a module name includes a target substring.

    Args:
        name (str): Full name of the module.
        target (str): Target substring to check for inclusion.

    Returns:
        bool: True if the module name includes the target substring, False otherwise.
    """
    _module_names = name.split(".")
    while len(_module_names) > 0:
        if ".".join(_module_names).startswith(target):
            return True
        _module_names.pop(0)
    return False


def load_module_and_get_attr(package_path, module_name):
    """
    Load a module from a package and get a specific attribute.

    Args:
        package_path (str): The path to the package.
        module_name (str): The name of the module.

    Returns:
        attr: The attribute from the module.
    """
    package = importlib.import_module(package_path)
    module = getattr(package, module_name)
    if module is None:
        raise ValueError(f"{module_name} not found in {package}.")
    return module


def get_quant_func_and_mod(quant_func_name, quant_mod_name):
    """
    Get a quant function and a quant module.

    Args:
        quant_func_name (str): The name of the quant function.
        quant_mod_name (str): The name of the quant module.

    Returns:
        quant_func: The quant function.
        quant_mod: The quant module.
    """
    for name in (quant_func_name, quant_mod_name):
        if not isinstance(name, str):
            raise ValueError("All names must be strings.")

    quant_func_package = "quantization.functions"
    quant_mod_package = "quantization.modules"

    quant_func = load_module_and_get_attr(quant_func_package, quant_func_name)
    quant_mod = load_module_and_get_attr(quant_mod_package, quant_mod_name)

    return quant_func, quant_mod


# =================================================================================
# Section 2: Quantized Model Loading Utilities
# =================================================================================


def _load_and_process_state_dict(model_path: str, torch_dtype: torch.dtype, default_device="cuda", pack_type="binary"):
    """
    Loads a state_dict from a local path, handling both sharded and single-file safetensors formats,
    and unpacks packed weights if necessary.

    Args:
        model_path (str): Path to the directory containing the model files.
        torch_dtype (torch.dtype): Desired torch dtype for the loaded tensors.

    Returns:
        Tuple[Dict[str, torch.Tensor], bool]: A tuple containing the processed state dict and a boolean
        indicating whether the weights were originally packed (True) or unpacked (False).
    """
    state_dict = {}

    # Handle both sharded and single-file models
    if os.path.exists(index_path := os.path.join(model_path, "model.safetensors.index.json")):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        for shard_file in set(index["weight_map"].values()):
            with safe_open(os.path.join(model_path, shard_file), framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
    else:
        single_file = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(single_file):
            raise FileNotFoundError(f"Could not find model weights at {single_file} or an index file.")
        with safe_open(single_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
                # print(key)
    #새코드
    # OneBit학습 잘못된거 로딩할때 필요
    # has_packed_weights = False 
    # # 원래 코드는 주석 처리
    # # has_packed_weights = any(key.endswith("_packed") for key in state_dict.keys())
    
    # if not has_packed_weights:
    #     print("INFO: Unpacked (legacy) pre-quantized format detected. Loading weights as is.")
        
    #     # [중요] 버그로 인해 생긴 'weight_packed' 키가 state_dict에 남아있으면 에러가 날 수 있으므로 제거
    #     if 'weight_packed' in state_dict: del state_dict['weight_packed']
    #     if 'weight_shape' in state_dict: del state_dict['weight_shape']
        
    #     return state_dict, False  # Unpacked status
    # 원코드
    has_packed_weights = any(key.endswith("_packed") for key in state_dict.keys())
    if not has_packed_weights:
        print("INFO: Unpacked (legacy) pre-quantized format detected. Loading weights as is.")
        return state_dict, False  # Unpacked status
    # 여기까지 주석

    print("INFO: Packed pre-quantized format detected. Unpacking weights...")
    packed_components = defaultdict(dict)
    other_state_dict = {}
    pattern = re.compile(r"^(.*)\.([^.]+?)_(packed|shape)$")

    for key, value in state_dict.items():
        if (match := pattern.match(key)):
            prefix, param_name, suffix_type = match.groups()
            packed_components[prefix][f"{param_name}_{suffix_type}"] = value
        else:
            other_state_dict[key] = value

    unpacked_state_dict = other_state_dict
    for prefix, components in packed_components.items():
        param_names = {
            key.replace("_packed", "") 
            for key in components.keys() 
            if key.endswith("_packed")
        }
        
        for name in param_names:
            packed_key = f"{name}_packed"
            shape_key = f"{name}_shape"
            
            if packed_key in components and shape_key in components:
                shape = tuple(components[shape_key].tolist())
                packed_tensor_gpu = components[packed_key]#.to("cuda")
                if pack_type == "int4":
                    unpacked_float = (int4_unpacker(packed_tensor_gpu, shape).float() / 8.0 )
                    unpacked_tensor = unpacked_float.to(torch_dtype)
                else:
                    unpacked_tensor = binary_unpacker(packed_tensor_gpu, shape).to(torch_dtype)
                unpacked_state_dict[f"{prefix}.{name}"] = unpacked_tensor
                
    return unpacked_state_dict, True
    # unpacked_state_dict = other_state_dict
    # for prefix, components in packed_components.items():
    #     param_names = {key.split('_')[0] for key in components.keys()}
    #     for name in param_names:
    #         packed_key = f"{name}_packed"
    #         shape_key = f"{name}_shape"
    #         if packed_key in components and shape_key in components:
    #             shape = tuple(components[shape_key].tolist())
    #             unpacked_tensor = binary_unpacker(components[packed_key], shape).to(torch_dtype)
    #             unpacked_state_dict[f"{prefix}.{name}"] = unpacked_tensor
    # return unpacked_state_dict, True  # Unpacked status


def load_quantized_model(model_path: str, quant_args, torch_dtype, device: str = "auto"):
    """
    Loads a pre-quantized model (packed or legacy) from a local directory.
    The model is initially loaded onto CPU, then optionally moved to the specified device.

    Args:
        model_path (str): Path to the saved pre-quantized model directory.
        quant_args: Namespace or similar object containing quantization parameters such as
                    `model_type`, `quant_func`, and `split_dim`.
        torch_dtype (torch.dtype): Desired torch dtype for model parameters.
        device (str, optional): Target device for the model. If "auto", selects CUDA if available,
                                otherwise CPU. Defaults to "auto".

    Returns:
        torch.nn.Module: The loaded and (if applicable) binarized model ready for inference.
    """
    import modeling
    from quantization.modules import LittleBitLinear, LittleBitRotLinear, BinaryMoSLinear, OneBitLinear

    if not os.path.isdir(model_path):
        raise ValueError(f"`load_quantized_model` expects a local directory path, but got: {model_path}")

    print(f"INFO: Loading pre-quantized model from local path '{model_path}'.")
    # Load config from the correct model_path
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Determine the correct LittleBit model class
    lm_dict = {
        "llama": "LittleBitLlamaForCausalLM",
        "gemma2": "LittleBitGemma2ForCausalLM",
        "gemma3": "LittleBitGemma3ForCausalLM",
        "phi4": "LittleBitPhi4ForCausalLM",
        "qwq": "LittleBitQwQForCausalLM",
        "opt": "LittleBitOPTForCausalLM",
    }
    # Use quant_args for model_type
    model_type_in_args = quant_args.model_type
    if not model_type_in_args:
        # Infer from config if needed
        config_model_type = getattr(config, "model_type", "").lower()
        for key in lm_dict.keys():
            if key in config_model_type:
                model_type_in_args = key
                print(f"INFO: Inferred model_type '{model_type_in_args}' from config.")
                break
    if not model_type_in_args:
        raise ValueError("Could not determine `model_type` from args or config for pre-quantized model.")

    ModelClass = getattr(modeling, lm_dict[model_type_in_args])

    # Instantiate using quant_args for extra_config
    model = ModelClass(config=config, extra_config=quant_args)
    # Load state dict from the correct model_path
    # print(model)
    processed_state_dict, was_unpacked = _load_and_process_state_dict(model_path, torch_dtype)
    # print(processed_state_dict)
    model.load_state_dict(processed_state_dict, strict=False, assign=True)

    if was_unpacked:
        # Packed model: Set binarized flag
        print("INFO: Setting custom modules to binarized inference mode.")
        custom_module_types = (LittleBitLinear, LittleBitRotLinear, BinaryMoSLinear, OneBitLinear)
        for module in model.modules():
            if isinstance(module, custom_module_types):
                module._binarized = True
    else:
        # Legacy unpacked model: Cast all parameters to the target dtype
        print(f"INFO: Casting legacy unpacked model parameters to {torch_dtype}...")
        for param in model.parameters():
            param.data = param.data.to(torch_dtype)
        print("INFO: Legacy model parameters cast.")

    # Cast lm_head unconditionally AFTER potential full-model casting
    print(f"INFO: Casting lm_head to {torch_dtype}...")
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        model.lm_head.to(torch_dtype)

    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    model.to(target_device)
    print(f"Model successfully loaded and moved to {target_device}")

    return model
