import collections
import copy
import fnmatch
import os
import re
from collections import defaultdict
from itertools import cycle
from typing import Any, Iterator, TypeVar, get_type_hints
from zipfile import is_zipfile
from contextlib import ExitStack

import torch
import torch.nn as nn
from safetensors import safe_open

from .utils import logging
from .utils import (
    DUMMY_INPUTS,
    DUMMY_MASK
)
from .utils import (
    log_state_dict_report,
    local_torch_dtype,
    ContextManagers,
    get_resolved_checkpoint_files
) 
from .configuration_utils import PreTrainedConfig

logger = logging.getLogger(__name__)

SpecificPreTrainedModelType = TypeVar("SpecificPreTrainedModelType", bound="PreTrainedModel")

str_to_torch_dtype = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _materialize_state_dict_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "to_tensor"):
        return value.to_tensor()
    if hasattr(value, "get_tensor"):
        return value.get_tensor()
    try:
        return torch.as_tensor(value)
    except Exception:
        return torch.tensor(value)


def _resolve_device_for_param(device_map: object, param_name: str) -> object:
    if device_map is None:
        return None
    if isinstance(device_map, dict):
        best_key = None
        best_value = None
        for key, value in device_map.items():
            if key == "" or param_name == key or param_name.startswith(f"{key}."):
                if best_key is None or len(key) > len(best_key):
                    best_key = key
                    best_value = value
        return best_value
    return device_map


def _resolve_dtype_for_param(
    param_name: str,
    dtype: torch.dtype | None,
    dtype_plan: dict[str, torch.dtype],
    param_tensor: torch.Tensor | None,
) -> torch.dtype | None:
    for pattern, plan_dtype in dtype_plan.items():
        if fnmatch.fnmatch(param_name, pattern):
            return plan_dtype
    if dtype is not None:
        return dtype
    if isinstance(param_tensor, torch.Tensor):
        return param_tensor.dtype
    return None


def _build_name_to_target(model: "PreTrainedModel") -> dict[str, tuple[nn.Module, str, bool]]:
    name_to_target: dict[str, tuple[nn.Module, str, bool]] = {}
    for module_name, module in model.named_modules():
        for name, param in module._parameters.items():
            if param is None:
                continue
            full_name = f"{module_name}.{name}" if module_name else name
            name_to_target[full_name] = (module, name, True)
        for name, buf in module._buffers.items():
            if buf is None:
                continue
            full_name = f"{module_name}.{name}" if module_name else name
            name_to_target[full_name] = (module, name, False)
    return name_to_target


def _get_dtype(
    dtype: str | torch.dtype | dict | None,
    checkpoint_files: list[str] | None,
    config: PreTrainedConfig,
    sharded_metadata: dict | None,
    state_dict: dict | None,
    weights_only: bool,
) -> tuple[PreTrainedConfig, torch.dtype]:
    """Find the correct `dtype` to use based on provided arguments. Also update the `config` based on the
    inferred dtype. We do the following:
    1. If dtype is "auto", we try to read the config, else auto-detect dtype from the loaded state_dict, by checking
    its first weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
    2. Else, use the dtype provided as a dict or str
    """
    is_sharded = sharded_metadata is not None

    if dtype is not None:
        if isinstance(dtype, str):
            if dtype == "auto":
                if hasattr(config, "dtype") and config.dtype is not None:
                    dtype = config.dtype
                    logger.info(f"Will use dtype={dtype} as defined in model's config object")
                else:
                    if is_sharded and "dtype" in sharded_metadata:
                        dtype = sharded_metadata["dtype"]
                    elif state_dict is not None:
                        dtype = get_state_dict_dtype(state_dict)
                    else:
                        state_dict = load_state_dict(
                            checkpoint_files[0], map_location="meta", weights_only=weights_only
                        )
                        dtype = get_state_dict_dtype(state_dict)
                    logger.info(
                        f"Since the `dtype` attribute can't be found in model's config object, "
                        f"will use dtype={dtype} as derived from model's weights"
                    )
            elif hasattr(torch, dtype):
                dtype = getattr(torch, dtype)
            else:
                raise ValueError(
                    "`dtype` provided as a `str` can only be `'auto'`, or a string representation of a valid `torch.dtype`"
                )

            # cast it to a proper `torch.dtype` object
            dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        elif not isinstance(dtype, (dict, torch.dtype)):
            raise ValueError(
                f"`dtype` can be one of: `torch.dtype`, `'auto'`, a string of a valid `torch.dtype` or a `dict` with valid `dtype` "
                f"for each sub-config in composite configs, but received {dtype}"
            )
    else:
        # set torch.get_default_dtype() (usually fp32) as the default dtype if `None` is provided
        dtype = torch.get_default_dtype()

    # Set it on the config and subconfigs
    config.dtype = dtype
    for sub_config_key in config.sub_configs:
        if (sub_config := getattr(config, sub_config_key)) is not None:
            sub_config.dtype = dtype

    return config, dtype


def get_torch_context_manager_or_global_device():
    """
    Test if a device context manager is currently in use, or if it is not the case, check if the default device
    is not "cpu". This is used to infer the correct device to load the model on, in case `device_map` is not provided.
    """
    device_in_context = torch.tensor([]).device
    default_device = torch.get_default_device()
    # This case means no context manager was used -> we still check if the default that was potentially set is not cpu
    if device_in_context == default_device:
        if default_device != torch.device("cpu"):
            return default_device
        return None
    return device_in_context


def get_state_dict_dtype(state_dict):
    """
    Returns the first found floating dtype in `state_dict` if there is one, otherwise returns the first dtype.
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype

    # if no floating dtype was found return whatever the first dtype is
    if len(state_dict) == 0:
        return torch.float32
    return next(iter(state_dict.values())).dtype


def load_state_dict(
    checkpoint_file: str | os.PathLike, map_location: str | torch.device = "cpu", weights_only: bool = True
) -> dict[str, torch.Tensor]:
    """
    Reads a `safetensor` or a `.bin` checkpoint file. We load the checkpoint on "cpu" by default.
    """
    # Use safetensors if possible
    if checkpoint_file.endswith(".safetensors"):
        with safe_open(checkpoint_file, framework="pt") as f:
            state_dict = {}
            for k in f.keys():
                if map_location == "meta":
                    _slice = f.get_slice(k)
                    k_dtype = _slice.get_dtype()
                    if k_dtype in str_to_torch_dtype:
                        dtype = str_to_torch_dtype[k_dtype]
                    else:
                        raise ValueError(f"Cannot load safetensors of unknown dtype {k_dtype}")
                    state_dict[k] = torch.empty(size=_slice.get_shape(), dtype=dtype, device="meta")
                else:
                    state_dict[k] = f.get_tensor(k).to(map_location)
            return state_dict

    extra_args = {}
    # mmap can only be used with files serialized with zipfile-based format.
    if isinstance(checkpoint_file, str) and map_location != "meta" and is_zipfile(checkpoint_file):
        extra_args = {"mmap": True}

    return torch.load(checkpoint_file, map_location=map_location, weights_only=weights_only, **extra_args)


def convert_and_load_state_dict_in_model(
    model: "PreTrainedModel",
    state_dict: dict[str, object],
    dtype: torch.dtype | None = None,
    device_map: dict | None = None,
    dtype_plan: dict[str, torch.dtype] | None = None,
):
    """
    Minimal state dict loader with dtype and device placement support.
    """
    if device_map == "disk" or (isinstance(device_map, dict) and "disk" in device_map.values()):
        raise ValueError("disk offload is not supported in this minimal loader")

    device_map = device_map or {"": "cpu"}
    dtype_plan = dtype_plan or {}

    model_state_dict = model.state_dict()
    missing_keys = set(model_state_dict.keys())
    unexpected_keys: set[str] = set()
    mismatched_keys: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    conversion_errors: dict = {}

    name_to_target = _build_name_to_target(model)

    for key, value in state_dict.items():
        if key not in model_state_dict:
            unexpected_keys.add(key)
            continue

        missing_keys.discard(key)

        target = name_to_target.get(key)
        if target is None:
            unexpected_keys.add(key)
            continue

        module, attr, is_param = target
        param_tensor = module._parameters[attr] if is_param else module._buffers[attr]
        loaded_tensor = _materialize_state_dict_tensor(value)

        target_dtype = _resolve_dtype_for_param(key, dtype, dtype_plan, param_tensor)
        if target_dtype is not None and loaded_tensor.is_floating_point():
            loaded_tensor = loaded_tensor.to(dtype=target_dtype)

        target_device = _resolve_device_for_param(device_map, key)
        if target_device is None and isinstance(param_tensor, torch.Tensor):
            target_device = param_tensor.device
        if target_device is None or (
            isinstance(target_device, torch.device) and target_device.type == "meta"
        ):
            target_device = torch.device("cpu")

        if isinstance(target_device, int):
            target_device = torch.device("cuda", target_device)
        elif isinstance(target_device, str):
            target_device = torch.device(target_device)

        loaded_tensor = loaded_tensor.to(device=target_device)

        if isinstance(param_tensor, torch.Tensor) and param_tensor.shape != loaded_tensor.shape:
            mismatched_keys.append((key, tuple(param_tensor.shape), tuple(loaded_tensor.shape)))
            continue

        if is_param:
            requires_grad = param_tensor.requires_grad if isinstance(param_tensor, torch.Tensor) else False
            module._parameters[attr] = nn.Parameter(loaded_tensor, requires_grad=requires_grad)
        else:
            module._buffers[attr] = loaded_tensor
            
    return missing_keys, unexpected_keys, mismatched_keys, conversion_errors



def expand_device_map(device_map: dict | None, param_names: list[str]):
    """
    Expand a device map to return the correspondence parameter name to device.
    """
    if device_map is None:
        return dict.fromkeys(param_names, "cpu")

    # Here, we first sort by number of submodules, then length of the full string, to make sure to match correctly
    device_map_regex = re.compile(
        "|".join(rf"({k})" for k in sorted(device_map.keys(), key=lambda x: (x.count("."), len(x)), reverse=True))
    )
    new_device_map = {}
    for param in param_names:
        device_match = device_map_regex.match(param)
        new_device_map[param] = device_map[device_match.group()] if device_match else device_map.get("", "cpu")

    return new_device_map

def get_module_from_name(module, tensor_name: str) -> tuple[Any, str]:
    if "." in tensor_name:
        module_name, tensor_name = tensor_name.rsplit(".", 1)
        module = module.get_submodule(module_name)
    return module, tensor_name

def load_parameter_into_model(model: "PreTrainedModel", param_name: str, tensor: torch.Tensor):
    """Cast a single parameter or buffer `param_name` into the `model`, with value `tensor`."""
    parent, param_type = get_module_from_name(model, param_name)
    if param_type in parent._parameters and not isinstance(tensor, nn.Parameter):
        tensor = nn.Parameter(tensor, requires_grad=tensor.is_floating_point())
    # We need to use setattr here, as we set non-persistent buffers as well with this function (`load_state_dict`
    # does not allow to do it)
    setattr(parent, param_type, tensor)


def get_device(device_map: dict | None, param_name: str, valid_torch_device: bool = False) -> torch.device | str | int:
    """Return the device on which `param_name` should be according to the `device_map`. If `valid_torch_device` is `True`,
    then if the device is `"disk"`, `"cpu"` will be returned instead."""
    device = expand_device_map(device_map, [param_name])[param_name]
    if valid_torch_device and device == "disk":
        return "cpu"
    return device

def get_total_byte_count(
    model: "PreTrainedModel", accelerator_device_map: dict
):
    """
    This utility function calculates the total bytes count needed to load the model on each device.
    This is useful for caching_allocator_warmup as we want to know how much cache we need to pre-allocate.
    """

    total_byte_count = defaultdict(lambda: 0)
    tied_param_names = model.all_tied_weights_keys.keys()

    for param_name, device in accelerator_device_map.items():
        # Skip if the parameter has already been accounted for (tied weights)
        if param_name in tied_param_names:
            continue

        param = model.get_parameter_or_buffer(param_name)

        dtype_size = param.element_size()

        param_byte_count = param.numel() * dtype_size

        total_byte_count[device] += param_byte_count
    return total_byte_count


def caching_allocator_warmup(model: "PreTrainedModel", expanded_device_map: dict):
    """This function warm-ups the caching allocator based on the size of the model tensors that will reside on each
    device. It allows to have one large call to Malloc, instead of recursively calling it later when loading
    the model, which is actually the loading speed bottleneck.
    Calling this function allows to cut the model loading time by a very large margin.

    A few facts related to loading speed (taking into account the use of this function):
    - When loading a model the first time, it is usually slower than the subsequent times, because the OS is very likely
    to cache the different state dicts (if enough resources/RAM are available)
    - Trying to force the OS to cache the files in advance (by e.g. accessing a small portion of them) is really hard,
    and not a good idea in general as this is low level OS optimizations that depend on resource usage anyway
    - As of 18/03/2025, loading a Llama 70B model with TP takes ~1 min without file cache, and ~13s with full file cache.
    The baseline, i.e. only loading the tensor shards on device and adjusting dtype (i.e. copying them) is ~5s with full cache.
    These numbers are reported for TP on 4 H100 GPUs.
    - It is useless to pre-allocate more than the model size in this function (i.e. using an `allocation_factor` > 1) as
    cudaMalloc is not a bottleneck at all anymore
    - Loading speed bottleneck is now almost only tensor copy (i.e. changing the dtype) and moving the tensors to the devices.
    However, we cannot really improve on those aspects obviously, as the data needs to be moved/copied in the end.
    """
    # Remove disk, cpu and meta devices, and cast to proper torch.device

    accelerator_device_map = {
        param: torch.device(device) for param, device in expanded_device_map.items()
    }

    if not accelerator_device_map:
        return

    total_byte_count = get_total_byte_count(model, accelerator_device_map)

    # This will kick off the caching allocator to avoid having to Malloc afterwards
    for device, byte_count in total_byte_count.items():
        if device.type in ["cuda", "xpu"]:
            torch_accelerator_module = getattr(torch, device.type)
            index = device.index if device.index is not None else torch_accelerator_module.current_device()
            device_memory = torch_accelerator_module.mem_get_info(index)[0]
            # Allow up to (max device memory - 1.2 GiB) in resource-constrained hardware configurations. Trying to reserve more
            # than that amount might sometimes lead to unnecessary cuda/xpu OOM, if the last parameter to be loaded on the device is large,
            # and the remaining reserved memory portion is smaller than the param size -> torch will then try to fully re-allocate all
            # the param size, instead of using the remaining reserved part, and allocating only the difference, which can lead
            # to OOM. See https://github.com/huggingface/transformers/issues/37436#issuecomment-2808982161 for more details.
            # Note that we use an absolute value instead of device proportion here, as a 8GiB device could still allocate too much
            # if using e.g. 90% of device size, while a 140GiB device would allocate too little
            byte_count = min(byte_count, max(0, int(device_memory - 1.2 * 1024**3)))
            # If there is *unused* reserved cuda/xpu memory, we can skip/reduce the allocation.
            unused_memory = torch_accelerator_module.memory_reserved(
                index
            ) - torch_accelerator_module.memory_allocated(index)
            byte_count = int(max(0, byte_count - unused_memory))
        # We divide by 2 here as we allocate in fp16
        _ = torch.empty(byte_count // 2, dtype=torch.float16, device=device, requires_grad=False)


class PreTrainedModel(nn.Module):
    r"""
    Base class for all models.

    [`PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading models.
    """
    
    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    model_tags = None
    dtype_plan: dict[str, torch.dtype] | None = None
    input_modalities: str | list[str] = "text"  # most models are text

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # For BC we keep the original `config_class` definition in case
        # there is a `config_class` attribute (e.g. remote code models),
        # otherwise we derive it from the annotated `config` attribute.

        # defined in this particular subclass
        child_annotation = cls.__dict__.get("__annotations__", {}).get("config", None)
        child_attribute = cls.__dict__.get("config_class", None)

        # defined in the class (this subclass or any parent class)
        full_annotation = get_type_hints(cls).get("config", None)
        full_attribute = cls.config_class

        # priority (child class_config -> child annotation -> global class_config -> global annotation)
        if child_attribute is not None:
            cls.config_class = child_attribute
        elif child_annotation is not None:
            cls.config_class = child_annotation
        elif full_attribute is not None:
            cls.config_class = full_attribute
        elif full_annotation is not None:
            cls.config_class = full_annotation

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise TypeError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config


    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return next(param.device for param in self.parameters())

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return next(param.dtype for param in self.parameters() if param.is_floating_point())

    @property
    def dummy_inputs(self) -> dict[str, torch.Tensor]:
        """
        `dict[str, torch.Tensor]`: Dummy inputs to do a forward pass in the network.
        """
        return {"input_ids": torch.tensor(DUMMY_INPUTS)}
    
    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]

        total_params = 0
        for name, param in self.named_parameters():
            if exclude_embeddings and name in embedding_param_names:
                continue
            if param.requires_grad or not only_trainable:
                total_params += param.numel()

        return total_params
    
    def named_non_persistent_buffers(
        self, recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Similar to `named_buffers`, but only yield non-persistent ones. It is handy as it's not perfectly straightforward
        to know if they are persistent or not"""
        for name, tensor in self.named_buffers(recurse=recurse, remove_duplicate=remove_duplicate):
            # We have to grab the parent here, as the attribute `_non_persistent_buffers_set` is on the immediate
            # parent only
            parent, buf_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = self.get_submodule(parent)
            if buf_name in parent._non_persistent_buffers_set:
                yield name, tensor


    def half(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.half()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().half(*args)

    def float(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.float()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().float(*args)

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            dtype (`torch.dtype`, *optional*):
                Override the default `dtype` and load the model under this dtype.
        """
        # For BC on the old `torch_dtype`
        dtype = kwargs.pop("dtype", config.dtype)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        init_contexts = []
        if dtype is not None:
            init_contexts.append(local_torch_dtype(dtype, cls.__name__))

        # Instantiate the model
        with ContextManagers(init_contexts):
            model = cls(config, **kwargs)

        return model

    @torch.no_grad()
    def _init_weights(self, module):
        """
        Initialize the weights. This is quite general on purpose, in the spirit of what we usually do. For more complex
        initialization scheme, it should be overridden by the derived `PreTrainedModel` class. In case a model adds an explicit
        `nn.Parameter`, this method should also be overridden in order to initialize it correctly.
        """
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range or 0.02
        elif hasattr(self.config, "init_std"):
            std = self.config.init_std
        elif hasattr(self.config, "initializer_factor"):
            std = self.config.initializer_factor
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            if getattr(module, "weight", None) is not None:
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # Here we need the check explicitly, as we slice the weight in the `zeros_` call, so it looses the flag
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                torch.nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.MultiheadAttention):
            # This uses torch's original init
            module._reset_parameters()
        # We cannot use `isinstance` on the RMSNorms or LayerNorms, as they usually are custom modules which change names
        # between modelings (because they are prefixed with the model name)
        elif (
            isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            or "LayerNorm" in module.__class__.__name__
            or "RMSNorm" in module.__class__.__name__
        ):
            # Norms can exist without weights (in which case they are None from torch primitives)
            if getattr(module, "weight", None) is not None:
                torch.nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)
            # And the potential buffers for the BatchNorms
            if getattr(module, "running_mean", None) is not None:
                torch.nn.init.zeros_(module.running_mean)
                torch.nn.init.ones_(module.running_var)
                torch.nn.init.zeros_(module.num_batches_tracked)

    def _initialize_weights(self, module):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_hf_initialized", False):
            return

        self._init_weights(module)
        module._is_hf_initialized = True

    @torch.no_grad()
    def initialize_weights(self):
        """
        This is equivalent to calling `self.apply(self._initialize_weights)`, but correctly handles composite models.
        This function dynamically dispatches the correct `init_weights` function to the modules as we advance in the
        module graph along the recursion. It can handle an arbitrary number of sub-models. Without it, every composite
        model would have to recurse a second time on all sub-models explicitly in the outer-most `_init_weights`, which
        is extremely error prone and inefficient.
        """
        if not hasattr(torch.nn.Module, "smart_apply"):
            # This function is equivalent to `torch.nn.Module.apply`, except that it dynamically adjust the function
            # to apply as we go down the graph
            def smart_apply(self, fn):
                for module in self.children():
                    # We found a sub-model: recursively dispatch its own init function now!
                    if isinstance(module, PreTrainedModel):
                        module.smart_apply(module._initialize_weights)
                    else:
                        module.smart_apply(fn)
                fn(self)
                return self

            torch.nn.Module.smart_apply = smart_apply

        # Let the magic happen with this simple call
        self.smart_apply(self._initialize_weights)

    def init_weights(self):
        """
        Maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # If we are initializing on meta device, there is no point in trying to run inits
        if get_torch_context_manager_or_global_device() != torch.device("meta"):
            # Initialize weights
            self.initialize_weights()


    @classmethod
    def from_pretrained(
        cls: type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        use_safetensors: bool | None = True,
        weights_only: bool = True,
        **kwargs,
    ) -> SpecificPreTrainedModelType:
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PreTrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PreTrainedConfig`],
                    - a string or path valid as input to [`~PreTrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (`dict[str, torch.Tensor]`, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.

            > Parameters for big model inference

            dtype (`str` or `torch.dtype`, *optional*, defaults to `"auto"`):
                Override the default `torch_dtype` and load the model under a specific `dtype`. The different options
                are:

                1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified
                  `dtype`, ignoring the model's `config.dtype` if one exists. If not specified
                  - the model will get loaded in `torch.float` (fp32).
                2. `"auto"` - A `dtype` or `torch_dtype` entry in the `config.json` file of the model will be
                  attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
                  the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
                  using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
                  the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.
                3. A string that is a valid `torch.dtype`. E.g. "float32" loads the model in `torch.float32`, "float16" loads in `torch.float16` etc.

            device_map (`str` or `dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
                like `1`) on which the model will be allocated, the device map will map the entire model to this
                device. Passing `device_map = 0` means put the whole model on GPU 0.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`
                is not installed, it will be set to `False`.
            weights_only (`bool`, *optional*, defaults to `True`):
                Indicates whether unpickler should be restricted to loading only tensors, primitive types,
                dictionaries and any types added via torch.serialization.add_safe_globals().
                When set to False, we can load wrapper tensor subclass weights.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model 
                Behaves differently depending on whether a `config` is provided or automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PreTrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.
        """
        state_dict = kwargs.pop("state_dict", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        dtype = kwargs.pop("dtype", None)
        device_map = kwargs.pop("device_map", None)
        variant = kwargs.pop("variant", None)

        if dtype is None:
            dtype = "auto"

        if device_map == "auto" and int(os.environ.get("WORLD_SIZE", "0")):
            logger.info(
                "You've set device_map=`auto` while triggering a distributed run with torchrun. This might lead to unexpected behavior. "
                "If your plan is to load the model on each device, you should set device_map={"
                ": PartialState().process_index} where PartialState comes from accelerate library"
            )

        # Load config if we don't provide a configuration
        if not isinstance(config, PreTrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                return_unused_kwargs=True,
                **kwargs,
            )
        else:
            config = copy.deepcopy(config)
            model_kwargs = kwargs

        checkpoint_files, sharded_metadata = get_resolved_checkpoint_files(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            variant=variant,
            use_safetensors=use_safetensors,
            transformers_explicit_filename=getattr(config, "transformers_weights", None),
        )
        
        # Find the correct dtype based on current state
        config, dtype = _get_dtype(
            dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only
        )

        config.name_or_path = pretrained_model_name_or_path
        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        model = cls(config, *model_args, **model_kwargs)

        # Finalize model weight initialization
        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
            model,
            state_dict,
            checkpoint_files,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            sharded_metadata=sharded_metadata,
            device_map=device_map,
            dtype=dtype,
            weights_only=weights_only,
        )

        model.eval()  # Set model in evaluation mode by default

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info
        return model

    def _move_missing_keys_from_meta_to_device(
        self,
        missing_keys: list[str],
        device_map: dict | None,
    ) -> None:
        """Move the missing keys (keys that are part of the model` parameters, but were NOT found in the loaded state dicts)
        back from meta device to their device according to the `device_map` if any, els`e cpu. Takes care of sharding those
        missing parameters if `device_mesh` is provided, i.e. we are using TP.
        All non-persistent buffers are also moved back to the correct device (they are not part of the state_dict, but are
        not missing either).
        """

        # The tied weight keys are in the "missing" usually, but they should not be moved (they will be tied anyway)
        # This is especially important because if they are moved, they will lose the `_is_hf_initialized` flag, and they
        # will be re-initialized for nothing (which can be quite long)
        for key in missing_keys:
            param = self.get_parameter_or_buffer(key)
            param_device = get_device(device_map, key, valid_torch_device=True)
            value = torch.empty_like(param, device=param_device)
            load_parameter_into_model(self, key, value)

        # We need to move back non-persistent buffers as well, as they are not part of loaded weights anyway
        for key, buffer in self.named_non_persistent_buffers():
            buffer_device = get_device(device_map, key, valid_torch_device=True)
            value = torch.empty_like(buffer, device=buffer_device)
            load_parameter_into_model(self, key, value)


    def _adjust_missing_and_unexpected_keys(
        self, missing_keys: set[str], unexpected_keys: set[str]
    ) -> tuple[set[str], set[str]]:
        """Adjust the `missing_keys` and `unexpected_keys` based on current model's exception rules, to avoid
        raising unneeded warnings/errors.
        """
        # Old checkpoints may have keys for rotary_emb.inv_freq forach layer, however we moved this buffer to the main model
        # (so the buffer name has changed). Remove them in such a case. This is another exception that was not added to
        # `_keys_to_ignore_on_load_unexpected` as it touches many models -> we add it manually to the existing patterns
        has_inv_freq_buffers = any(buffer.endswith("rotary_emb.inv_freq") for buffer, _ in self.named_buffers())
        additional_unexpected_patterns = [r"rotary_emb\.inv_freq"] if has_inv_freq_buffers else []

        missing_patterns = []
        unexpected_patterns = ([]) + additional_unexpected_patterns
        ignore_missing_regex, ignore_unexpected_regex = None, None
        if len(missing_patterns) > 0:
            ignore_missing_regex = re.compile("|".join(rf"({pattern})" for pattern in missing_patterns))
        if len(unexpected_patterns) > 0:
            ignore_unexpected_regex = re.compile("|".join(rf"({pattern})" for pattern in unexpected_patterns))

        # Clean-up missing keys
        if ignore_missing_regex is not None:
            missing_keys = {key for key in missing_keys if ignore_missing_regex.search(key) is None}

        # Clean-up unexpected keys
        if ignore_unexpected_regex is not None:
            unexpected_keys = {key for key in unexpected_keys if ignore_unexpected_regex.search(key) is None}

        return missing_keys, unexpected_keys

    @classmethod
    def _load_pretrained_model(
        cls,
        model: "PreTrainedModel",
        state_dict: dict | None,
        checkpoint_files: list[str] | None,
        pretrained_model_name_or_path: str | None,
        ignore_mismatched_sizes: bool = False,
        sharded_metadata: dict | None = None,
        device_map: dict | None = None,
        dtype: torch.dtype | None = None,
        weights_only: bool = True,
    ):        
        # Initialize
        model.init_weights()

        # Model's definition arriving here is final (TP hooks added, quantized layers replaces)
        expected_keys = list(model.state_dict().keys())

        # Warmup cuda to load the weights much faster on devices
        if device_map is not None:
            expanded_device_map = expand_device_map(device_map, expected_keys)
            caching_allocator_warmup(model, expanded_device_map)

        error_msgs = []

        with ExitStack() as stack:
            # Checkpoints are safetensors
            if checkpoint_files is not None and checkpoint_files[0].endswith(".safetensors"):
                merged_state_dict = {}
                for file in checkpoint_files:
                    f = stack.enter_context(safe_open(file, framework="pt", device="cpu"))
                    for k in f.keys():
                        merged_state_dict[k] = f.get_slice(k)  # lazy slice

            # User passed an explicit state_dict
            elif state_dict is not None:
                merged_state_dict = state_dict

            # Checkpoints are .bin
            elif checkpoint_files is not None:
                merged_state_dict = {}
                for ckpt_file in checkpoint_files:
                    merged_state_dict.update(load_state_dict(ckpt_file))

            else:
                raise ValueError("Neither a state dict nor checkpoint files were found.")

            missing_keys, unexpected_keys, mismatched_keys, conversion_errors = (
                convert_and_load_state_dict_in_model(
                    model=model,
                    state_dict=merged_state_dict,
                    dtype=dtype,
                    device_map=device_map,
                    dtype_plan=model.dtype_plan,
                )
            )

        # Adjust missing and unexpected keys
        missing_keys, unexpected_keys = model._adjust_missing_and_unexpected_keys(missing_keys, unexpected_keys)

        log_state_dict_report(
            model=model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            logger=logger,
            error_msgs=error_msgs,
            unexpected_keys=unexpected_keys,
            missing_keys=missing_keys,
            mismatched_keys=mismatched_keys,
            mismatched_shapes=mismatched_keys,
            conversion_errors=conversion_errors,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs