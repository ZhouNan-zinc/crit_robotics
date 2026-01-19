# """Utility layer around the OpenVINO runtime for YOLO models."""

# from pathlib import Path
# import json
# import numpy as np
# import openvino as ov



# class OpenVinoBackend:
#     """Loads a pre/post-processed OpenVINO model and exposes a call interface."""

#     def __init__(
#         self, 
#         model_path: str | Path | None = None,
#         device: str | None = "CPU",
#         dtype: str | None = "float32",
#         **kwargs
#     ) -> None:
#         super().__init__()

#         model_path = Path(model_path)

#         config_path = model_path / "config.json"
#         if not config_path.exists():
#             raise FileNotFoundError(f"config.json not found in {model_path}")

#         with open(config_path, "r", encoding="utf-8") as f:
#             config = json.load(f)

#         input_size = config.get("input_size", None)
#         if input_size is None:
#             raise ValueError("config.json must contain `input_size`")

#         if len(input_size) == 4:
#             batch_size = input_size[0]

#             if input_size[1] in (1, 3):
#                 num_channels, height, width = input_size[1:]
#                 input_layout = ov.Layout("NCHW")
#                 tensor_layout = ov.Layout("NHWC")
#             else:
#                 height, width, num_channels = input_size[1:]
#                 input_layout = ov.Layout("NCHW")
#                 tensor_layout = ov.Layout("NHWC")
#         else:
#             raise ValueError("input_size must be a 4D shape")

#         core = ov.Core()
#         model = core.read_model(model_path)

#         processor = ov.preprocess.PrePostProcessor(model)

#         processor.output().tensor().set_element_type(
#             ov.Type.f32 if dtype == "float32" else ov.Type.bf16
#         )

#         processor.input().tensor() \
#             .set_element_type(ov.Type.u8) \
#             .set_layout(tensor_layout) \
#             .set_shape([batch_size, height, width, num_channels])

#         processor.input().model().set_layout(input_layout)

#         model = processor.build()

#         self.model = core.compile_model(model, device)

#     def __call__(self, inputs, **kwargs):
#         """Run inference on a single image or batch."""

#         return self.model(inputs)


# from pathlib import Path

# from .yolo_end2end import OpenVinoBackend


# class AutoModel:
#     def __init__(self):
#         pass

#     @classmethod
#     def from_pretrained(
#         self, 
#         model_path: str | Path | None = None,
#         device: str | None = "CPU",
#         dtype: str | None = "float32",
#         **kwargs
#     ):
#         if model_path.endswith((".xml", ".bin")):
#             return OpenVinoBackend(
#                 model_path,
#                 device,
#                 dtype,
#                 **kwargs
#             )
#         else:
#             raise ValueError("Unsupported type.")

# import os
# from os import PathLike
# from torch import Tensor
# from pathlib import Path
# import json
# import torch
# from ...hub import Yolov10Pose
# import torch
# import torch.nn as nn
# ROOT_PATH = Path(__file__).resolve().parent.parent.parent

# class AutoModel:
#     def __init__(self):
#         pass

#     @classmethod
#     def from_pretrained(
#         self, 
#         pretrained_model_name_or_path: str|PathLike,
#         model_args = None,
#         config = None,
#         state_dict: dict[str, Tensor]|None = None,
#         cache_dir: str|PathLike|None = None,
#         force_download: bool = False,
#         proxies: dict[str, str]|None = None,
#         output_loading_info: bool = False,
#         local_files_only: bool = False,
#         revision: str = "main",
#         trust_remote_code: bool = False,
#         code_revision: str = "main",
#         **kwargs
#     ) -> nn.Module:
#         with open(model_path / "config.json", "r", encoding="utf-8") as f:
#             config = json.load(f)
        
#         if kwargs.get("device_map", "auto") == "auto":
#             map_location = "cpu"

#         model = Yolov10Pose(**config, **model_args) if model_args else Yolov10Pose(**config)
#         model.eval()

#         state_dict = state_dict or torch.load(
#             model_path / "pytorch_model.bin",
#             map_location=map_location,
#             weights_only=True,
#         )

#         len_updated_state_dict = sum(
#             (k in model.state_dict()) and (model.state_dict()[k].shape == v.shape)
#             for k, v in state_dict.items()
#         )

#         print(f"Transferred {len_updated_state_dict}/{len(model.state_dict())} items from pretrained weights\n")
#         model.load_state_dict(state_dict)

#         return model

#     @classmethod
#     def from_config(self, **kwargs):
#         raise NotImplementedError()

#     @classmethod
#     def register(self, config, model, exist_ok=False):
#         pass
