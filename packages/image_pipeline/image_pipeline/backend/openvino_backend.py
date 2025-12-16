"""Utility layer around the OpenVINO runtime for YOLO models."""

from pathlib import Path
import json
import numpy as np
import openvino as ov


class OpenVinoBackend:
    """Loads a pre/post-processed OpenVINO model and exposes a call interface."""

    def __init__(
        self, 
        model_path: str | Path | None = None,
        device: str | None = "CPU",
        dtype: str | None = "float32",
        **kwargs
    ) -> None:
        super().__init__()

        model_path = Path(model_path)

        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        input_size = config.get("input_size", None)
        if input_size is None:
            raise ValueError("config.json must contain `input_size`")

        if len(input_size) == 4:
            batch_size = input_size[0]

            if input_size[1] in (1, 3):
                num_channels, height, width = input_size[1:]
                input_layout = ov.Layout("NCHW")
                tensor_layout = ov.Layout("NHWC")
            else:
                height, width, num_channels = input_size[1:]
                input_layout = ov.Layout("NCHW")
                tensor_layout = ov.Layout("NHWC")
        else:
            raise ValueError("input_size must be a 4D shape")

        core = ov.Core()
        model = core.read_model(model_path)

        processor = ov.preprocess.PrePostProcessor(model)

        processor.output().tensor().set_element_type(
            ov.Type.f32 if dtype == "float32" else ov.Type.bf16
        )

        processor.input().tensor() \
            .set_element_type(ov.Type.u8) \
            .set_layout(tensor_layout) \
            .set_shape([batch_size, height, width, num_channels])

        processor.input().preprocess() \
            .resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)

        processor.input().model().set_layout(input_layout)

        model = processor.build()

        self.model = core.compile_model(model, device)

    def __call__(self, inputs: np.ndarray, **kwargs):
        """Run inference on a single image or batch."""
        if inputs.ndim == 3:
            inputs = inputs[None, ...]

        return self.model(inputs)
