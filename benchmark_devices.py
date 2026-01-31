import os
import torch
import openvino as ov
import sys
import numpy as np
import time

sys.path.append(os.path.join(os.getcwd(), "packages/imagepipe"))

from imagepipe.runtime.models import Yolov10PoseModel

model_path = "install/imagepipe/share/imagepipe/yolo/v10"

# Setup model and test only 640x640 as requested
try:
    print("Loading model for 640x640 benchmark...")
    model_pt = Yolov10PoseModel.from_pretrained(
        model_path,
        use_safetensors=False,
        weights_only=True,
        dtype=torch.float32
    ).export()

    # prepare dummy input 640x640
    dummy_inputs = torch.randn((1, 3, 640, 640)).to(torch.float32)
    model_pt(dummy_inputs)  # dry run

    intermediate_model = ov.convert_model(model_pt, input=[dummy_inputs.shape], example_input=dummy_inputs)
    core = ov.Core()

    n_warmup = 10
    n_runs = 50
    input_data = dummy_inputs.detach().numpy()

    def run_bench(name, device, config=None):
        print(f"\nBenchmarking {name}...")
        try:
            compiled = core.compile_model(intermediate_model, device_name=device, config=config)
            req = compiled.create_infer_request()

            for _ in range(n_warmup):
                req.infer([input_data])

            start = time.time()
            for _ in range(n_runs):
                req.infer([input_data])
            end = time.time()

            avg = ((end - start) / n_runs) * 1000
            out = list(req.infer([input_data]).values())[0]
            has_nan = np.isnan(out).any()
            print(f"{name}: {avg:.2f} ms | NaN: {has_nan}")
            return avg, has_nan
        except Exception as e:
            print(f"{name} failed: {e}")
            return None, None

    # Run CPU benchmark
    run_bench("CPU", "CPU", {"INFERENCE_NUM_THREADS": "8"})

    # Run GPU default (likely FP16) and FP32 hint if GPU available
    if "GPU" in core.available_devices:
        run_bench("GPU (Default)", "GPU", {})
        run_bench("GPU (FP32Hint)", "GPU", {"INFERENCE_PRECISION_HINT": "f32"})
    else:
        print("No GPU available on this system.")

except Exception as e:
    print("Benchmark failed:", e)
