import os
import time
from imagepipe.runtime.models import Yolov10PoseModel

import torch
import openvino as ov
import cv2
from PIL import Image

def main():
    model = Yolov10PoseModel.from_pretrained("yolo/v10", use_safetensors=False, dtype=torch.float32).export()

    dummy_inputs = torch.randn((1, 3, 640, 640)).to(
        model.device).to(model.dtype)
        
    for _ in range(2):
        model(dummy_inputs) # dry run

    
    core = ov.Core()
    core.set_property({
        "CACHE_DIR": os.path.expanduser("~/.cache/openvino"),
         "PERFORMANCE_HINT": "LATENCY",
    })
    intermediate_model = ov.convert_model(model, input=[dummy_inputs.shape] ,example_input=dummy_inputs)

    ppp = ov.preprocess.PrePostProcessor(intermediate_model)
    ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(
        ov.Layout("NHWC")).set_color_format(ov.preprocess.ColorFormat.BGR)
    ppp.input().model().set_layout(ov.Layout("NCHW"))
    ppp.input().preprocess().convert_color(ov.preprocess.ColorFormat.RGB).convert_element_type(
        ov.Type.f32).resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR).scale(255.0)
    intermediate_model = ppp.build()

    ov_model = core.compile_model(intermediate_model, device_name="AUTO")
    
    # torchscript_model = torch.compile(model)


    im = cv2.imread("assets/example.jpg")
    im = im[None, :]
    # pixel_values = model.preprocess(im)

    # pixel_values = pixel_values.cpu().numpy()

    cnt = 0

    start_time = time.time()
    
    while True:
    # with torch.inference_mode():
    # print("start inference!")
    # print(pixel_values.shape)
        cnt += 1
        prediction = ov_model([im])[ov_model.output(0)]
    # print(type(prediction))
    # print("Done!")
    
        results = model.postprocess(prediction)
        stop_time = time.time()
        print(results)
        print(cnt / (stop_time-start_time))
    # [tensor([[358.8298, 109.3541, 458.2892, 134.8625,   0.9406,   4.0000, 358.7072,
        #  111.8455, 358.8260, 134.6084, 457.5612, 132.9225, 457.8420, 109.5807]])]
    # [[408.5595     122.10835     99.45932     25.5084       0.94064856
    #     4.         358.70724    111.84549    358.82605    134.60844
    #   457.5612     132.92247    457.84198    109.58075   ]]

if __name__ == "__main__":
    main()
