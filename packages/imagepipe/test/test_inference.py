from imagepipe.runtime.models import Yolov10PoseModel

import torch
import openvino as ov
from PIL import Image

from copy import deepcopy

def main():
    model = Yolov10PoseModel.from_pretrained("yolo/v10", use_safetensors=False, dtype=torch.float32)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    
    model = model.export()

    # torchscript_model = torch.compile(model)


    im = Image.open("assets/example.jpg").convert("RGB")

    pixel_values = model.preprocess(im)

    print(pixel_values.shape)


    _ = model(pixel_values)

    # opts = {"device" : "CPU", "config" : {"PERFORMANCE_HINT" : "LATENCY"}}
    # model = torch.compile(model, backend="openvino", options=opts)

    intermediate_model = ov.convert_model(model, input=[pixel_values.shape], example_input=pixel_values)
    

    print("Maybe We have converted???")
    # ov_model = ov.compile_model(intermediate_model, "CPU")

    # pixel_values = pixel_values.cpu().numpy()

    # while True:
    #     with torch.inference_mode():
    #         print("start inference!")
    #         # print(pixel_values.shape)
    #         prediction = ov_model([pixel_values])
    #         # print(type(prediction))
    #         print("Done!")
    
    # results = model.postprocess(prediction)

    # print(results)
    # [tensor([[358.8298, 109.3541, 458.2892, 134.8625,   0.9406,   4.0000, 358.7072,
        #  111.8455, 358.8260, 134.6084, 457.5612, 132.9225, 457.8420, 109.5807]])]
    # [[408.5595     122.10835     99.45932     25.5084       0.94064856
    #     4.         358.70724    111.84549    358.82605    134.60844
    #   457.5612     132.92247    457.84198    109.58075   ]]

if __name__ == "__main__":
    main()
