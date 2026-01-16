from imagepipe.runtime.models import Yolov10PoseModel

import torch
from PIL import Image

def main():
    model = Yolov10PoseModel.from_pretrained("yolo/v10", use_safetensors=False, dtype=torch.float32)

    opts = {"device" : "CPU", "config" : {"PERFORMANCE_HINT" : "LATENCY"}}
    model = torch.compile(model, backend="openvino", options=opts)

    im = Image.open("assets/example.jpg").convert("RGB")

    with torch.inference_mode():
        pixel_values = model.preprocess(im)
        print("start inference!")
        print(pixel_values.shape)
        prediction, _ = model(pixel_values)
        print("Done!")
    
        results = model.postprocess(prediction)

    print(results)
    # [tensor([[358.8298, 109.3541, 458.2892, 134.8625,   0.9406,   4.0000, 358.7072,
        #  111.8455, 358.8260, 134.6084, 457.5612, 132.9225, 457.8420, 109.5807]])]
    # [[408.5595     122.10835     99.45932     25.5084       0.94064856
    #     4.         358.70724    111.84549    358.82605    134.60844
    #   457.5612     132.92247    457.84198    109.58075   ]]

if __name__ == "__main__":
    main()
