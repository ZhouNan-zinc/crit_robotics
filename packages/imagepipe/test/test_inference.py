from imagepipe.runtime.models import Yolov10PoseModel

import torch
from PIL import Image
from torchvision.transforms import functional as TF

def main():
    model = Yolov10PoseModel.from_pretrained("yolo/v10", use_safetensors=False, dtype=torch.float32).cuda()
    im = Image.open("assets/example.jpg").convert("RGB")

    inputs = TF.to_tensor(im).unsqueeze(0).to(model.device).to(model.dtype)  # [1,3,H,W] float32 0~1


    with torch.inference_mode():
        print("start inference!")
        prediction, _ = model(inputs)
        print("Done!")
    
    results = model.postprocess(prediction)

    print(results[0])
    # [tensor([[358.8298, 109.3541, 458.2892, 134.8625,   0.9406,   4.0000, 358.7072,
        #  111.8455, 358.8260, 134.6084, 457.5612, 132.9225, 457.8420, 109.5807]])]

if __name__ == "__main__":
    main()
