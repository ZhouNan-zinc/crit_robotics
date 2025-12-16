from pathlib import Path

import openvino as ov

class OpenVinoImagePipeline:
    def __init__(
        self, 
        model_path: str|Path|None=None,
        device:str|None="gpu",
        dtype:any=ov.Type.f32,
        batch_size=1,
        height=256,
        width=256,
        num_channels=3,
        **kwargs
    ) -> None:
        super().__init__()
        core = ov.Core()
        model = core.read_model(model_path)
        processor = ov.preprocess.PrePostProcessor(model)
        processor.output().tensor().set_element_type(dtype)
        processor.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
        processor.input().model().set_layout(ov.Layout('NCHW'))
        processor.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout('NHWC'))
        processor.input().tensor().set_shape([batch_size,
                                              height,
                                              width,
                                              num_channels])
        model = processor.build()
        self.model = core.compile_model(model, device)

    def pipe(self, inputs:np.ndarray, **kwargs):
        if inputs.ndim == 3:
            return self.model(inputs[None, ...])
        return self.model(inputs)