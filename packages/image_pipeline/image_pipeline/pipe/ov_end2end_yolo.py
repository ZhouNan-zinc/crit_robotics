import os

from ..ops import end2end_fastnms, pose_estimate
from ..node import ImagePipelineNodeInterface
from ..backend import OpenVinoImagePipeline
from ..tracking import ByteTrack

class OpenVinoEnd2endYolo(OpenVinoImagePipeline, ImagePipelineNodeInterface):
    def __init__(self):
        model_name_or_path = self.get_parameter_or("model_name_or_path", "namespace/model")
        if os.path.exists(model_name_or_path):
            model_path = model_name_or_path
        else:
            model_path = os.path.join([self.get_package_share_directory(), model_name_or_path])

        device = self.get_parameter_or("device", "gpu")

        self.conf_thres = self.get_parameter_or("conf_thres", 0.25)
        self.iou_thres = self.get_parameter_or("iou_thres", 0.45)

        self.mot_tracker = ByteTrack(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            track_thresh=0.5,
            det_thresh=0.1
        )

        super().__init__(
            model_path=model_path,
            device=device,
            batch_size=1,
            height=640,
            width=640,
            num_channels=3
        )

    def pipe(self, inputs, **kwargs):
        prediction = super().pipe(inputs) # [H, W, C] -> [bs, max_det, bbox]
        outputs = end2end_fastnms(
            prediction,
            conf_thres=kwargs.get("conf_thres", 0.25),
            iou_thres=kwargs.get("iou_thres", 0.45)
        )
        return outputs

    def callback(self, image):
        prediction = self.pipe(
            inputs=image,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres
        )
        
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        track_ids, outputs = self.mot_tracker.update(prediction)

        coords = [pose_estimate(info[5:5+8], info[5]) for info in outputs]

        
        
