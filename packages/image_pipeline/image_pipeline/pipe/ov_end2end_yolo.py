"""OpenVINO-powered YOLO pipeline with ByteTrack post-processing."""

import os
from functools import partial

import numpy as np
import cv2

from ..ops import end2end_fastnms, pose_estimate
from ..node import PosePipelineNodeInterface
from ..backend import OpenVinoBackend
from ..tracking import ByteTrack

class OpenVinoEnd2endYolo(PosePipelineNodeInterface):
    """ROS 2 node that wraps an end-to-end YOLO model accelerated by OpenVINO."""

    def __init__(self):
        super().__init__()

        model_name_or_path = self.get_parameter_or("model_name_or_path", "namespace/model")
        if os.path.exists(model_name_or_path):
            model_path = model_name_or_path
        else:
            model_path = os.path.join([self.get_package_share_directory(), model_name_or_path])

        dd = {
            "device": self.get_parameter_or("device", "CPU"),
            "dtype": self.get_parameter_or("dtype", "float32")
        }
        
        self.model = OpenVinoBackend(model_path, **dd)

        self.mot_tracker = ByteTrack(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            track_thresh=0.5,
            det_thresh=0.1
        )

    def pipe(self, inputs, *, conf_thres=0.25, iou_thres=0.45, **kwargs):
        """Run inference, apply NMS, and update the tracker."""
        prediction = self.model(inputs, **kwargs) # [H, W, C] -> [bs, max_det, bbox]

        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        outputs = end2end_fastnms(
            prediction,
            conf_thres=self.get_parameter_or("conf_thres", conf_thres),
            iou_thres=self.get_parameter_or("iou_thres", iou_thres)
        )

        track_ids, outputs_info = self.mot_tracker.update(outputs)

        return outputs_info, track_ids
