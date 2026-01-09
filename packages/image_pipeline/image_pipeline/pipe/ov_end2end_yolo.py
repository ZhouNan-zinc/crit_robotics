"""OpenVINO-powered YOLO pipeline with ByteTrack post-processing."""

import os
from functools import partial

import numpy as np
import cv2

from ..ops import end2end_fastnms, pose_estimate
from ..node import PosePipelineNodeInterface
from ..backend import OpenVinoBackend
from ..tracking import ByteTrack


"""Pose-aware image pipeline node that subscribes to camera topics."""

import numpy as np
from image_transport_py import ImageTransport
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile

from sensor_msgs.msg import (
    Image,
    CameraInfo
)
from geometry_msgs.msg import (
    Point,
    Quaternion,
    Pose,
    PoseWithCovariance
)
from vision_msgs.msg import (
    Pose2D,
    Point2D,
    BoundingBox2D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
    Detection2D,
    Detection2DArray
)

from ..pipe.node_interface import ImagePipelineNodeInterface
from ..ops import pose_estimate

class End2endYolo(ImagePipelineNodeInterface):
    """"""

    def __init__(self):
        super().__init__()

        self.br = CvBridge()
        
        self.detection_array_pub = self.create_publisher(
            Detection2DArray,
            "detection_array",
            QoSProfile(10)
        )

        
        model_name_or_path = self.get_parameter_or("model_name_or_path", "namespace/model")
        if os.path.exists(model_name_or_path):
            model_path = model_name_or_path
        else:
            model_path = os.path.join([self.get_package_share_directory(), model_name_or_path])

        self.model = AutoBackend.from_pretrained(
            model_path
        )

    def predict(self, inputs, **kwargs):
        prediction = self.model(inputs, **kwargs) # [H, W, C] -> [bs, max_det, bbox]

        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        outputs = end2end_fastnms(
            prediction,
            conf_thres=self.get_parameter_or("conf_thres", 0.25),
            iou_thres=self.get_parameter_or("iou_thres", 0.45)
        )

        return outputs

    def estimate_poses_from_prediction(
        self,
        prediction,
        cinfo,
    ):
        """
        Estimate object poses from network prediction and camera info.

        Args:
            prediction: iterable, each element format:
                [ ..., class_id, kp0_x, kp0_y, kp1_x, kp1_y, kp2_x, kp2_y, kp3_x, kp3_y, ... ]
            cinfo: sensor_msgs.msg.CameraInfo

        Returns:
            poses: list of (position, orientation)
        """

        camera_matrix = np.array(cinfo.k, dtype=np.float64).reshape(3, 3)
        distortion_coefficients = np.array(cinfo.d, dtype=np.float64)
        W, H = cinfo.width, cinfo.height

        poses = []

        for pred in prediction:
            if len(pred) < 14:
                raise ValueError(f"Prediction length {len(pred)} < 14")

            class_id = int(pred[5])

            keypoints = np.array(
                pred[6:14],
                dtype=np.float64
            ).reshape(-1, 2)

            keypoints[:, 0] *= W
            keypoints[:, 1] *= H

            position, orientation = pose_estimate(
                keypoints=keypoints,
                class_id=class_id,
                camera_matrix=camera_matrix,
                distortion_coefficients=distortion_coefficients,
            )

            poses.append((position, orientation))

        return poses
    
    def publish_message_from_prediction(
        self,
        header,
        prediction,
        poses
    ):
        self.detection_array_pub.publish(Detection2DArray(
            header=header,
            detections=[Detection2D(
                header=header,
                results=[ObjectHypothesisWithPose(
                    hypothesis=ObjectHypothesis(
                        class_id=int(pred[5]),
                        score=float(pred[4])
                    ),
                    pose=PoseWithCovariance(
                        pose=Pose(
                            position=Point(
                                x=float(pos[0]),
                                y=float(pos[1]),
                                z=float(pos[2])
                            ),
                            orientation=Quaternion(
                                x=float(orient[0]),
                                y=float(orient[1]),
                                z=float(orient[2])
                            )
                        )
                    )
                )],
                bbox=BoundingBox2D(
                    center=Pose2D(
                        position=Point2D(
                            x=float(pred[0]),
                            y=float(pred[1])
                        )
                    ),
                    size_x=float(pred[2]),
                    size_y=float(pred[3])
                )
            ) for pred, (pos, orient) in zip(prediction, poses)]
        ))
        
        

    def callback(self, cimage:Image, cinfo:CameraInfo):
        """Convert incoming image/camera info into detection messages."""
        image = self.br.imgmsg_to_cv2(cimage, desired_encoding="bgr8")
        
        prediction = self.predict([image])

        poses = self.estimate_poses_from_prediction(prediction)

        header = cinfo.header
        self.publish_message_from_prediction(header, prediction, poses)