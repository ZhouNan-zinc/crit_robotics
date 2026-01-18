"""Common ROS 2 node interfaces for detector."""

import os
import time
from functools import partial
from abc import abstractmethod, ABC

import cv2
import torch
import openvino as ov
import numpy as np
from rclpy.node import Node
from rclpy.logging import RcutilsLogger
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from ament_index_python import get_package_share_directory
from image_transport_py import ImageTransport
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

from ..runtime.models import Yolov10PoseModel
from ..solutions.pose_estimate import pose_estimate


SMALL_ARMOR_POINTS: list[tuple[float, float, float]] = [
    (1e-6, 6.75e-2, 2.75e-2),
    (1e-6, -6.75e-2, 2.75e-2),
    (1e-6, -6.75e-2, -2.75e-2),
    (1e-6, 6.75e-2, -2.75e-2)
]
BIG_ARMOR_POINTS: list[tuple[float, float, float]] = [
    (1e-6, 1.15e-1, 2.9e-2),
    (1e-6, -1.15e-1, 2.9e-2),
    (1e-6, -1.15e-1, -2.9e-2),
    (1e-6, 1.15e-1, -2.9e-2)
]
BASE_POINTS: list[tuple[float, float, float]] = []

CLASS_TO_POINTS = {
    1: BIG_ARMOR_POINTS,
    6: BIG_ARMOR_POINTS,
    2: SMALL_ARMOR_POINTS,
    4: SMALL_ARMOR_POINTS,
    5: BASE_POINTS,
}

def cimage_to_cv2_bgr(cimage: Image) -> np.ndarray:
    encoding = (cimage.encoding or "").lower()
    height, width, step = int(cimage.height), int(cimage.width), int(cimage.step)

    if encoding in ("bgr8", "rgb8"):
        channels = 3
        image = np.ndarray(
            shape=(height, width, channels),
            dtype=np.uint8,
            buffer=memoryview(cimage.data),
            strides=(step, channels, 1),
        )
        if encoding == "rgb8":
            image = image[:, :, ::-1].copy()
        return image

    if encoding in ("bgra8", "rgba8"):
        channels = 4
        image = np.ndarray(
            shape=(height, width, channels),
            dtype=np.uint8,
            buffer=memoryview(cimage.data),
            strides=(step, channels, 1),
        )
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR if encoding == "bgra8" else cv2.COLOR_RGBA2BGR,
        )

    if encoding == "mono8":
        image = np.ndarray(
            shape=(height, width),
            dtype=np.uint8,
            buffer=memoryview(cimage.data),
            strides=(step, 1),
        )
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported image encoding: {cimage.encoding!r}")

class DetectorNodeInterface(Node, ABC):
    """Base node that declares parameters up-front and exposes helpers."""
    node_name = "imagepipe"

    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)

    #     orig_init = cls.__init__

    #     def __init__(self, *args, **kw):
    #         orig_init(self, *args, **kw)
    #         self.post_init()

    #     cls.__init__ = __init__
        
    def __init__(self):
        super().__init__(
            node_name=self.node_name,
            automatically_declare_parameters_from_overrides=True
        )

        self.image_subs = [
            self.create_subscription(Image, f"{topic_prefix}/image_raw", partial(self.callback, topic_name=topic_prefix), 10)
            for topic_prefix in self.get_parameter("subscriptions").value
        ]

        self.camera_info_subs = [
            self.create_subscription(CameraInfo, f"{topic_prefix}/camera_info", partial(self.camera_info_callback, topic_name=topic_prefix), 10)
            for topic_prefix in self.get_parameter("subscriptions").value
        ]

        self.camera_infos = {topic_prefix: None for topic_prefix in  self.get_parameter("subscriptions").value}

    @property
    def logger(self) -> RcutilsLogger:
        return self.get_logger()

    @abstractmethod
    def callback(*args, **kwargs):
        raise NotImplementedError()
    
    def get_camera_info(self, topic_name:str):
        return self.camera_infos.get(topic_name, None)

    def camera_info_callback(self, camera_info:CameraInfo, topic_name:str|None=None):
        if topic_name is None:
            raise ValueError("Invalid topic name.")
        self.camera_infos[topic_name] = camera_info

class YoloPoseDetector(DetectorNodeInterface):
    """"""

    def __init__(self):

        self.model = Yolov10PoseModel.from_pretrained(
            os.path.join(get_package_share_directory("imagepipe"),"yolo", "v10"),
            use_safetensors=False,
            weights_only=True,
            dtype=torch.float32
        ).export()

        dummy_inputs = torch.randn((1, 3, 640, 640)).to(
            self.model.device).to(self.model.dtype)
        
        for _ in range(2):
            self.model(dummy_inputs) # dry run

        intermediate_model = ov.convert_model(self.model, input=[dummy_inputs.shape] ,example_input=dummy_inputs)

        core = ov.Core()
        core.set_property({
            "CACHE_DIR": os.path.expanduser("~/.cache/openvino"),
            "PERFORMANCE_HINT": "LATENCY",
        })

        ppp = ov.preprocess.PrePostProcessor(intermediate_model)
        ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(
            ov.Layout("NHWC")).set_color_format(ov.preprocess.ColorFormat.BGR)
        ppp.input().model().set_layout(ov.Layout("NCHW"))
        ppp.input().preprocess().convert_color(ov.preprocess.ColorFormat.RGB).convert_element_type(
            ov.Type.f32).resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR).scale(255.0)
        
        intermediate_model = ppp.build()

        self.ov_model = ov.compile_model(intermediate_model, device_name="AUTO")

        super().__init__()

        self.vision_raw_pub = self.create_publisher(
            Detection2DArray,
            "vision/raw",
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
        )
        
        # HACK:
        # model_name_or_path = self.get_parameter_or("model_name_or_path", "namespace/model")
        # if os.path.exists(os.path.join(get_package_share_directory("imagepipe"), model_name_or_path)):
        #     model_name_or_path = os.path.join(get_package_share_directory("imagepipe"), model_name_or_path)

    def callback(self, cimage:Image, topic_name:str|None=None):
        """Convert incoming image/camera info into detection messages."""
        camera_info = self.get_camera_info(topic_name)
        if camera_info is None:
            self.logger.warning(f"Waiting for camera info {topic_name}/camera_info to synchronize", throttle_duration_sec=1.0, skip_first=True)
            return

        pixel_values = cimage_to_cv2_bgr(cimage)

        outputs = self.ov_model([pixel_values])[self.ov_model.output(0)]

        predictions = self.model.postprocess(outputs)

        poses = self.estimate_poses_from_predictions(predictions, camera_info)

        self.publish_message_from_prediction(cimage.header, predictions, poses)

    def estimate_poses_from_predictions(
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

        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]

        camera_matrix = np.array(cinfo.k, dtype=np.float64).reshape(3, 3)
        distortion_coefficients = np.array(cinfo.d, dtype=np.float64)
        W, H = cinfo.width, cinfo.height

        poses = []

        for pred in prediction:
            if len(pred) < 14:
                raise ValueError(f"Prediction length {len(pred)} < 14")

            keypoints = np.array(
                pred[6:14],
                dtype=np.float64
            ).reshape(-1, 2)

            keypoints[:, 0] *= W
            keypoints[:, 1] *= H

            class_id = int(pred[5])
            object_points = np.asarray(CLASS_TO_POINTS.get(class_id, SMALL_ARMOR_POINTS), dtype=np.float32)

            position, orientation = pose_estimate(
                keypoints,
                object_points,
                camera_matrix,
                distortion_coefficients,
            )

            if position is not None and orientation is not None:
                poses.append((position, orientation))

        return poses
    
    def publish_message_from_prediction(
        self,
        header,
        prediction,
        poses
    ):
        self.vision_raw_pub.publish(Detection2DArray(
            header=header,
            detections=[Detection2D(
                header=header,
                results=[ObjectHypothesisWithPose(
                    hypothesis=ObjectHypothesis(
                        class_id=str(int(pred[5])),
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
