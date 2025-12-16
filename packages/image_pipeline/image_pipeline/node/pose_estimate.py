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
    BoundingBox2D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
    Detection2D,
    Detection2DArray
)

from .node_interface import ImagePipelineNodeInterface
from ..ops import pose_estimate

class PosePipelineNodeInterface(ImagePipelineNodeInterface):
    """Node base class that converts detector output into Detection2DArray."""

    def __init__(self):
        super().__init__()

        self.br = CvBridge()
        
        self.detection_array_pub = self.create_publisher(
            Detection2DArray,
            "detection_array",
            QoSProfile(10)
        )

        subscribe_to = self.get_parameter_or("subscribe_to", ["hikcam"]).value
        it = ImageTransport(
            node_name=self.node_name,
            image_transport="raw"
        )
        self.image_subs = [
            it.subscribe_camera(
                base_topic=f"{namespace}/image_raw",
                queue_size=1,
                callback=self.callback
            )
            for namespace in subscribe_to
        ]

    def callback(self, cimage:Image, cinfo:CameraInfo):
        """Convert incoming image/camera info into detection messages."""
        image = self.br.imgmsg_to_cv2(cimage, desired_encoding="bgr8")
        
        outputs_info, track_ids = self.pipe(image)

        header = cinfo.header
        camera_matrix = np.array(cinfo.k, dtype=np.float64).reshape(3, 3)
        distortion_coefficients = np.array(cinfo.d, dtype=np.float64)
        projection_matrix = np.array(cinfo.p, dtype=np.float64).reshape(3, 4)
        W, H = cinfo.width, cinfo.height

        detection_array = Detection2DArray(
            header=header,
            detections=[]
        )
        
        for track_id, info in zip(track_ids, outputs_info):
            bbox = np.array(info[:4]) * np.array([W, H, W, H])
            keypoints  = np.array(info[6:6+8]).reshape(-1, 2) * np.array([W, H])
            score = info[4]
            class_id = info[5]

            position, orientation = pose_estimate(
                keypoints=keypoints,
                class_id=class_id,
                camera_matrix=camera_matrix,
                distortion_coefficients=distortion_coefficients
            )
            
            detection_array.detections.append(
                Detection2D(
                    header=header,
                    results=ObjectHypothesisWithPose(
                        hypothesis=ObjectHypothesis(
                            class_id=str(class_id),
                            score=float(score)
                        ),
                        pose=PoseWithCovariance(
                            pose=Pose(
                                position=Point(
                                    x=float(position[0]),
                                    y=float(position[1]),
                                    z=float(position[2])
                                ),
                                orientation=Quaternion(
                                    x=float(orientation[0]),
                                    y=float(orientation[1]),
                                    z=float(orientation[2])
                                )
                            )
                        )
                    ),
                    bbox=BoundingBox2D(
                        center=Pose2D(
                            x=float(bbox[0]),
                            y=float(bbox[1])
                        ),
                        size_x=float(bbox[2]),
                        size_y=float(bbox[3])
                    ),
                    id=str(track_id)
                )
            )

        self.detection_array_pub.publish(detection_array)
