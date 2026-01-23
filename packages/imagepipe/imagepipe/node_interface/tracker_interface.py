"""Common ROS 2 node interfaces for detector."""

from abc import abstractmethod, ABC

from rclpy.node import Node
from rclpy.time import Time
from rclpy.logging import RcutilsLogger
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
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

from ..solutions.bytetrack import TrackingObject, ByteTrack
import numpy as np

def cxcywh2xyxy(bboxes: np.ndarray):
    """Convert bounding boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).
    
    Args:
        bboxes: Array of shape [batch_size, 4] with format [cx, cy, w, h]
    
    Returns:
        Array of shape [batch_size, 4] with format [x1, y1, x2, y2]
    """
    cx, cy, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


class TrackerNodeInterface(Node, ABC):
    """
    Docstring for TrackerNodeInterface
    """
    node_name = "tracker"

    def __init__(self):
        super().__init__(
            node_name=self.node_name,
            automatically_declare_parameters_from_overrides=True
        )

        self.vision_raw_sub = self.create_subscription(
            Detection2DArray,
            "vision/raw",
            self.callback,
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
        )

        self.vision_tracked_pub = self.create_publisher(
            Detection2DArray,
            "vision/tracked",
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
        )

        self.stamp_thres = self.get_clock().now()

    def callback(self, msg: Detection2DArray):
        stamp = Time.from_msg(msg.header.stamp, clock_type=self.get_clock().clock_type)
        # if stamp < self.stamp_thres:
            # self.logger.warning("Message not in order, check your image pipeline. Dropping message.")
            # return
        self.stamp_thres = stamp

        raw_detections = [TrackingObject(
            class_id=int(float(det.results[0].hypothesis.class_id)),
            score=float(det.results[0].hypothesis.score),
            bbox=cxcywh2xyxy(np.array([det.bbox.center.position.x, det.bbox.center.position.y, det.bbox.size_x, det.bbox.size_y])),
            message=det
        ) for det in msg.detections]

        trackers = self.update(raw_detections)

        tracked_detections = []
        for trk in trackers:
            msg = trk.message
            msg.id = str(int(trk.id))
            tracked_detections.append(msg)

        self.vision_tracked_pub.publish(Detection2DArray(
            header=msg.header,
            detections=tracked_detections
        ))

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError()

    @property
    def logger(self) -> RcutilsLogger:
        return self.get_logger()


class MotTracker(TrackerNodeInterface):
    """"""

    def __init__(self):
        super().__init__()  
        
        self.mot_tracker = ByteTrack(
            max_age=80,
            min_hits=2,
            iou_thres=None,
            conf_thres=0.3
        )

    def update(self, detections):
        return self.mot_tracker.update(detections)