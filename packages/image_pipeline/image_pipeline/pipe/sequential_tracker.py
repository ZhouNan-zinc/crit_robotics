
from .node_interface import TrackerNodeInterface
from ..ops import pose_estimate
from ..ops import ByteTrack

from vision_msgs.msg import (
    Pose2D,
    BoundingBox2D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
    Detection2D,
    Detection2DArray
)

from rclpy.time import Time

import numpy as np


class SequentialTracker(TrackerNodeInterface):
    def __init__(self):
        super().__init__()

        self.mot_tracker = ByteTrack(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            track_thresh=0.5,
            det_thresh=0.1
        )

        self.stamp_thres = Time()

    def callback(self, msg: Detection2DArray):
        stamp = Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec)
        if stamp < self.stamp_thres:
            self.logger.warn("Message not in order, check your image pipeline. Dropping message.")
            return
        else:
            self.stamp_thres = stamp
        
        dets = []
        for det in msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y

            best = max(det.results, key=lambda res: res.score)
            score = float(best.score)
            class_id = int(best.hypothesis.class_id)

            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

            dets.append([x1, y1, x2, y2, score, class_id])

        dets = np.asarray(dets, dtype=np.float32) if len(dets) else np.empty((0, 6), dtype=np.float32)

        track_ids, outputs_info = self.mot_tracker.update(dets)

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





            