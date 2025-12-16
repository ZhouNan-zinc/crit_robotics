from abc import abstractmethod, ABC

import sensor_msgs
from rclpy.node import Node
from rclpy.logging import RcutilsLogger
from cv_bridge import CvBridge
from ament_index_python import get_package_share_directory

class ImagePipelineNodeInterface(Node, ABC):
    node_name = "image_pipeline"
    def __init__(self):
        super().__init__(
            node_name=self.node_name,
            automatically_declare_parameters_from_overrides=True
        )

        self.br = CvBridge()
        self.image_subs = []

        subscribe_to = self.get_parameter_or("subscribe_to", ["hikcam"]).value
        for namespace in subscribe_to:
            self.create_subscription(
                sensor_msgs.msg.Image,
                topic=f"{namespace}/image_raw",
                callback=lambda msg: self.callback(self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8'))
            )

    @property
    def get_package_share_directory(self):
        return get_package_share_directory("image_pipeline")

    @abstractmethod
    def callback(self, image):
        raise NotImplementedError()

    @property
    def logger(self) -> RcutilsLogger:
        return self.get_logger()