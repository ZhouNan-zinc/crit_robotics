"""Common ROS 2 node interfaces for image pipeline components."""

from abc import abstractmethod, ABC

from rclpy.node import Node
from rclpy.logging import RcutilsLogger
from ament_index_python import get_package_share_directory

class ImagePipelineNodeInterface(Node, ABC):
    """Base node that declares parameters up-front and exposes helpers."""

    node_name = "image_pipeline"
    def __init__(self):
        super().__init__(
            node_name=self.node_name,
            automatically_declare_parameters_from_overrides=True
        )

    @abstractmethod
    def pipe(self, **kwargs):
        raise NotImplementedError()

    @property
    def get_package_share_directory(self):
        return get_package_share_directory("image_pipeline")

    @property
    def logger(self) -> RcutilsLogger:
        return self.get_logger()
