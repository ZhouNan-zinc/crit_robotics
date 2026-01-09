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

    @abstractmethod
    def callback(self, cimage:Image, cinfo:CameraInfo):
        raise NotImplementedError()

    @property
    def get_package_share_directory(self):
        return get_package_share_directory(self.node_name)

    @property
    def logger(self) -> RcutilsLogger:
        return self.get_logger()


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

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError()

    @property
    def get_package_share_directory(self):
        return get_package_share_directory("image_pipeline")

    @property
    def logger(self) -> RcutilsLogger:
        return self.get_logger()
    