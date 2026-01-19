import os
import yaml
import sys

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.actions import Node

sys.path.append(os.path.join(get_package_share_directory("rm_bringup"), "launch"))

# 根据机器人修改
robot_dir = "rudder"
use_can = False
bag_dir = "/home/ubuntu/rosbag_new/rosbag2_outpost"


def generate_launch_description():
    from launch_ros.descriptions import ComposableNode
    from launch_ros.actions import ComposableNodeContainer, Node
    from launch.actions import TimerAction, Shutdown
    from launch import LaunchDescription
    from launch.actions import ExecuteProcess

    node_params = os.path.join(
        get_package_share_directory("rm_bringup"),
        "config",
        robot_dir,
        "node_params.yaml",
    )

    launch_params = yaml.safe_load(
        open(
            os.path.join(
                get_package_share_directory("rm_bringup"),
                "config",
                robot_dir,
                "launch_params.yaml",
            )
        )
    )
    ros_parameters = [node_params, {"pitch2yaw_t": launch_params["pitch2yaw_t"]}]

    robot_description = Command(
        [
            "xacro ",
            os.path.join(
                get_package_share_directory("rm_bringup"),
                "urdf",
                "rm_gimbal.urdf.xacro",
            ),
            " pitch2cam_xyz:=",
            launch_params["camera_to_pitch"]["xyz"],
            " pitch2cam_rpy:=",
            launch_params["camera_to_pitch"]["rpy"],
        ]
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {"robot_description": robot_description, "publish_frequency": 1000.0}
        ],
    )

    detector_dir = os.path.join(get_package_share_directory("rm_detector"))
    camera_info_url = os.path.join(
        "package://rm_bringup/config", robot_dir, "camera_info.yaml"
    )

    detector = Node(
        package="rm_detector",
        executable="rm_detector_node",
        name="rm_detector",
        output="both",
        emulate_tty=True,
        parameters=[node_params, {"detector_dir": detector_dir}],
    )

    enemy_predictor = Node(
        package="enemy_predictor",
        executable="enemy_predictor_node",
        name="enemy_predictor",
        output="both",
        emulate_tty=True,
        parameters=ros_parameters,
    )

    energy_predictor = Node(
        package="energy_predictor",
        executable="energy_predictor_node",
        name="energy_predictor",
        output="both",
        emulate_tty=True,
        parameters=ros_parameters,
    )

    fit = Node(
        package="fit",
        executable="fit_node",
        name="fit",
        output="both",
        emulate_tty=True,
        parameters=ros_parameters,
    )

    outpost_predictor = Node(
        package="outpost_predictor",
        executable="outpost_predictor_node",
        name="outpost_predictor",
        output="both",
        emulate_tty=True,
        parameters=ros_parameters,
    )

    foxglove = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
    )

    hkcam = Node(
        package="rm_camera",
        executable="rm_camera_node",
        name="hik_camera",
        output="both",
        emulate_tty=True,
        parameters=[node_params, {"camera_info_url": camera_info_url}],
    )

    serial_driver_node = Node(
        package="simple_serial_driver",
        executable="simple_serial_driver_node",
        name="simple_serial_driver",
        output="both",
        emulate_tty=True,
        parameters=ros_parameters,
        on_exit=Shutdown(),
        ros_arguments=[
            "--ros-args",
            "--log-level",
            "serial_driver:=" + launch_params["serial_log_level"],
        ],
    )

    return LaunchDescription(
        [
            robot_state_publisher,
            detector,
            enemy_predictor,
            # energy_predictor,
            # outpost_predictor,
            # foxglove,
            hkcam,
            serial_driver_node,
        ]
    )
