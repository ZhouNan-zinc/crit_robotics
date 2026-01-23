import os
import sys
import yaml

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.actions import Node
from launch.actions import Shutdown
from launch import LaunchDescription

sys.path.append(os.path.join(get_package_share_directory("rm_bringup"), "launch"))

robot_dir = "new_rudder_2"

def generate_launch_description():

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

    robot_description = Command([
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
    ])

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[
                {"robot_description": robot_description, "publish_frequency": 1000.0}
            ],
        ),
        Node(
            package="simple_serial_driver",
            executable="simple_serial_driver_node",
            name="simple_serial_driver",
            parameters=ros_parameters,
            on_exit=Shutdown(),
            ros_arguments=[
                "--ros-args",
                "--log-level",
                "serial_driver:=" + launch_params["serial_log_level"],
            ],
        ),
        Node(
            package="enemy_predictor",
            executable="enemy_predictor_node",
            name="enemy_predictor",
            parameters=ros_parameters,
        )
    ])
