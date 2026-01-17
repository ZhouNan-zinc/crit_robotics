import os
import yaml
import sys

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.actions import Node

sys.path.append(os.path.join(get_package_share_directory("rm_bringup"), "launch"))

# 根据机器人修改
robot_dir = "hero"


def generate_launch_description():
    from launch_ros.descriptions import ComposableNode
    from launch_ros.actions import ComposableNodeContainer, Node
    from launch.actions import TimerAction, Shutdown
    from launch import LaunchDescription

    node_params = os.path.join(
        get_package_share_directory("rm_bringup"),
        "config",
        robot_dir,
        "node_params.yaml",
    )

    multi_cam_params = os.path.join(
        get_package_share_directory("rm_bringup"),
        "config",
        robot_dir,
        "multi_cam.yaml",
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
                "rm_gimbal_multi_cam.urdf.xacro",
            ),
            " pitch2cam_xyz:=",
            launch_params["camera_to_pitch"]["xyz"],
            " pitch2cam_rpy:=",
            launch_params["camera_to_pitch"]["rpy"],
            " pitch2cam2_xyz:=",
            launch_params["camera2_to_pitch"]["xyz"],
            " pitch2cam2_rpy:=",
            launch_params["camera2_to_pitch"]["rpy"],
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
    camera2_info_url = os.path.join(
        "package://rm_bringup/config", robot_dir, "camera2_info.yaml"
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

    outpost_predictor = Node(
        package="outpost_predictor",
        executable="outpost_predictor_node",
        name="outpost_predictor",
        output="both",
        emulate_tty=True,
        parameters=ros_parameters,
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

    switch_camera_dis1 = 5.0
    switch_camera_dis2 = 6.0
    switch_camera_t = 2.0
    telephoto = 2
    shortfocal = 1
    hkcam1 = Node(
        package="hik_camera",
        executable="hik_camera_node",
        name="hik_camera",
        output="both",
        emulate_tty=True,
        parameters=[
            node_params,
            {
                "camera_info_url": camera_info_url,
                "use_multi_cam": True,
                "switch_camera_dis1": switch_camera_dis1,
                "switch_camera_dis2": switch_camera_dis2,
                "switch_camera_t": switch_camera_t,
                "camera_mode": shortfocal,
            },
        ],
    )

    hkcam2 = Node(
        package="hik_camera",
        executable="hik_camera_node",
        name="hik_camera_2",
        output="both",
        emulate_tty=True,
        parameters=[
            node_params,
            multi_cam_params,
            {
                "camera_info_url": camera2_info_url,
                "use_sensor_data_qos": False,
                "use_multi_cam": True,
                "switch_camera_dis1": switch_camera_dis1,
                "switch_camera_dis2": switch_camera_dis2,
                "switch_camera_t": switch_camera_t,
                "camera_mode": telephoto,
            },
        ],
    )

    delay_serial_node = TimerAction(
        period=1.5,
        actions=[serial_driver_node],
    )

    # delay_tracker_node = TimerAction(
    #     period=2.0,
    #     actions=[tracker_node],
    # )

    return LaunchDescription(
        [
            robot_state_publisher,
            hkcam1,
            # hkcam2,
            detector,
            enemy_predictor,
            # outpost_predictor,
            serial_driver_node,
        ]
    )
