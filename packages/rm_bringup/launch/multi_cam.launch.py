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
    switch_camera_dis1 = 15.0
    switch_camera_dis2 = 16.0
    switch_camera_t = 2.0  # 切换相机CD
    telephoto = 2
    shortfocal = 1

    def get_intra_container():
        return ComposableNodeContainer(
            name="camera_detector_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container_isolated",
            composable_node_descriptions=[
                ComposableNode(
                    package="hik_camera",
                    plugin="hik_camera::HikCameraNode",
                    name="hik_camera",
                    parameters=[
                        node_params,
                        {
                            ## new_policy
                            "auto_switch_cam": False,
                            "camera_info_url": camera_info_url,
                            "use_multi_cam": True,
                            "switch_camera_dis1": switch_camera_dis1,
                            "switch_camera_dis2": switch_camera_dis2,
                            "switch_camera_t": switch_camera_t,
                            "camera_mode": shortfocal,
                            # ## old_policy
                            # "camera_info_url": camera_info_url,
                            # "use_multi_cam": True,
                            # "switch_camera_dis1": switch_camera_dis1,
                            # "switch_camera_dis2": switch_camera_dis2,
                            # "switch_camera_t": switch_camera_t,
                            # "camera_mode": shortfocal,
                        },
                    ],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
                ComposableNode(
                    package="hik_camera",
                    plugin="hik_camera::HikCameraNode",
                    name="hik_camera_2",
                    parameters=[
                        node_params,
                        multi_cam_params,
                        {
                            ## new_policy
                            "auto_switch_cam": False,
                            "camera_info_url": camera2_info_url,
                            "use_sensor_data_qos": False,
                            "use_multi_cam": True,
                            "switch_camera_dis1": switch_camera_dis1,
                            "switch_camera_dis2": switch_camera_dis2,
                            "switch_camera_t": switch_camera_t,
                            "camera_mode": telephoto,
                            # ## old_policy
                            # "camera_info_url": camera2_info_url,
                            # "use_sensor_data_qos": False,
                            # "use_multi_cam": True,
                            # "switch_camera_dis1": switch_camera_dis1,
                            # "switch_camera_dis2": switch_camera_dis2,
                            # "switch_camera_t": switch_camera_t,
                            # "camera_mode": telephoto,
                        },
                    ],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
                ComposableNode(
                    package="rm_detector",
                    plugin="rm_detector::DetectorNode",
                    name="rm_detector",
                    parameters=[node_params, {"detector_dir": detector_dir}],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
                ComposableNode(
                    package="enemy_predictor",
                    plugin="enemy_predictor::EnemyPredictorNode",
                    name="enemy_predictor",
                    parameters=ros_parameters,
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
                ComposableNode(
                    package="outpost_predictor",
                    plugin="outpost_predictor::OutpostPredictorNode",
                    name="outpost_predictor",
                    parameters=ros_parameters,
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
            ],
            output="both",
            emulate_tty=True,
            #            ros_arguments=['--ros-args', '--log-level',
            #                           'armor_detector:='+launch_params['detector_log_level']],
            on_exit=Shutdown(),
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

    intra_container = get_intra_container()

    delay_serial_node = TimerAction(
        period=1.5,
        actions=[serial_driver_node],
    )

    # delay_tracker_node = TimerAction(
    #     period=2.0,
    #     actions=[tracker_node],
    # )

    return LaunchDescription(
        [robot_state_publisher, intra_container, serial_driver_node]
    )
