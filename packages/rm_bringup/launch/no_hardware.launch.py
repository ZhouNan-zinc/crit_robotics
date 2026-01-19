import os
import yaml
import sys

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.actions import Node

sys.path.append(os.path.join(get_package_share_directory("rm_bringup"), "launch"))

# 根据机器人修改
robot_dir = "new_rudder"
use_can = False
# rosbag config
bag_dir = "/workspaces/crit_robotics/rosbag2_2026_01_17-23_38_02"
rate = "1"

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
                "rm_gimbal_multi_cam.urdf.xacro",
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

    #detector_dir = os.path.join(get_package_share_directory("rm_detector"))
    camera_info_url = os.path.join(
        "package://rm_bringup/config", robot_dir, "camera_info.yaml"
    )

    def get_intra_container():
        return ComposableNodeContainer(
            name="intra_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container_isolated",
            composable_node_descriptions=[
                # ComposableNode(
                #     package="rm_republish",
                #     plugin="ngxy_republish::RepublishNode",
                #     name="rm_republish",
                #     parameters=[node_params],
                #     extra_arguments=[{"use_intra_process_comms": True}],
                # ),
                # ComposableNode(
                #     package="video_recorder",
                #     plugin="ngxy_record::RecorderNode",
                #     name="recorder_node",
                #     parameters=[node_params],
                #     extra_arguments=[{"use_intra_process_comms": True}],
                # ),
                # ComposableNode(
                #     package='rm_camera',
                #     plugin='ngxy_camera::HikCameraNode',
                #     name='hik_camera',
                #     parameters=[node_params, {
            	#         'camera_info_url': camera_info_url
                #     }],
                #     extra_arguments=[{'use_intra_process_comms': True}]
                # ),
                # ComposableNode(
                #     package="rm_detector",
                #     plugin="ngxy_detect::DetectorNode",
                #     name="rm_detector",
                #     parameters=[node_params, {"detector_dir": detector_dir}],
                #     extra_arguments=[{"use_intra_process_comms": True}],
                # ),
                #ComposableNode(
                #    package="rm_detector",
                #    plugin="ngxy_detect::DetectorNode",
                #    name="rm_detector",
                #    parameters=[node_params, {"detector_dir": detector_dir}],
                #    extra_arguments=[{"use_intra_process_comms": True}],
                #),
                ComposableNode(
                    package="enemy_predictor",
                    plugin="EnemyPredictorNode",
                    name="enemy_predictor",
                    parameters=ros_parameters,
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
                # ComposableNode(
                #     package="energy_predictor",
                #     plugin="energy_predictor::EnergyPredictorNode",
                #     name="energy_predictor",
                #     parameters=ros_parameters,
                #     extra_arguments=[{"use_intra_process_comms": True}],
                # ),
                # ComposableNode(
                #     package="outpost_predictor",
                #     plugin="ngxy_outpost_predictor::OutpostPredictorNode",
                #     name="outpost_predictor",
                #     parameters=ros_parameters,
                #     extra_arguments=[{"use_intra_process_comms": True}],
                # ),
            ],
            output="both",
            emulate_tty=True,
            #            ros_arguments=['--ros-args', '--log-level',
            #                           'armor_detector:='+launch_params['detector_log_level']],
            on_exit=Shutdown(),
        )

    intra_container = get_intra_container()

    rosbag = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            bag_dir,
            "-r",
            rate,
        ],
        output="screen",
    )

    # delay_tracker_node = TimerAction(
    #     period=2.0,
    #     actions=[tracker_node],
    # )


    foxglove = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(get_package_share_directory("rm_bringup"), "config", "autoaim.rviz")]
    )

    energy_detect = Node(
        package="energy_detect",
        executable="energy_detect_node",
        name="energy_detect"
    )

    # 添加 fit 节点
    fit = Node(
        package="energy_predictor",
        executable="fit_node",
        name="fit"
    )

    return LaunchDescription(
        [
            intra_container,
            robot_state_publisher,
            # foxglove,
            rosbag,
            # rviz2,
            # recorder,
        ]
    )
