from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description() -> LaunchDescription:
    config = PathJoinSubstitution([
        FindPackageShare("udp_bridge"),
        "config",
        LaunchConfiguration("config", default="default.yaml")
    ])

    return LaunchDescription([
        Node(
            package="udp_bridge",
            executable="udp_bridge",
            name="udp_bridge",
            parameters=[config],
        )
    ])
