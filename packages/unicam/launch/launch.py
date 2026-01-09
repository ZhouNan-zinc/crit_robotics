from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description() -> LaunchDescription:
    namespace = LaunchConfiguration("camera")
    config = PathJoinSubstitution([
        FindPackageShare("unicam"),
        "config",
        LaunchConfiguration("config", default="default.yaml")
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            "camera", 
            default_value="hikcam",
            description="The camera namespace or frame used for the camera topics."
        ),
        Node(
            package="unicam",
            executable="unicam",
            name="camera",
            namespace=namespace,
            parameters=[config]
        )
    ])
