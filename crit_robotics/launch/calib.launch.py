from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description() -> LaunchDescription:
    namespace = LaunchConfiguration("camera")
    size = LaunchConfiguration('size')
    square = LaunchConfiguration("square")
    config = PathJoinSubstitution([
        FindPackageShare("unicam"),
        "config",
        LaunchConfiguration("config", default="default.yaml")
    ])
    
    remappings = [
        ("/image", PathJoinSubstitution([namespace, "image_raw"])),
        ("/camera", PathJoinSubstitution([namespace, "camera_info"])),
    ]

    return LaunchDescription([
        DeclareLaunchArgument("camera", default_value="camera_optical_frame"),
        DeclareLaunchArgument("size"),
        DeclareLaunchArgument("square"),

        Node(
            package="unicam",
            executable="unicam",
            name="camera",
            namespace=namespace,
            parameters=[config]
        ),

        Node(
            package="camera_calibration",
            executable="cameracalibrator",
            name="calibrator",
            output='both',
            emulate_tty=True,
            arguments=['--size', size,'--square', square],
            remappings=remappings
        )
    ])
