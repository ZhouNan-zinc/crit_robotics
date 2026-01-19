from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, StringJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description() -> LaunchDescription:
    camera = LaunchConfiguration("camera")
    image = PathJoinSubstitution([camera, "image_raw"])
    size = LaunchConfiguration('size')
    square = LaunchConfiguration("square")

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
        DeclareLaunchArgument(
            "size",
            description="The number of internal corners per checkerboard row and column (e.g., 8x11 for 8 rows and 11 columns)."
        ),
        DeclareLaunchArgument(
            "square",
            description="The square length of the checkerboard square in meters, e.g., 0.025 for a 2.5 cm square."
        ),

        Node(
            package="unicam",
            executable="unicam",
            name="camera",
            namespace=camera,
            parameters=[config]
        ),

        Node(
            package='camera_calibration',
            executable='cameracalibrator',
            output='both',
            emulate_tty=True,
            arguments=['--size', size,'--square', square, '--no-service-check'],
            remappings=[('camera',StringJoinSubstitution(['/', camera])), ('image',StringJoinSubstitution(['/', image]))]
        )
    ])
