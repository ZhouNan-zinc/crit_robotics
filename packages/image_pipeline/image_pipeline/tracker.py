"""Entry point for launching the OpenVINO tracking pipeline."""

import rclpy

from .pipe import 

def main():
    rclpy.init()

    node = OpenVinoEnd2endYolo()

    rclpy.spin(node)
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()
