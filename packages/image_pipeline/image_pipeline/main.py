"""Entry point for launching the OpenVINO detection pipeline."""

import rclpy

from .pipe import OpenVinoEnd2endYolo

def main():
    rclpy.init()

    node = OpenVinoEnd2endYolo()

    rclpy.spin(node)
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()
