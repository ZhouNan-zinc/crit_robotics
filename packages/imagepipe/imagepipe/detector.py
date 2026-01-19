"""Entry point for launching the OpenVINO detection pipeline."""

import rclpy

from .node_interface import YoloPoseDetector

def main():
    rclpy.init()

    node = YoloPoseDetector()

    rclpy.spin(node)
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()
