"""Entry point for launching the OpenVINO tracking pipeline."""

import rclpy

from .node_interface import MotTracker

def main():
    rclpy.init()

    node = MotTracker()

    rclpy.spin(node)
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()
