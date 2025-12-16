#include "unicam/hik_usb_cam.hpp"

#include <rclcpp/utilities.hpp>
#include <rclcpp/executors.hpp>

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HikVisionUsbCam>();
    rclcpp::spin(node);
    rclcpp::shutdown();
}