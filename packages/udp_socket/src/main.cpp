#include "udp_socket/node_interface.hpp"

#include <rclcpp/executors.hpp>
#include <rclcpp/utilities.hpp>

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ProtobufLayer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

