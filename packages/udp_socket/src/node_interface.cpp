#include "udp_socket/node_interface.hpp"

#include <rclcpp/logging.hpp>
#include <rclcpp/timer.hpp>
#include <rclcpp/utilities.hpp>

#include <boost/asio.hpp>

#include <chrono>
#include <exception>
#include <stop_token>
#include <utility>

const char* UdpSocketNodeInterface::node_name = "udp_socket";

rclcpp::NodeOptions UdpSocketNodeInterface::options = rclcpp::NodeOptions()
    .use_intra_process_comms(true)
    .automatically_declare_parameters_from_overrides(true);

UdpSocketNodeInterface::UdpSocketNodeInterface(): 
    rclcpp::Node(node_name, options),
    logger(get_logger()),
    ctx(),
    socket(ctx),
    work_guard(asio::make_work_guard(ctx)) 
{
    auto packet_size = get_parameter_or<int>("packet_size", 1024);
    buffer = std::make_shared<std::string>(packet_size, '\0');

    auto ip_addr = get_parameter_or<std::string>("ip_addr", "127.0.0.1");
    auto network = boost::asio::ip::make_network_v4(ip_addr + "/24");
    auto broadcast = network.broadcast();

    auto local_port = get_parameter_or("local_port", 6000);
    socket.open(asio::ip::udp::v4());
    socket.set_option(asio::socket_base::broadcast(true));
    socket.bind(asio::ip::udp::endpoint(asio::ip::udp::v4(), local_port));

    auto remote_port = get_parameter_or("remote_port", 6000);
    endpoint = asio::ip::udp::endpoint(broadcast, remote_port);

    auto timeout_sec = get_parameter_or<double>("timeout", 1.0);
    auto timeout = std::chrono::duration<double>(timeout_sec);

    auto daemon = [this]() -> void {
        if (executor.joinable()) {
            return;
        }
        async_activate();
        executor = std::jthread (
            [this](std::stop_token /*st*/) {
                try {
                    ctx.run();
                } catch (const std::exception& ex) {
                    RCLCPP_ERROR_STREAM(logger, "Network thread error: " << ex.what());
                }
            });
    };

    timer = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(timeout),
        daemon);
}

UdpSocketNodeInterface::~UdpSocketNodeInterface() {
    ctx.stop();
}

void UdpSocketNodeInterface::transfer_message(const std::shared_ptr<std::string>& msg) {
    socket.async_send_to(
        asio::buffer(*msg),
        endpoint,
        [this](std::error_code /*ecc*/, std::size_t bytes_sent) {
            RCLCPP_INFO(logger, "I sent %zu bytes", bytes_sent);
        });
}

void UdpSocketNodeInterface::async_activate() {
    socket.async_receive_from(
        asio::buffer(*buffer),
        endpoint,
        [this](std::error_code ecc, std::size_t /*bytes*/) {
            if (!ecc) {
                if (callback && *callback) {
                    (*callback)(buffer);
                } else {
                    RCLCPP_ERROR(logger, "Callback must be set before event loop.");
                }
            }
            async_activate();
        });
}

ProtobufLayer::ProtobufLayer() {
    proto_pub = create_publisher<std_msgs::msg::String>(
        "test_recv",
        rclcpp::SensorDataQoS());

    proto_sub = create_subscription<std_msgs::msg::String>(
        "test_send",
        rclcpp::SensorDataQoS(),
        [this](std_msgs::msg::String::SharedPtr msg) {
            transfer_message(std::make_shared<std::string>(msg->data));
        });

    proto_callback = register_callback(
        [this](const std::shared_ptr<std::string>& msg) {
            RCLCPP_INFO(logger, "I heard: %s", msg->data());
            auto proto_msg = std_msgs::msg::String().set__data(*msg);
            proto_pub->publish(proto_msg);
        });
}

