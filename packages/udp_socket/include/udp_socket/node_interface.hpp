#pragma once

#include <rclcpp/node.hpp>
#include <boost/asio.hpp>

#include <std_msgs/msg/string.hpp>

#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace asio = boost::asio;

class UdpSocketNodeInterface : public rclcpp::Node {
public:
    explicit UdpSocketNodeInterface();

    ~UdpSocketNodeInterface();

    void transfer_message(const std::shared_ptr<std::string>& msg);

    using CallbackHook = std::function<void(const std::shared_ptr<std::string>&)>;

    template <typename Tp>
    std::shared_ptr<CallbackHook> register_callback(Tp&& callback_fn) {
        static_assert(
            std::is_invocable_v<Tp, const std::shared_ptr<std::string>&>,
            "Callback must be callable as void(const std::shared_ptr<std::string>&)"
        );
        callback = std::make_shared<CallbackHook>(std::forward<Tp>(callback_fn));
        return callback;
    }

protected:
    static const char* node_name;
    static rclcpp::NodeOptions options;
    rclcpp::Logger logger;

private:
    void async_activate();

    asio::io_context ctx;
    asio::ip::udp::socket socket;
    asio::ip::udp::endpoint endpoint;

    std::shared_ptr<std::string> buffer;

    std::jthread executor;
    asio::executor_work_guard<asio::io_context::executor_type> work_guard;

    std::shared_ptr<CallbackHook> callback;

    std::shared_ptr<rclcpp::TimerBase> timer;
};

class ProtobufLayer : public UdpSocketNodeInterface {
public:
    explicit ProtobufLayer();

private:
    std::shared_ptr<rclcpp::Publisher<std_msgs::msg::String>> proto_pub;
    std::shared_ptr<rclcpp::Subscription<std_msgs::msg::String>> proto_sub;
    std::shared_ptr<CallbackHook> proto_callback;
};
