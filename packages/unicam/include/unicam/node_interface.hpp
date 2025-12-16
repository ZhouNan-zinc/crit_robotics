#pragma once

#include <rclcpp/node.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <opencv2/core/mat.hpp>

namespace rclcpp {
class Parameter;
namespace node_interfaces {
class OnSetParametersCallbackHandle;
}
class TimerBase;
}

namespace image_transport {
class CameraPublisher;
}

namespace camera_info_manager {
class CameraInfoManager;
}


std::string deduce_encoding(const cv::Mat &image);

int deduce_step(int width, const std::string &encoding);

cv::Mat shrink_resize_crop(const cv::Mat& image, const cv::Size& size);

class CameraNodeInterface : public rclcpp::Node {
public:
    CameraNodeInterface();

    virtual void publish(const cv::Mat& image);
    virtual void publish(int height, int width, const std::string& encoding, const std::vector<unsigned char>& data);
    virtual rcl_interfaces::msg::SetParametersResult dynamic_reconfigure([[__attribute_maybe_unused__]] const std::vector<rclcpp::Parameter> &parameters);

protected:
    static const char* node_name;
    static const char* ns;
    static rclcpp::NodeOptions options;
    rclcpp::Logger logger;

    virtual bool is_alive() = 0;
    virtual void run() = 0;

private:
    std::shared_ptr<image_transport::CameraPublisher> camera_pub;
    std::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_manager;
    std::shared_ptr<rclcpp::node_interfaces::OnSetParametersCallbackHandle> callback_handle;
    std::shared_ptr<rclcpp::TimerBase> timer;
};
