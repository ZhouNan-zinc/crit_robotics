#pragma once

#include <rclcpp/node.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <opencv2/core/mat.hpp>
#include <boost/asio/thread_pool.hpp>

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

/**
 * @brief Infer the ROS image encoding string for a given OpenCV matrix.
 *
 * @param image Source image to inspect.
 * @return Encoding compatible with sensor_msgs/Image::encoding such as "bgr8".
 * @throws std::runtime_error if the type cannot be mapped to a known encoding.
 */
std::string deduce_encoding(const cv::Mat &image);

/**
 * @brief Compute the stride (bytes per row) for a specific encoding.
 *
 * @param width Image width in pixels.
 * @param encoding Encoding identifier (e.g., "mono8").
 * @return Step/stride measured in bytes.
 * @throws std::runtime_error if the encoding or bit depth is unknown.
 */
cv::Mat shrink_resize_crop(const cv::Mat& image, const cv::Size& size);

/**
 * @brief Base class implementing shared camera node behaviors.
 *
 * Handles camera info publication, timer-based device bring-up, and a dynamic
 * parameter callback that derived cameras can override.
 */
class CameraNodeInterface : public rclcpp::Node {
public:
    CameraNodeInterface();
    ~CameraNodeInterface();

    /**
     * @brief Publish an OpenCV frame by converting it to sensor_msgs/Image.
     */
    virtual void publish(cv::Mat image);

    /**
     * @brief Default implementation logs unsupported parameter updates.
     */
    virtual rcl_interfaces::msg::SetParametersResult dynamic_reconfigure([[maybe_unused]] const std::vector<rclcpp::Parameter> &parameters);

protected:
    static const char* node_name;
    static const char* ns;
    static rclcpp::NodeOptions options;

    rclcpp::Logger logger;

    /**
     * @brief Return true when the camera is already running.
     */
    virtual bool is_alive() = 0;

    /**
     * @brief Open the device and attach callbacks. Called until success.
     */
    virtual void run() = 0;

private:
    std::shared_ptr<image_transport::CameraPublisher> camera_pub;
    std::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_manager;
    std::shared_ptr<rclcpp::node_interfaces::OnSetParametersCallbackHandle> callback_handle;
    std::shared_ptr<rclcpp::TimerBase> timer;

    boost::asio::thread_pool pool;
};
