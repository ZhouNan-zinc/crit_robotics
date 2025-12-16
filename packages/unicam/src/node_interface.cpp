#include "unicam/node_interface.hpp"

#include <opencv2/imgproc.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>

using namespace std::chrono_literals;

std::string deduce_encoding(const cv::Mat &image)
{
    const int type = image.type();
    const int depth = CV_MAT_DEPTH(type);
    const int channels = CV_MAT_CN(type);

    switch (depth) {
        case CV_8U:
            if (channels == 1) return "mono8";
            if (channels == 3) return "bgr8";
            if (channels == 4) return "bgra8";
            break;

        case CV_16U:
            if (channels == 1) return "mono16";
            if (channels == 3) return "bgr16";
            break;

        case CV_32F:
            if (channels == 1) return "32FC1";
            if (channels == 3) return "32FC3";
            break;
    }

    throw std::runtime_error("Unsupported cv::Mat type");
}

int deduce_step(int width, const std::string &encoding)
{
    int channels = 0;
    int bytes_per_channel = 0;

    if (encoding == "mono8" || encoding == "mono16" || encoding == "32FC1") {
        channels = 1;
    } else if (encoding == "bgr8" || encoding == "rgb8" ||
               encoding == "bgr16" || encoding == "rgb16" ||
               encoding == "32FC3") {
        channels = 3;
    } else if (encoding == "bgra8" || encoding == "rgba8") {
        channels = 4;
    } else {
        throw std::runtime_error("Unsupported encoding for step inference: " + encoding);
    }

    if (encoding.find("8") != std::string::npos) {
        bytes_per_channel = 1;
    } else if (encoding.find("16") != std::string::npos) {
        bytes_per_channel = 2;
    } else if (encoding.find("32F") != std::string::npos) {
        bytes_per_channel = 4;
    } else {
        throw std::runtime_error("Unsupported bit depth in encoding: " + encoding);
    }

    return width * channels * bytes_per_channel;
}

cv::Mat shrink_resize_crop(const cv::Mat& image, const cv::Size& size)
{
    /// NOTE: If got negative size, this can also work.
    double scale_ratio = std::min(
        std::fabs(1.0 * image.rows / size.height),
        std::fabs(1.0 * image.cols / size.width)
    );

    cv::Mat resized;
    cv::resize(
        image,
        resized,
        cv::Size(
            std::round(scale_ratio * image.cols),
            std::round(scale_ratio * image.rows)
        ),
        0., 0., cv::INTER_AREA);

    int crop_x = (resized.cols - size.width) / 2;
    int crop_y = (resized.rows - size.height) / 2;

    cv::Rect roi(crop_x, crop_y, size.width, size.height);
    cv::Mat cropped = resized(roi);

    return cropped;
}


const char* CameraNodeInterface::node_name = "camera";
const char* CameraNodeInterface::ns = "camera";
rclcpp::NodeOptions CameraNodeInterface::options = rclcpp::NodeOptions()
    .use_intra_process_comms(true)
    .automatically_declare_parameters_from_overrides(true);

CameraNodeInterface::CameraNodeInterface()
    : Node(node_name, ns, options), logger(get_logger())
{
    camera_pub = std::make_shared<image_transport::CameraPublisher>(
        image_transport::create_camera_publisher(
            this, "image_raw", rmw_qos_profile_sensor_data));

    auto camera_name = std::string(get_namespace()).substr(1);
    auto url = get_parameter_or<std::string>("url", ""); 
    // auto url = "/workspaces/RoboDevel/robo_devel/unicam/calibration/hikcam.yaml";
    /// FIXME: pass url to camera info manager
    cinfo_manager = std::make_shared<camera_info_manager::CameraInfoManager>(this, camera_name, url);

    if (cinfo_manager->validateURL(url)) {
        if (url == "") {
            RCLCPP_ERROR_STREAM(logger, "Unconfigured cinfo url, please configure it.");
        } else {
            RCLCPP_ERROR_STREAM(logger, "Invalid cinfo url:" << url);
        }
        RCLCPP_ERROR_STREAM(logger, "Setting up camera `" << camera_name << "` with empty cinfo");
    } else {
        if (not cinfo_manager->isCalibrated()) {
            RCLCPP_WARN_STREAM(logger, "Got uncalibrated camera `" << camera_name << "`, please check your yaml file.");
        }
    }

    callback_handle = add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter> &parameters) -> rcl_interfaces::msg::SetParametersResult {
            return dynamic_reconfigure(parameters);
        });
    
    auto timeout = static_cast<std::chrono::duration<double>>(get_parameter_or<double>("timeout", 1.0));
    auto daemon = [this]() -> void {
        if (this->is_alive()) {
            return;
        }
        try {
            this->run();
        } catch (std::exception &ex) {
            RCLCPP_ERROR_STREAM(logger, "Error while opening camera: " << ex.what() << ", Retrying...");
            return;
        }
        RCLCPP_INFO_STREAM(logger, "Open device.");
    };
    timer = create_timer(timeout, daemon);
}

void CameraNodeInterface::publish(const cv::Mat& image)
{
    auto pixel_width = get_parameter_or<int>("pixel_width", -1);
    auto pixel_height = get_parameter_or<int>("pixel_height", -1);
    auto roi = shrink_resize_crop(image, cv::Size(pixel_width, pixel_height));

    auto cinfo = std::make_unique<camera_info_manager::CameraInfo>(cinfo_manager->getCameraInfo());
    auto cimage = std::make_unique<sensor_msgs::msg::Image>();

    cimage->header = cinfo->header;
    cimage->height = roi.rows;
    cimage->width = roi.cols;
    cimage->encoding = deduce_encoding(roi);
    cimage->step = roi.step;
    cimage->is_bigendian = false;
    cimage->data.assign(roi.data, roi.data + roi.total() * roi.elemSize());

    camera_pub->publish(std::move(cimage), std::move(cinfo), now());
}

void CameraNodeInterface::publish(int height, int width, const std::string& encoding, const std::vector<unsigned char>& data)
{
    auto cinfo = std::make_unique<camera_info_manager::CameraInfo>(cinfo_manager->getCameraInfo());
    auto cimage = std::make_unique<sensor_msgs::msg::Image>();

    auto pixel_width = get_parameter_or<int>("pixel_width", -1);
    auto pixel_height = get_parameter_or<int>("pixel_height", -1);
    if (height != pixel_height and width != pixel_width) {
        RCLCPP_ERROR(logger, "Current we do not support resize here.");
        /// TODO: Implement resize with cv::Mat
    }

    cimage->header = cinfo->header;
    cimage->height = height;
    cimage->width = width;
    cimage->encoding = encoding;
    cimage->step = deduce_step(width, encoding);
    cimage->is_bigendian = false;
    cimage->data = data;

    camera_pub->publish(std::move(cimage), std::move(cinfo), now());
}

rcl_interfaces::msg::SetParametersResult CameraNodeInterface::dynamic_reconfigure([[maybe_unused]] const std::vector<rclcpp::Parameter> &parameters)
{
    auto result = rcl_interfaces::msg::SetParametersResult().set__successful(false);
    
    for (const auto &param : parameters) {
        RCLCPP_WARN_STREAM(logger, "Param `" << param.get_name() << "` does not support dynamic reconfigure");
    }

    return result;
}
