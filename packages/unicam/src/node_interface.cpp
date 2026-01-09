#include "unicam/node_interface.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

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

cv::Mat shrink_resize_crop(const cv::Mat& image, const cv::Size& size)
{
    /// NOTE: If got negative size, this can also work.
    double scale_ratio = std::min(
        std::min(
            std::fabs(1.0 * image.rows / size.height),
            std::fabs(1.0 * image.cols / size.width)
        ),
        1.0
    );

    cv::Mat resized = image;
    cv::resize(
        image,
        resized,
        cv::Size(
            std::round(scale_ratio * image.cols),
            std::round(scale_ratio * image.rows)
        ),
        0., 0., cv::INTER_AREA);

    if ((resized.cols != size.width) xor (resized.rows != size.height)) {
        int crop_x = size.width > 0 ? std::round((resized.cols-size.width)/2) : resized.cols;
        int crop_y = size.height > 0 ? std::round((resized.rows-size.height)/2) : resized.rows;
        
        int roi_width = size.width > 0 ? size.width : resized.cols;
        int roi_height = size.height > 0 ? size.height : resized.rows;

        cv::Rect roi(crop_x, crop_y, roi_width, roi_height);
        cv::Mat cropped = resized(roi);

        return cropped;
    } else if ((resized.cols != size.width) && (resized.rows != size.height)) {
        return resized;
    }

    // auto only
    return resized;
}


const char* CameraNodeInterface::node_name = "camera";
const char* CameraNodeInterface::ns = "hikcam";
rclcpp::NodeOptions CameraNodeInterface::options = rclcpp::NodeOptions()
    .use_intra_process_comms(true)
    .automatically_declare_parameters_from_overrides(true);

CameraNodeInterface::CameraNodeInterface()
    : Node(node_name, ns, options), logger(get_logger())
{
    camera_pub = std::make_shared<image_transport::CameraPublisher>(
        image_transport::create_camera_publisher(
            this, "image_raw"));

    auto camera_name = std::string(get_namespace()).substr(1);
    auto url = get_parameter_or<std::string>("url", ""); 
    cinfo_manager = std::make_shared<camera_info_manager::CameraInfoManager>(this, camera_name, url);

    if (!cinfo_manager->validateURL(url)) {
        if (url == "") {
            RCLCPP_WARN_STREAM(logger, "Unconfigured cinfo url. You should provide a valid url unless you are calibrating the camera.");
        } else {
            RCLCPP_ERROR_STREAM(logger, "Invalid cinfo url:" << url);
        }
        cinfo_manager = std::make_shared<camera_info_manager::CameraInfoManager>(this, camera_name, "");
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
        } else {
            try {
                this->run();
            } catch (std::exception &ex) {
                RCLCPP_ERROR_STREAM(logger, "Error while opening camera: " << ex.what() << ", Retrying...");
            }
        }
    };
    timer = create_timer(timeout, daemon);
}
void CameraNodeInterface::publish(const cv::Mat& image)
{
    auto pixel_width = get_parameter_or<int>("pixel_width", -1);
    auto pixel_height = get_parameter_or<int>("pixel_height", -1);

    auto roi = shrink_resize_crop(image, cv::Size(pixel_width, pixel_height));  // FIXME: deduce pixel width and height

    auto cinfo = cinfo_manager->getCameraInfo();
    auto cimage = sensor_msgs::msg::Image()
        .set__header(cinfo.header)
        .set__height(roi.rows)
        .set__width(roi.cols)
        .set__encoding(deduce_encoding(roi))
        .set__step(roi.step)
        .set__is_bigendian(false);

    /// NOTE: using copy assignment can decline copy time from 2ms to 0.5ms
    cimage.data = std::vector<unsigned char>(image.datastart, image.dataend);

    camera_pub->publish(
        std::make_unique<sensor_msgs::msg::Image>(cimage),
        std::make_unique<sensor_msgs::msg::CameraInfo>(cinfo),
        now()
    );
}

rcl_interfaces::msg::SetParametersResult CameraNodeInterface::dynamic_reconfigure([[maybe_unused]] const std::vector<rclcpp::Parameter> &parameters)
{
    auto result = rcl_interfaces::msg::SetParametersResult().set__successful(false);
    
    for (const auto &param : parameters) {
        RCLCPP_WARN_STREAM(logger, "Param `" << param.get_name() << "` does not support dynamic reconfigure");
    }

    return result;
}
