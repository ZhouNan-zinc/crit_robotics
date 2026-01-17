#include "unicam/node_interface.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/asio/post.hpp>

#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>

using namespace std::chrono_literals;

std::string deduce_encoding(int type)
{
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

std::string deduce_encoding(const cv::Mat &image)
{
    return deduce_encoding(image.type());
}


const char* CameraNodeInterface::node_name = "camera";
const char* CameraNodeInterface::ns = "hikcam";
rclcpp::NodeOptions CameraNodeInterface::options = rclcpp::NodeOptions()
    .use_intra_process_comms(true)
    .automatically_declare_parameters_from_overrides(true);

CameraNodeInterface::CameraNodeInterface()
    : Node(node_name, ns, options), logger(get_logger()), pool(2)
{
    camera_pub = std::make_shared<image_transport::CameraPublisher>(
        image_transport::create_camera_publisher(
            this, "image_raw"));

    if (!has_parameter("pixel_width")) {
        declare_parameter("pixel_width", -1);
    }
    if (!has_parameter("pixel_height")) {
        declare_parameter("pixel_height", -1);
    }

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

    auto timeout = static_cast<std::chrono::duration<double>>(get_parameter_or<double>("timeout", 1.0));
    auto daemon = [this]() -> void {
        if (this->is_alive()) {
            return;
        } else {
            if (callback_handle == nullptr) {
                callback_handle = add_on_set_parameters_callback(
                    [this](const std::vector<rclcpp::Parameter> &parameters) -> rcl_interfaces::msg::SetParametersResult {
                    return dynamic_reconfigure(parameters);
                });
            }
            try {
                this->run();
            } catch (std::exception &ex) {
                RCLCPP_ERROR_STREAM(logger, "Error while opening camera: " << ex.what() << ", Retrying...");
            }
        }
    };

    timer = create_timer(timeout, daemon);
    
}

CameraNodeInterface::~CameraNodeInterface() { 
    pool.join();
}

void CameraNodeInterface::publish(cv::Mat image)
{
    boost::asio::post(pool, [this, image = std::move(image)] mutable {
        auto width = get_parameter_or<int>("width", -1);
        auto height = get_parameter_or<int>("height", -1);

        /// NOTE: If got negative size, this can also work.
        double scale_ratio = std::max(1.0 * width / image.cols, 1.0 * height / image.rows);
        if (scale_ratio > 0) {
            cv::Rect roi(
                width > 0 ? std::round((image.cols-width/scale_ratio)/2) : 0,
                height > 0 ? std::round((image.rows-height/scale_ratio)/2) : 0,
                width > 0 ? width/scale_ratio : image.cols,
                height > 0 ? height/scale_ratio: image.rows
            );

            image = image(roi);

            cv::resize(
                image,
                image,
                cv::Size(
                    std::round(image.cols * scale_ratio),
                    std::round(image.rows * scale_ratio)
                ),
                0., 0., cv::INTER_LINEAR);
        }

        if (not image.isContinuous()) {
            image = image.clone();
        }


        auto cinfo = cinfo_manager->getCameraInfo();
        auto cimage = sensor_msgs::msg::Image()
            .set__header(cinfo.header)
            .set__height(image.rows)
            .set__width(image.cols)
            .set__encoding(deduce_encoding(image))
            .set__step(image.cols * image.elemSize())
            .set__is_bigendian(false);

        /// NOTE: using copy assignment can decline copy time from 2ms to 0.3ms
        cimage.data.assign(image.data, image.data + image.cols * image.elemSize() * image.rows);

        camera_pub->publish(
            std::make_unique<sensor_msgs::msg::Image>(std::move(cimage)),
            std::make_unique<sensor_msgs::msg::CameraInfo>(std::move(cinfo)),
            now()
        );
    });
}

rcl_interfaces::msg::SetParametersResult CameraNodeInterface::dynamic_reconfigure([[maybe_unused]] const std::vector<rclcpp::Parameter> &parameters)
{
    auto result = rcl_interfaces::msg::SetParametersResult().set__successful(false);
    
    for (const auto &param : parameters) {
        RCLCPP_WARN_STREAM(logger, "Param `" << param.get_name() << "` does not support dynamic reconfigure");
    }

    return result;
}
