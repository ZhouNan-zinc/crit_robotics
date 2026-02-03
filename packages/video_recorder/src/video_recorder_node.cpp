#include <sys/io.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <iomanip>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>

rclcpp::Rate rate(50);

namespace ngxy_record {
class RecorderNode : public rclcpp::Node {
   public:
    // 使用RCLCPP_COMPONENTS_REGISTER_NODE时，需要一个显式的构造函数
    explicit RecorderNode(const rclcpp::NodeOptions &options) : Node("recorder_node", options) {
        RCLCPP_INFO(get_logger(), "START RECORD");

        std::string hik_camera_image_topic =
            this->declare_parameter<std::string>("hik_camera_image_topic", "/image_topic");
        
        std::string detect_image_topic = 
            this->declare_parameter<std::string>("detect_image_topic", "/detect_record");

        std::string video_save_dir =
            this->declare_parameter<std::string>("video_save_path", "./");  // 默认保存在当前目录
        
        // printf("rrrrrr %s", video_save_dir);
        // video_save_dir = "/home/ubuntu/video_record";
        check_dir(video_save_dir);
        RCLCPP_INFO(this->get_logger(), "rrrrrr %s", video_save_dir.c_str());
        
        max_width = this->declare_parameter<int>("max_width", 1440);
        max_height = this->declare_parameter<int>("max_height", 1080);
        
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y_%m_%d_%H_%M_%S");
        std::string file_name = "record_" + ss.str() + ".avi";
        std::string file_name2 = "record2_" + ss.str() + ".avi";

        std::string video_save_path = video_save_dir + "/" + file_name;
        std::string video_save_path2 = video_save_dir + "/" + file_name2;
        
        video_writer = cv::VideoWriter(video_save_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                       50, cv::Size(max_width, max_height));
        
        video_writer2 = cv::VideoWriter(video_save_path2, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                       50, cv::Size(max_width, max_height));


        hik_camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            hik_camera_image_topic, rclcpp::SensorDataQoS(),
            std::bind(&RecorderNode::camera_image_callback, this, std::placeholders::_1));
        detect_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            detect_image_topic, rclcpp::SensorDataQoS(),
            std::bind(&RecorderNode::detect_image_callback, this, std::placeholders::_1));
    }

   private:
    void camera_image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat output(cv::Size(max_width, max_height), CV_8UC3);
            cv::resize(cv_ptr->image, output, output.size());
            video_writer.write(output);
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        rate.sleep();
    }

    void detect_image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat output(cv::Size(max_width, max_height), CV_8UC3);
            cv::resize(cv_ptr->image, output, output.size());
            video_writer2.write(output);
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        rate.sleep();
    }

    void check_dir(std::string dir){
        if(access(dir.c_str(), F_OK)== -1 && dir != "./"){
            RCLCPP_INFO(this->get_logger(), "mkdir %s", dir.c_str());
            int flag = mkdir(dir.c_str(), 0777);
            if(flag == -1){
                RCLCPP_ERROR(this->get_logger(), "mkdir %s failed", dir.c_str());
                throw std::exception();
            }
            return;
        }
        RCLCPP_INFO(this->get_logger(), "dir %s exist", dir.c_str());
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr hik_camera_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr detect_sub_;
    cv::VideoWriter video_writer;
    cv::VideoWriter video_writer2;
    int max_height;
    int max_width;
};

}  // namespace ngxy_record

// 使用RCLCPP_COMPONENTS_REGISTER_NODE宏来注册节点作为组件
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ngxy_record::RecorderNode)