#ifndef _SERIAL_DRIVER_NODE_H
#define _SERIAL_DRIVER_NODE_H

#include <cstdint>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>

#include <tf2_ros/transform_broadcaster.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/int8.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <std_srvs/srv/trigger.hpp>

#include <rm_msgs/msg/rm_robot.hpp>
#include <rm_msgs/msg/control.hpp>

#include <simple_serial_driver/serial_protocol.h>
#include <simple_serial_driver/crc.h>

#include <serial/serial.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <cassert>
#include <sys/types.h>
#include <unistd.h>
#include <filesystem>

namespace ngxy_simple_serial{

static const std::vector<std::string> dev_names = {"/dev/ttyUSB0", "/dev/ttyACM0"};

struct SerialDriverNodeParams{
    float timestamp_offset;
    int baud_rate;
    int imu_msg_id;
    std::string device_name;
    std::string robot_name;
    std::vector<double> pitch2yaw_t;
};

class SerialDriverNode : public rclcpp::Node{
public:
    explicit SerialDriverNode(const rclcpp::NodeOptions& _options);

    ~SerialDriverNode() override;

    void loadParams();

    //ros topic callback
    void readPortCallback(uint8_t* buffer);
    void autoaimReadPortCallback(const autoaim_recv_from_port_data_t* _data);
    void autolonReadPortCallback(const autolob_recv_from_port_data_t* _data){
        RCLCPP_ERROR(this->get_logger(), "recv autolob");
        RCLCPP_INFO(this->get_logger(), std::to_string(_data->content.xy).c_str());
        RCLCPP_INFO(this->get_logger(), std::to_string(_data->content.z).c_str());
        RCLCPP_INFO(this->get_logger(), std::to_string(_data->content.yaw).c_str());
        RCLCPP_INFO(this->get_logger(), std::to_string(_data->content.pitch).c_str());
        RCLCPP_INFO(this->get_logger(), std::to_string(_data->content.k).c_str());
        RCLCPP_INFO(this->get_logger(), std::to_string(_data->content.v0).c_str());
    }
    void ControlCallback(const rm_msgs::msg::Control::SharedPtr _msg);

    // serial Port
    bool isDeviceValid(const std::string& dev_name){
        return std::filesystem::exists(dev_name) && std::filesystem::is_block_file(dev_name);
    }
    void initPort();
    void closePort();
    void reOpenPort();
    void writeToPort(autoaim_send_to_port_data_t _data);
    void readFromPort();

    bool read(uint8_t* buffer, int size){
        int res = port_->read(buffer, size);
        if(res != size)
            RCLCPP_WARN(get_logger(),"Read Failed, read: %d", res);
        // RCLCPP_INFO(get_logger(), "\033[0;32m Should Read %d, actually Read %d\033[0m", size, res);
        return res == size;
    }
    
    uint8_t buffer_check_valid(uint8_t* buffer, uint32_t buffer_size, CRC16& checker) {
        uint16_t crc_val;
        memcpy(&crc_val, buffer + buffer_size - 2, 2);
        uint16_t crc_chk = checker.check_sum(buffer, buffer_size - 2);
        if (crc_chk != crc_val){
            crc_chk = checker.check_sum(buffer + 1, buffer_size - 3);
            if (crc_chk != crc_val) RCLCPP_WARN(this->get_logger(), "crc check error");
        }
        return crc_chk == crc_val;
    }

private:
    // parameters
    SerialDriverNodeParams params_;

    // port
    std::unique_ptr<serial::Serial> port_;
    std::thread read_thread_;
    protocol_header_t protocol_header_;
    protocol_tail_t protocol_tail_;

    // Broadcast tf from odom to gimbal_link
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

    // ros communication
    rclcpp::Publisher<rm_msgs::msg::RmRobot>::SharedPtr robot_pub;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub;
    
    rclcpp::Subscription<rm_msgs::msg::Control>::SharedPtr control_sub;

};
}

#endif // _SERIAL_DRIVER_NODE_H_