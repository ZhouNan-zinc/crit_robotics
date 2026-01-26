#pragma once

#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/subscription.hpp>

#include <simple_serial_driver/serial_protocol.h>
#include <simple_serial_driver/crc.h>
#include <serial/serial.h>

#include <rm_msgs/msg/control.hpp>
#include <rm_msgs/msg/state.hpp>

static const std::vector<std::string> dev_names = {"/dev/ttyUSB0", "/dev/ttyACM0"};

class SerialDriverNode : public rclcpp::Node{
public:
    explicit SerialDriverNode(rclcpp::NodeOptions& options);

    ~SerialDriverNode() override;

    void loadParams();

    //ros topic callback
    void readPortCallback(uint8_t* buffer);
    void autoaimReadPortCallback(const autoaim_recv_from_port_data_t* data);
    void ControlCallback(const rm_msgs::msg::Control::SharedPtr msg);

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
        int res = port->read(buffer, size);
        if(res != size)
            RCLCPP_WARN(get_logger(),"Read Failed, read: %d", res);
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
    // port
    std::unique_ptr<serial::Serial> port;
    std::thread executor;
    protocol_header_t protocol_header;
    protocol_tail_t protocol_tail;

    // ros communication
    rclcpp::Publisher<rm_msgs::msg::State>::SharedPtr state_pub;
    rclcpp::Subscription<rm_msgs::msg::Control>::SharedPtr control_sub;
};