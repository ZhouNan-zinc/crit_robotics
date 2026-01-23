#include "simple_serial_driver/serial_protocol.h"
#include <algorithm>
#include <cstdint>
#include <simple_serial_driver/serial_driver_node.h>
#include <std_msgs/msg/detail/int8__struct.hpp>

namespace ngxy_simple_serial
{


enum VisionMode {
    NO_AIM = 0,    // 无瞄准
    AUTO_AIM,  // 普通自瞄，打车，打基地
    OUTPOST_AIM,   // 前哨站
    OUTPOST_LOB,   // 吊射前哨站
    S_WM,      // 小风车
    B_WM,      // 大风车
    LOB,       // 吊射
    HALT,      // 停机
    AUTOLOB,   // 自动吊射
    Unknown,   // 未知
};


std::string vision_mode2string(VisionMode mode){
    switch (mode) {
        case NO_AIM:
            return "NO_AIM";
        case AUTO_AIM:
            return "AUTO_AIM";
        case OUTPOST_AIM:
            return "OUTPOST_AIM";
        case OUTPOST_LOB:
            return "OUTPOST_LOB";
        case S_WM:
            return "S_WM";
        case B_WM:
            return "B_WM";
        case LOB:
            return "LOB";
        case HALT:
            return "HALT";
        case AUTOLOB:
            return "AUTOLOB";
        case Unknown:
        default:
            return "Unknown";
    }
}


VisionMode get_vision_mode(uint8_t mode){
    switch (mode) {
        case 0:
            return AUTO_AIM;  // NO_AIM
        case 1:
            return AUTO_AIM;
        case 2:
            return OUTPOST_AIM;
        case 3:
            return OUTPOST_LOB;
        case 4:
            return S_WM;
        case 5:
            return B_WM;
        case 6:
            return HALT;
        case 7:
        default:
            return Unknown;
    }
}


void SerialDriverNode::loadParams(){
    params_.timestamp_offset = this->declare_parameter("timestamp_offset", 0.0);

    params_.robot_name = this->declare_parameter("robot_name", "114514_robot");
    
    params_.pitch2yaw_t = this->declare_parameter("pitch2yaw_t", std::vector<double>({0, 0, 0}));
    assert(params_.pitch2yaw_t.size() == 3 && "pitch2yaw_t should be a 3-dimensional vector");
    
    params_.device_name = declare_parameter<std::string>("device_name", "");

    params_.baud_rate = declare_parameter<int>("baud_rate", 0);
}

SerialDriverNode::SerialDriverNode(const rclcpp::NodeOptions& _options):Node("simple_serial_driver", _options){
    RCLCPP_INFO(this->get_logger(), "SerialDriverNode started");
    loadParams();

    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Robot publisher
    robot_pub = create_publisher<rm_msgs::msg::RmRobot>(params_.robot_name, rclcpp::SensorDataQoS());

    joint_state_pub =
        create_publisher<sensor_msgs::msg::JointState>("joint_states", rclcpp::SensorDataQoS());

    // Control subscription
    control_sub = create_subscription<rm_msgs::msg::Control>(
        "enemy_predictor", rclcpp::SensorDataQoS(),
        std::bind(&SerialDriverNode::ControlCallback, this, std::placeholders::_1));
    
    initPort();
}

SerialDriverNode::~SerialDriverNode(){
    closePort();
}

void SerialDriverNode::readPortCallback(uint8_t* buffer){
    // 根据不同id, 调用不同callback
    if(buffer[1] == 0x03){
        autoaim_recv_from_port_data_t* rmsg = (autoaim_recv_from_port_data_t*)(buffer + sizeof(protocol_header_t));
        autoaimReadPortCallback(rmsg);
    }else if(buffer[1] == 0x04){
        autolob_recv_from_port_data_t* lmsg = (autolob_recv_from_port_data_t*)(buffer + sizeof(protocol_header_t));
        autolonReadPortCallback(lmsg);
    }else{
        RCLCPP_ERROR(this->get_logger(), "Unknown protocol id: %d", buffer[1]);
    }
}
//ros topic callback

void SerialDriverNode::autoaimReadPortCallback(const autoaim_recv_from_port_data_t* _data){
    // RCLCPP_INFO(get_logger(), "\033[0;32m AutoAim Recv From Port\033[0m");
    static long long last_time = -1;
    static long long now_time;
    now_time = this->now().nanoseconds();
    if (last_time != -1) {
        long long diff_time = now_time - last_time;
        if ( diff_time / (double)1000000 > 100) {
            RCLCPP_INFO(get_logger(), "too big diff time!: %lf ms", diff_time / (double)1000000);
        }
    }
    last_time = now_time;

    // 仅适用于pitch和yaw相交的云台
    tf2::Quaternion pitch2gimbal_r;
    pitch2gimbal_r.setRPY(_data->roll, _data->pitch, _data->yaw);
    tf2::Transform pitch2gimbal(pitch2gimbal_r);
    // tf2::Quaternion pitch2yaw_r, yaw2gimbal_r;
    // pitch2yaw_r.setRPY(0, _data->pitch, 0);
    // yaw2gimbal_r.setRPY(0, 0, _data->yaw);
    // tf2::Transform pitch2yaw(pitch2yaw_r, tf2::Vector3(params_.pitch2yaw_t[0], params_.pitch2yaw_t[1], params_.pitch2yaw_t[2]));
    // tf2::Transform yaw2gimbal(yaw2gimbal_r);
    // tf2::Transform pitch2gimbal = yaw2gimbal * pitch2yaw;

    geometry_msgs::msg::TransformStamped t;
    params_.timestamp_offset = this->get_parameter("timestamp_offset").as_double();
    t.header.stamp = this->now() + rclcpp::Duration::from_seconds(params_.timestamp_offset);
    t.header.frame_id = "gimbal";
    t.child_frame_id = "pitch_link";
    t.transform = tf2::toMsg(pitch2gimbal);
    tf_broadcaster->sendTransform(t);

    sensor_msgs::msg::JointState joint_msg;
    joint_msg.header.stamp = this->now() + rclcpp::Duration::from_seconds(params_.timestamp_offset);
    std::vector<std::string> names = {"leg_joint"};
    std::vector<double> positions = {_data->height};
    joint_msg.name = names;
    joint_msg.position = positions;
    joint_state_pub->publish(joint_msg);

    rm_msgs::msg::RmRobot robot_msg;
    robot_msg.z_velocity = _data->z_velocity;
    robot_msg.height = _data->height;
    robot_msg.robot_id = _data->robot_id;
    robot_msg.bullet_velocity = _data->bullet_speed;

    robot_msg.vision_mode = vision_mode2string(get_vision_mode(_data->mode));
    robot_msg.right_press = _data->mode == 1;
    robot_msg.switch_cam = _data->switch_cam == 1;
    robot_msg.autoshoot_rate = _data->autoshoot_rate;
    robot_msg.priority_type_arr.assign(
        _data->priority_type_arr,
        _data->priority_type_arr + sizeof(_data->priority_type_arr) / sizeof(uint8_t));
    robot_msg.priority_level_arr.assign(
        _data->priority_level_arr,
        _data->priority_level_arr + sizeof(_data->priority_level_arr) / sizeof(uint8_t));
    robot_msg.imu.roll = _data->roll;
    robot_msg.imu.pitch = _data->pitch;
    robot_msg.imu.yaw = _data->yaw;
    // robotpub_low_freq(robot_msg);
    robot_pub->publish(robot_msg);
}

void SerialDriverNode::ControlCallback(const rm_msgs::msg::Control::SharedPtr _msg){
    autoaim_send_to_port_data_t data;
    data.fromControlMsg(*_msg);

    protocol_header_.start = 0x7d;
    protocol_header_.protocol_id = 0x01;
    protocol_tail_.end = 0x7e;
    writeToPort(data);
    RCLCPP_INFO(get_logger(), "Publish Control Msg");
}

// serial Port
void SerialDriverNode::initPort(){
    // 检查串口设备是否存在
    // std::string use_device_name = "";
    // if(params_.device_name != "" && isDeviceValid(params_.device_name)){
    //     use_device_name = params_.device_name;
    // }else{
    //     for(auto dn : dev_names){
    //         if(isDeviceValid(dn)){
    //             use_device_name = dn;
    //             break;
    //         }
    //     }
    // }

    // if(use_device_name == ""){
    //     // RCLCPP_ERROR(get_logger(), "No valid serial port found");
    //     throw std::runtime_error("No valid serial port found");
    // }

    try {
        port_ = std::make_unique<serial::Serial>(params_.device_name, params_.baud_rate,
                                                        serial::Timeout::simpleTimeout(1000));
        if (!port_->isOpen()) {
            port_->open();
        }
        read_thread_ = std::thread(&SerialDriverNode::readFromPort, this);
    } catch (const std::exception& ex) {
        RCLCPP_ERROR(get_logger(), "Error creating serial port: %s - %s", params_.device_name.c_str(),
                    ex.what());
        throw ex;
    }
}

void SerialDriverNode::closePort(){
    if (read_thread_.joinable()) {
        read_thread_.join();
    }

    if (port_->isOpen()) {
        port_->close();
    }
}

void SerialDriverNode::reOpenPort(){
    RCLCPP_WARN(get_logger(), "Attempting to reopen port");
    try {
        if (port_->isOpen()) {
            port_->close();
        }
        port_->open();
        RCLCPP_INFO(get_logger(), "Successfully reopened port");
    } catch (const std::exception& ex) {
        RCLCPP_ERROR(get_logger(), "Error while reopening port: %s", ex.what());
        if (rclcpp::ok()) {
            rclcpp::sleep_for(std::chrono::seconds(1));
            reOpenPort();
        }
    }
}

void SerialDriverNode::writeToPort(autoaim_send_to_port_data_t _data){
    static std::vector<uint8_t> send_buffer_vec;

    static int data_len = sizeof(autoaim_send_to_port_data_t);
    static int header_len = sizeof(protocol_header_t);
    static int tail_len = sizeof(protocol_tail_t);
    
    static int buffer_len = data_len + header_len + tail_len;

    send_buffer_vec.resize(buffer_len);
    uint8_t* send_buffer = send_buffer_vec.data();

    protocol_header_t* data_header = (protocol_header_t*)(send_buffer);
    autoaim_send_to_port_data_t* data_content = (autoaim_send_to_port_data_t*)(send_buffer + header_len);
    protocol_tail_t* data_tail = (protocol_tail_t*)(send_buffer + header_len + data_len);

    memcpy(data_header, &protocol_header_, sizeof(protocol_header_t));

    memcpy(data_content, &_data, sizeof(autoaim_send_to_port_data_t));

    protocol_tail_.crc16 = CRC16::crc16_ccitt.check_sum(send_buffer + sizeof(protocol_header_t::start), 
                                                    sizeof(protocol_header_t::protocol_id) + data_len);

    memcpy(data_tail, &protocol_tail_, sizeof(protocol_tail_t));

    // 转义处理 不包括头尾
    static std::vector<std::pair<size_t, uint8_t>> escape_pairs;
    static size_t has_excape_cnts;
    escape_pairs.clear();
    has_excape_cnts = 0;

    for(int i = 1; i < buffer_len - 1; ++i){
        if(send_buffer[i] == 0x7d || send_buffer[i] == 0x7e || send_buffer[i] == 0x7f){
            // printf("escape %d\n", i);
            escape_pairs.emplace_back(i, (send_buffer[i] - 0x7d));
        }
    }

    for(const auto& ep : escape_pairs){
        send_buffer_vec[ep.first + has_excape_cnts] = 0x7f;
        send_buffer_vec.insert(send_buffer_vec.begin() + ep.first + has_excape_cnts + 1, ep.second);
        has_excape_cnts += 1;
    }

    try {
        port_->write(send_buffer_vec);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Send Failed: %s", e.what());
    }
}

void SerialDriverNode::readFromPort(){
    std::vector<uint8_t> read_buffer_vec;
    int header_size = sizeof(protocol_header_t);
    int autoaim_data_size = sizeof(autoaim_recv_from_port_data_t);
    int autolob_data_size = sizeof(autolob_recv_from_port_data_t);
    int tail_size = sizeof(protocol_tail_t);

    read_buffer_vec.resize(std::max(autoaim_data_size, autolob_data_size) + header_size + tail_size);
    uint8_t* buffer = read_buffer_vec.data();

    int no_serial_data = 0;

    while (rclcpp::ok()) 
    {
        try 
        {   
            bool is_success = this->read(buffer, 1);
            if(is_success && buffer[0] == 0x7d)
            {   
                // NGXY_DEBUG("correct start");
                is_success = this->read(buffer+1, 1);
                
                if(is_success && (buffer[1] == 0x03 || buffer[1] == 0x04))
                {   
                    // NGXY_DEBUG( "correct id");
                    uint8_t* read_end = buffer+1;
                    uint8_t read_data_size = 0;
                    // 循环读直到读到0x7e
                    while(read_end[0] != 0x7e){
                        if(read_data_size >= std::max(autoaim_data_size, autolob_data_size) + tail_size) break;
                        ++read_end;
                        is_success = this->read(read_end, 1);
                        if (read_end[0] == 0x7f) { // 转义处理
                            is_success = this->read(read_end, 1);
                            read_end[0] = read_end[0] + 0x7d;
                        }
                        if(is_success)++read_data_size;
                        else break;
                    }
                    // NGXY_DEBUG("read count %d", read_data_size);
                    if(is_success && read_end[0] == 0x7e){
                        // NGXY_DEBUG("correct end");
                        bool is_size_correct = false;
                        if(buffer[1] == 0x03) 
                            is_size_correct = read_data_size - tail_size == autoaim_data_size;
                        else if(buffer[1] == 0x04) 
                            is_size_correct = read_data_size - tail_size == autolob_data_size;
                        else
                            RCLCPP_WARN(get_logger(), "Read Error Protocol");
                        if(is_size_correct && 
                            buffer_check_valid(buffer + sizeof(protocol_header_t::start), 
                            read_data_size, 
                            CRC16::crc16_ccitt)
                            ) 
                            readPortCallback(buffer);
                    }
                }
            }

            if(!is_success) ++no_serial_data;

            if (no_serial_data > 5) {
                RCLCPP_WARN(get_logger(), "no serial data....");
                no_serial_data = 0;
            }

        } catch (const std::exception& ex) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 20, "Error while receiving data: %s",
                                  ex.what());
            reOpenPort();
        }
    }
}

}

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(ngxy_simple_serial::SerialDriverNode)