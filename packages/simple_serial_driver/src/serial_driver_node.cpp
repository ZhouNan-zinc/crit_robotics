#include <rclcpp/logging.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "serial/serial.h"

#include "simple_serial_driver/serial_protocol.h"
#include "simple_serial_driver/serial_driver_node.h"


namespace ngxy_simple_serial
{

SerialDriverNode::SerialDriverNode(rclcpp::NodeOptions& options)
    :Node("simple_serial_driver", options.automatically_declare_parameters_from_overrides(true)){
    loadParams();

    state_pub = create_publisher<rm_msgs::msg::State>(
        "/robot_state", rclcpp::QoS(10));

    control_sub = create_subscription<rm_msgs::msg::Control>(
        "/control", rclcpp::QoS(10),
        std::bind(&SerialDriverNode::ControlCallback, this, std::placeholders::_1));
    
    initPort();
}

SerialDriverNode::~SerialDriverNode(){
    closePort();
}

void SerialDriverNode::readPortCallback(uint8_t* buffer){
    // 根据不同id, 调用不同callback
    if(buffer[1] == 0x03){
        autoaim_recv_from_port_data_t* data = (autoaim_recv_from_port_data_t*)(buffer + sizeof(protocol_header_t));
        autoaimReadPortCallback(data);
    } else {
        RCLCPP_ERROR(this->get_logger(), 
        "Unknown protocol id: %d, and notice that in fact I killed autolob callback XD", buffer[1]);
    }
}
//ros topic callback

void SerialDriverNode::autoaimReadPortCallback(const autoaim_recv_from_port_data_t* data){
    state_pub->publish(rm_msgs::msg::State()
        .set__imu(rm_msgs::msg::Imu()
            .set__roll(data->roll)
            .set__pitch(data->pitch)
            .set__yaw(data->yaw))
        .set__robot_id(data->robot_id)
        .set__vision_follow_enable(data->mode == 1)
    );
}

void SerialDriverNode::ControlCallback(const rm_msgs::msg::Control::SharedPtr msg){
    autoaim_send_to_port_data_t data;
    data.fromControlMsg(*msg);

    protocol_header.start = 0x7d;
    protocol_header.protocol_id = 0x01;
    protocol_tail.end = 0x7e;

    writeToPort(data);
}

// serial Port
void SerialDriverNode::initPort(){
    try {
        port = std::make_unique<serial::Serial>(
            get_parameter("device_name").as_string(),
            get_parameter("baud_rate").as_int(),
            serial::Timeout::simpleTimeout(1000)
        );
        if (not port->isOpen()) {
            port->open();
        }
        executor = std::thread(&SerialDriverNode::readFromPort, this);
    } catch (const std::exception& ex) {
        RCLCPP_ERROR_STREAM(get_logger(), "Error creating serial port: " << ex.what());
    }
}

void SerialDriverNode::closePort(){
    if (executor.joinable()) {
        executor.join();
    }

    if (port->isOpen()) {
        port->close();
    }
}

void SerialDriverNode::reOpenPort(){
    RCLCPP_WARN(get_logger(), "Attempting to reopen port");
    try {
        if (port->isOpen()) {
            port->close();
        }
        port->open();
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

    memcpy(data_header, &protocol_header, sizeof(protocol_header_t));

    memcpy(data_content, &_data, sizeof(autoaim_send_to_port_data_t));

    protocol_tail.crc16 = CRC16::crc16_ccitt.check_sum(send_buffer + sizeof(protocol_header_t::start), 
                                                    sizeof(protocol_header_t::protocol_id) + data_len);

    memcpy(data_tail, &protocol_tail, sizeof(protocol_tail_t));

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
        port->write(send_buffer_vec);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Send Failed: %s", e.what());
    }
}

void SerialDriverNode::readFromPort(){
    std::vector<uint8_t> read_buffer_vec;
    int header_size = sizeof(protocol_header_t);
    int autoaim_data_size = sizeof(autoaim_recv_from_port_data_t);
    int tail_size = sizeof(protocol_tail_t);

    read_buffer_vec.resize(autoaim_data_size + header_size + tail_size);
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
                        if(read_data_size >= autoaim_data_size + tail_size) break;
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
RCLCPP_COMPONENTS_REGISTER_NODE(SerialDriverNode)