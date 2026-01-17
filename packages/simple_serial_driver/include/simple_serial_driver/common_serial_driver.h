#ifndef _SERIAL_DRIVER_H
#define _SERIAL_DRIVER_H

#include <cassert>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logger.hpp>
#include <serial/serial.h>

#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <functional>

#include <simple_serial_driver/crc.h>
#include <simple_serial_driver/serial_protocol.h>

template <typename Object>
concept IsBaseOnNode = std::is_base_of_v<rclcpp::Node, Object>;   

template<typename sendT, typename recvT>
class SerialDriver{
public:
   SerialDriver() = default;

   void init();

   void write(sendT data);

   void read();

   void reopen();

private:
   std::unique_ptr<serial::Serial> port_;
   
};




#endif