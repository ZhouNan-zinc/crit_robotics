cmake_minimum_required(VERSION 3.28)
project(udp_socket)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Protobuf REQUIRED)

set(PROTO_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/message.proto
)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

add_executable(${PROJECT_NAME}
    src/node_interface.cpp
    src/main.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    rclcpp::rclcpp
    ${std_msgs_TARGETS}
)

install(
    TARGETS ${PROJECT_NAME}
    DESTINATION lib/${PROJECT_NAME}
)

install(
    DIRECTORY proto launch config
    DESTINATION share/${PROJECT_NAME}
)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # There is a known issue with ROS2 and fsanitize. See ...
    # target_compile_options(${PROJECT_NAME} PRIVATE -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer)
    # target_link_options(${PROJECT_NAME} PRIVATE -fsanitize=address,undefined)
    target_compile_options(${PROJECT_NAME} PRIVATE -g -O0)

else()
    add_link_options(-flto=auto)
    target_compile_options(${PROJECT_NAME} PRIVATE -O3 -mtune=native -flto)
    target_link_options(${PROJECT_NAME} PRIVATE -flto)
    
endif()

ament_package()
