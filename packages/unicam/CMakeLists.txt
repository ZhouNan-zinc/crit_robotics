cmake_minimum_required(VERSION 3.28)
project(unicam)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(MVS REQUIRED)

add_executable(${PROJECT_NAME}
    src/hik_usb_cam.cpp
    src/node_interface.cpp
    src/main.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PUBLIC 
    ${OpenCV_INCLUDE_DIRS}
    ${MVS_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${OpenCV_LIBS}
    ${MVS_LIBS}
    ${sensor_msgs_TARGETS}
    camera_info_manager::camera_info_manager
    image_transport::image_transport
    rclcpp::rclcpp
    sensor_msgs::sensor_msgs_library
)

install(
    TARGETS ${PROJECT_NAME}
    DESTINATION lib/${PROJECT_NAME}
)

install(
    DIRECTORY calibration launch config
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

elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_link_options(-flto=auto)
    target_compile_options(${PROJECT_NAME} PRIVATE -O3 -mtune=native -flto)
    target_link_options(${PROJECT_NAME} PRIVATE -flto)
endif()

ament_package()
