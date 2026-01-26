cmake_minimum_required(VERSION 3.8)
project(simple_serial_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

#平台配置
add_compile_options(-O3)
set(CMAKE_CXX_STANDARD 20)

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rm_interface REQUIRED)
find_package(rm_base REQUIRED)
ament_auto_find_build_dependencies()

# yaml-cpp
find_package(yaml-cpp REQUIRED)
set(include ${YAML_INCLUDE_DIRS} ${include})
set(link ${YAML_CPP_LIBRARIES} ${link})

# OpenCV
find_package(OpenCV 4 REQUIRED)
set(link ${OpenCV_LIBS} ${link})

# Eigen
find_package(Eigen3 3.3 REQUIRED)
set(link Eigen3::Eigen ${link})

file(GLOB_RECURSE src "src/**")
ament_auto_add_library(${PROJECT_NAME} SHARED ${src})
target_include_directories(${PROJECT_NAME} PUBLIC include)
ament_target_dependencies(${PROJECT_NAME} rm_interface)
ament_target_dependencies(${PROJECT_NAME} rm_base)
target_link_libraries(${PROJECT_NAME} ${link})

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN ngxy_simple_serial::SerialDriverNode
  EXECUTABLE ${PROJECT_NAME}_node
)

add_executable(serial_test src/test.cpp)
target_link_libraries(serial_test ${PROJECT_NAME})

install(TARGETS serial_test
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  # config
  # launch
)

