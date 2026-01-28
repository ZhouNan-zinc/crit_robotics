cmake_minimum_required(VERSION 3.8)
project(enemy_predictor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

#平台配置
add_compile_options(-O3)
set(CMAKE_CXX_STANDARD 20)

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# OpenCV
find_package(OpenCV 4 REQUIRED)
set(link ${OpenCV_LIBS} ${link})

# Eigen
find_package(Eigen3 3.3 REQUIRED)
set(link Eigen3::Eigen ${link})

find_package(cv_bridge REQUIRED)  # 添加这一行
find_package(sensor_msgs REQUIRED)  # 添加这一行
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(rm_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(Ceres REQUIRED)
include_directories(
  ${CERES_INCLUDE_DIRS}
)
# find ament_cmake
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
# find src
file(GLOB_RECURSE src "src/*.cpp")
set(src
  src/enemy_predictor_node.cpp
  src/armor_filter.cpp
  src/enemy_filter.cpp
  src/enemy_calculate.cpp
  src/enemy_ballistic.cpp
  src/enemy_sync.cpp
  src/enemy_trajctory.cpp
)


ament_auto_add_library(${PROJECT_NAME} SHARED ${src})
target_include_directories(${PROJECT_NAME} PUBLIC include ${include})

target_link_libraries(
  ${PROJECT_NAME} 
  cv_bridge::cv_bridge
  image_transport::image_transport
  yaml-cpp::yaml-cpp
  ${link} ${CERES_LIBRARIES}
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN EnemyPredictorNode
  EXECUTABLE ${PROJECT_NAME}_node
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
