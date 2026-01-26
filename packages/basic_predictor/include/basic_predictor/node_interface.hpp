#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <rclcpp/node.hpp>

#include <vision_msgs/msg/detection2_d_array.hpp>
#include <message_filters/subscriber.hpp>
#include <message_filters/cache.hpp>

#include <rm_msgs/msg/imu_stamped.hpp>
#include <rm_msgs/msg/state.hpp>
#include <rm_msgs/msg/control.hpp>
#include <rm_msgs/srv/set_vision_mode.hpp>
#include <rm_msgs/srv/toggle_vision_follow_id.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <foxglove_msgs/msg/color.hpp>
#include <foxglove_msgs/msg/image_annotations.hpp>
#include <foxglove_msgs/msg/point2.hpp>

#include "basic_predictor/ballestic_utils.hpp"

class PredictorInterface {
public:
    virtual ~PredictorInterface() noexcept = default;
};

class EnemyPredictor: public PredictorInterface {
public:
    explicit EnemyPredictor(const rclcpp::Node* node);
};

class EnergyPredictor: public PredictorInterface {
public:
    explicit EnergyPredictor(const rclcpp::Node* node);
};

class OutpostPredictor: public PredictorInterface {
public:
    explicit OutpostPredictor(const rclcpp::Node* node);
};

class BasePredictor: public PredictorInterface {
public:
    explicit BasePredictor(const rclcpp::Node* node);
};

class PredictorNodeInterface: public rclcpp::Node {
public:
    PredictorNodeInterface();

    rm_msgs::msg::Imu get_imu(rclcpp::Time stamp = rclcpp::Time(0, 0, RCL_ROS_TIME));

    uint8_t get_vision_follow_id();
    uint8_t get_robot_id();

    BallisticResult solve_ballestic(geometry_msgs::msg::Point point);

    void send_command(const rm_msgs::msg::Imu& imu, uint8_t control_mode, bool booster_enable);

    void annotate(const std::vector<geometry_msgs::msg::Pose>& poses);
    void annotate(const geometry_msgs::msg::Pose& poses);

    std::shared_ptr<std::function<void(std::shared_ptr<vision_msgs::msg::Detection2DArray>)>>
    register_callback(const std::function<void(std::shared_ptr<vision_msgs::msg::Detection2DArray>)>& callback);

private:
    static constexpr const char* kNodeName = "basic_predictor";

    // polymorphism
    std::unique_ptr<PredictorInterface> predictor;

    // callback handle
    std::weak_ptr<std::function<void(std::shared_ptr<vision_msgs::msg::Detection2DArray>)>> vision_callback;

    // message cache
    std::shared_ptr<sensor_msgs::msg::CameraInfo> camera_info;
    std::shared_ptr<message_filters::Cache<rm_msgs::msg::ImuStamped>> imu_stamped_cache;
    std::shared_ptr<rm_msgs::msg::State> robot_state;
    std::shared_ptr<foxglove_msgs::msg::ImageAnnotations> image_annotations;
    uint8_t vision_follow_id;

    // publishers
    std::shared_ptr<rclcpp::Publisher<rm_msgs::msg::Control>> control_pub;
    std::shared_ptr<rclcpp::Publisher<foxglove_msgs::msg::ImageAnnotations>> image_annotations_pub;

    // subscriptions
    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>> camera_info_sub;
    std::shared_ptr<rclcpp::Subscription<rm_msgs::msg::State>> state_sub;
    std::shared_ptr<rclcpp::Subscription<vision_msgs::msg::Detection2DArray>> vision_sub;
    std::shared_ptr<rclcpp::Subscription<rm_msgs::msg::Control>> ext_control_sub;

    // services
    std::shared_ptr<rclcpp::Service<rm_msgs::srv::SetVisionMode>> set_vision_mode_srv;
    std::shared_ptr<rclcpp::Service<rm_msgs::srv::ToggleVisionFollowId>> toggle_vision_follow_id_srv;

    // clients
    std::shared_ptr<rclcpp::Client<rm_msgs::srv::ToggleVisionFollowId>> toggle_vision_follow_id_cli;

    // ballestic solver
    /// NOTE: To be honest, I dont want to couple ballestic and node functions but I have to.
    std::shared_ptr<BallisticSolver> ballestic_solver;
};
