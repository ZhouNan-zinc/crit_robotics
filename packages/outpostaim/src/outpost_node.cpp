#include "outpostaim/outpost_node.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2_ros/create_timer_ros.h>
#include <cmath>

OutpostNode::OutpostNode(const rclcpp::NodeOptions& options)
    : Node("outpostaim", options),  manager_() {
    
    RCLCPP_INFO(this->get_logger(), "OutpostPredictor Start!");
    loadAllParams();

    // 初始化 TF2 缓冲区
    tf2_buffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    
    // 创建 TF2 定时器接口
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_buffer->setCreateTimerInterface(timer_interface);
    tf2_listener = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer);

    std::string shared_dir = ament_index_cpp::get_package_share_directory("outpostaim");
    RCLCPP_INFO(get_logger(), "shared_dir: %s", shared_dir.c_str());

    bac = std::make_shared<Ballistic>(this, shared_dir);
    //初始化为小弹,弹速30
    bac->refresh_velocity(false, 30.);

    // 初始化manager_.outpost指向outpost_
    manager_.outpost = outpost;

    // 注册重投影回调到 manager_，便于在 OutpostManager 中绘制重投影结果
    manager_.projector = std::bind(&OutpostNode::projectPointToImage, this, std::placeholders::_1);

    off_cmd = createControlMsg(0, 0, 0, 0, 0/*, 15, send_cam_mode(params_.cam_mode)*/);

    // 订阅识别检测结果话题
    detection_sub = this->create_subscription<vision_msgs::msg::Detection2DArray>(
        params_.detection_topic, rclcpp::SensorDataQoS(), 
        std::bind(&OutpostNode::detectionCallback, this, std::placeholders::_1));

    // 订阅相机图像与内参
    camera_sub = image_transport::create_camera_subscription(
        this,
        "/hikcam/image_raw",
        [this](auto image_msg, auto camera_info_msg) {
            this->camera_callback(image_msg, camera_info_msg);
        },
        "raw"
    );

    robot_sub = this->create_subscription<RmRobotMsg>(params_.robot_topic, rclcpp::SensorDataQoS(),
        std::bind(&OutpostNode::robotCallback, this, std::placeholders::_1));

    // 创建控制指令发布器
    control_pub = this->create_publisher<ControlMsg>("enemy_predictor", rclcpp::SensorDataQoS());
    
    target_dis_pub = this->create_publisher<std_msgs::msg::Float64>("target_dis", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("outpost_vis", 10);

    CLK = rclcpp::Clock{RCL_STEADY_TIME};
}

void OutpostNode::loadAllParams() {

    params_.mode = VisionMode::AUTO_AIM;

    params_.detection_topic = this->declare_parameter("detection_topic", "vision/tracked");
    params_.robot_topic = this->declare_parameter("robot_topic", "rm_robot");
    params_.control_topic = this->declare_parameter("control_topic", "control"); // 暂时没用
    params_.image_topic = this->declare_parameter("image_topic", "hikcam/image_raw");
    params_.camera_info_topic = this->declare_parameter("camera_info_topic", "hikcam/camera_info");
    params_.target_frame = this->declare_parameter("target_frame", "odom");
    params_.camera_frame = this->declare_parameter("camera_frame", "camera_optical_frame");

    params_.top_offset_z = this->declare_parameter("top_offset_z", 0.5);
    params_.top_offset_dis = this->declare_parameter("top_offset_dis", 0.5);
    params_.size_ratio_thresh = this->declare_parameter("size_ratio_thresh", 0.5);
    params_.timestamp_thresh = this->declare_parameter("timestamp_thresh", 0.5);
    params_.midshot_period = declare_parameter("midshot_period", 0.02);

    params_.hero_fixed_armor_id = this->declare_parameter("hero_fixed_armor_id", 0);
    params_.enable_hero_dynamic_selection = declare_parameter("enable_hero_dynamic_selection", true);

    // 火控参数
    params_.change_armor_time_thresh = declare_parameter("change_armor_time_thresh", 0.0);
    params_.dis_yaw_thresh = declare_parameter("dis_yaw_thresh", 0.0);
    params_.dis_thresh_kill = declare_parameter("dis_thresh_kill", 0.0);
    params_.low_spd_thresh = declare_parameter("low_spd_thresh", 0.0);
    params_.gimbal_error_dis_thresh = declare_parameter("gimbal_error_dis_thresh", 0.0);
    params_.gimbal_error_dis_thresh_old = declare_parameter("gimbal_error_dis_thresh_old", 0.0);
    params_.residual_thresh = declare_parameter("residual_thresh", 0.0);
    params_.tangential_spd_thresh = declare_parameter("tangential_spd_thresh", 0.0);
    params_.normal_spd_thresh = declare_parameter("normal_spd_thresh", 0.0);
    params_.decel_delay_time = declare_parameter("decel_delay_time", 0.0);
    params_.choose_enemy_without_autoaim_signal = declare_parameter("choose_enemy_without_autoaim_signal", false);
    params_.disable_auto_shoot = declare_parameter("disable_auto_shoot", false);
    // 延迟参数
    params_.response_delay = declare_parameter("response_delay", 0.0);
    params_.shoot_delay = declare_parameter("shoot_delay", 0.0);
    params_.gimbal_adjust_delay = declare_parameter("gimbal_adjust_delay", 0.3);  //云台调整延迟

    params_.robot_2armor_dis_thresh = this->declare_parameter("robot_2armor_dis_thresh", 0.5);

    params_.rmcv_id.robot_id = RobotId::ROBOT_ERROR;
    
    // 滤波器参数
    // armor_ekf
    std::vector<double> vec_Q = this->declare_parameter("armor_ekf.Q", std::vector<double>());
    std::vector<double> vec_R = this->declare_parameter("armor_ekf.R", std::vector<double>());
    OutpostArmorEkf::config_.vec_Q = OutpostArmorEkf::Vx(vec_Q.data());
    OutpostArmorEkf::config_.vec_R = OutpostArmorEkf::Vz(vec_R.data());
    
    // yaw_ekf
    vec_R = this->declare_parameter("yaw_ekf.R", std::vector<double>());
    OutpostYawEkf::config_.vec_R = OutpostYawEkf::Vz(vec_R.data());
    OutpostYawEkf::config_.sigma2_Q = this->declare_parameter("yaw_ekf.Q", 0.01);

    // enemy_ckf
    OutpostCkf::config_.Q2_XY = this->declare_parameter("Q2_XY", 0.01);
    OutpostCkf::config_.Q2_YAW = this->declare_parameter("Q2_YAW", 0.01);
    OutpostCkf::config_.R_XYZ = this->declare_parameter("R_XYZ", 0.01);
    OutpostCkf::config_.R_YAW = this->declare_parameter("R_YAW", 0.01);
    std::vector<double> vec_p = this->declare_parameter("P", std::vector<double>());
    OutpostCkf::config_.init_P = OutpostCkf::Vx(vec_p.data());
    OutpostCkf::const_dis_ = this->declare_parameter("const_dis", 0.2765);
    RCLCPP_INFO(this->get_logger(), "filter params loaded");
    
    // 管理参数
    params_.census_period_min = this->declare_parameter("census_period_min", 0.5);
    params_.census_period_max = this->declare_parameter("census_period_max", 0.5);

    params_.anti_outpost_census_period 
                        = this->declare_parameter("anti_outpost_census_period", 0.5);
    
    params_.anti_outpost_census_period_max 
                        = this->declare_parameter("anti_outpost_census_period_max", 0.5);
    
    params_.anti_outpost_census_period_min
                        = this->declare_parameter("anti_outpost_census_period_min", 0.5);

    params_.top_pitch_thresh = this->declare_parameter("top_pitch_thresh", 0.5);

    params_.sight_limit = this->declare_parameter("sight_limit", 0.5);
    params_.high_limit = this->declare_parameter("high_limit", 0.5);
    params_.size_limit = this->declare_parameter("size_limit", 0.5);
    params_.bound_limit = this->declare_parameter("bound_limit", 0.5);
    params_.aspect_limit_big = this->declare_parameter("aspect_limit_big", 0.5);
    params_.aspect_limit_small = this->declare_parameter("aspect_limit_small", 0.5);
    params_.reset_time = this->declare_parameter("reset_time", 0.5);
    std::vector<double> collimation_vec = declare_parameter("collimation", std::vector<double>());
    assert(collimation_vec.size() == 2 && "collimation size must be 2!");
    params_.collimation.x = collimation_vec[0];
    params_.collimation.y = collimation_vec[1];

    params_.interframe_dis_thresh = this->declare_parameter("interframe_dis_thresh", 0.5);
    params_.id_inertia = this->declare_parameter("id_inertia", 0);

    params_.enable_imshow = this->declare_parameter("enable_imshow", false);
    params_.debug = this->declare_parameter("debug", false);

    RCLCPP_INFO(this->get_logger(), "outpost_manager over");

    // 将节点加载到的参数同步到 manager_，使 OutpostManager 使用相同的参数
    manager_.params = params_;
}


Eigen::Isometry3d OutpostNode::getTrans(const std::string& source_frame, const std::string& target_frame) {
    
    geometry_msgs::msg::TransformStamped t = tf2_buffer->lookupTransform(
        target_frame, 
        source_frame, 
        rclcpp::Time(0),
        rclcpp::Duration::from_seconds(0.5));
    // to Eigen::Isometry3d
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() = Eigen::Vector3d(
        t.transform.translation.x,
        t.transform.translation.y,
        t.transform.translation.z
    );
    Eigen::Quaterniond quat(
        t.transform.rotation.w,
        t.transform.rotation.x,
        t.transform.rotation.y,
        t.transform.rotation.z
    );
    transform.rotate(quat);
    
    return transform;
}

Eigen::Vector3d OutpostNode::transPoint(const Eigen::Vector3d& source_point, 
                                        const std::string& source_frame, 
                                        const std::string& target_frame) {
    // 使用 getTrans 返回的变换（从 source_frame 到 target_frame）直接乘以向量
    try {
        Eigen::Isometry3d tf = getTrans(source_frame, target_frame);
        return tf * source_point;
    } catch (const std::exception &e) {
        RCLCPP_WARN(get_logger(), "transPoint transform error: %s", e.what());
        return Eigen::Vector3d::Zero();
    }
}

cv::Point2d OutpostNode::projectPointToImage(const Eigen::Vector3d& point_odom) {
    // 需要相机内参
    if (manager_.camera_k_.size() < 9) {
        return cv::Point2d(-1, -1);
    }

    // 将点从 odom 投影到相机坐标系
    Eigen::Vector3d p_cam = transPoint(point_odom, "odom", params_.camera_frame);
    if (p_cam[2] <= 1e-6) return cv::Point2d(-1, -1);

    Eigen::Matrix3d K;
    for (int i = 0; i < 9; ++i) K(i / 3, i % 3) = manager_.camera_k_[i];

    Eigen::Vector3d proj = K * p_cam;
    proj /= proj[2];
    return cv::Point2d(proj[0], proj[1]);
}

bool OutpostNode::transformPoseToOdom(const geometry_msgs::msg::Pose& pose_camera, 
                                              const std_msgs::msg::Header& header,
                                              geometry_msgs::msg::Pose& pose_odom) {
    try {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.pose = pose_camera;
        
        geometry_msgs::msg::PoseStamped transformed_pose;
        transformed_pose = tf2_buffer->transform(pose_stamped, "odom", 
                                                tf2::Duration(std::chrono::milliseconds(100)));
        
        pose_odom = transformed_pose.pose;
        return true;
    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(get_logger(), "Transform error: %s", ex.what());
        return false;
    }
}

bool OutpostNode::convertCameraToOdom(const Eigen::Vector3d& pos_camera,
                                             const Eigen::Quaterniond& ori_camera,
                                             const builtin_interfaces::msg::Time& stamp,
                                             Eigen::Vector3d& pos_odom,
                                             Eigen::Quaterniond& ori_odom) {
    try {
        // 将Eigen类型转换为ROS类型
        geometry_msgs::msg::PoseStamped pose_camera_stamped;
        pose_camera_stamped.header.frame_id = params_.camera_frame;
        pose_camera_stamped.header.stamp = stamp;
        
        pose_camera_stamped.pose.position.x = pos_camera.x();
        pose_camera_stamped.pose.position.y = pos_camera.y();
        pose_camera_stamped.pose.position.z = pos_camera.z();
        
        pose_camera_stamped.pose.orientation.w = ori_camera.w();
        pose_camera_stamped.pose.orientation.x = ori_camera.x();
        pose_camera_stamped.pose.orientation.y = ori_camera.y();
        pose_camera_stamped.pose.orientation.z = ori_camera.z();
        
        // 转换到odom系
        geometry_msgs::msg::PoseStamped pose_odom_stamped;
        pose_odom_stamped = tf2_buffer->transform(pose_camera_stamped, "odom", 
                                                 tf2::Duration(std::chrono::milliseconds(100)));
        
        // 转换回Eigen类型
        pos_odom = Eigen::Vector3d(
            pose_odom_stamped.pose.position.x,
            pose_odom_stamped.pose.position.y,
            pose_odom_stamped.pose.position.z
        );
        
        ori_odom = Eigen::Quaterniond(
            pose_odom_stamped.pose.orientation.w,
            pose_odom_stamped.pose.orientation.x,
            pose_odom_stamped.pose.orientation.y,
            pose_odom_stamped.pose.orientation.z
        );
        
        return true;
    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(get_logger(), "Transform error: %s", ex.what());
        return false;
    }
}

void OutpostNode::camera_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg, 
                                        const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info_msg){
    if (!image_msg || !camera_info_msg) {
        RCLCPP_ERROR(get_logger(), "Received null image or camera info message!");
        return;
    }
    
    // 保存当前图像用于可视化
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        current_image_ = cv_ptr->image;
        manager_.setImage(current_image_);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    
    // 保存相机内参
    if (!camera_info_received_) {
        camera_info_received_ = true;
        std::vector<double> k(9);
        std::vector<double> d(5);
        
        for (int i = 0; i < 9; ++i) {
            k[i] = camera_info_msg->k[i];
        }
        
        if (camera_info_msg->d.size() >= 5) {
            for (int i = 0; i < 5; ++i) {
                d[i] = camera_info_msg->d[i];
            }
        } else {
            RCLCPP_WARN(get_logger(), "Camera distortion parameters size < 5");
            d = std::vector<double>(5, 0.0);
        }
        
        manager_.setCameraInfo(k, d);
    }
}

void OutpostNode::detectionCallback(DetectionMsg::UniquePtr detection_msg){
    // Cal Compute Time
    tick = CLK.now();

    // 检查相机信息是否已接收
    if (!camera_info_received_) {
        RCLCPP_WARN(get_logger(), "Camera info not received yet, skipping detection");
        return;
    }
    
    // 获取时间戳
    double timestamp = rclcpp::Time(detection_msg->header.stamp).seconds();
    manager_.recv_detection.time_stamp = timestamp;
    
    // 从机器人消息获取当前模式
    if (!manager_.robot.vision_mode.empty()) {
        params_.mode = string2vision_mode(manager_.robot.vision_mode);
        // RCLCPP_INFO(get_logger(), "Received vision mode: %s", manager_.robot.vision_mode.c_str());
    } else {
        params_.mode = VisionMode::AUTO_AIM;
        RCLCPP_WARN(get_logger(), "No vision mode in robot message, using default AUTO_AIM");
    }
    // 从机器人消息获取右键状态
    params_.right_press = manager_.robot.right_press;
    // RCLCPP_INFO(get_logger(), "Right press state: %d", params_.right_press);
    
    // 从机器人消息获取机器人ID
    if (manager_.robot.robot_id > 0) {
        params_.robot_id = static_cast<RobotIdDji>(manager_.robot.robot_id);
        // RCLCPP_INFO(get_logger(), "Robot ID: %d", manager_.robot.robot_id);
    } else {
        RCLCPP_WARN(get_logger(), "Invalid robot ID: %d", manager_.robot.robot_id);
    }
    // 更新自身ID
    manager_.self_id = RmcvId(static_cast<RobotIdDji>(manager_.robot.robot_id));
    params_.rmcv_id = manager_.self_id;
    
    // 检查模式是否有效
    if (params_.mode != VisionMode::OUTPOST_AIM && params_.mode != VisionMode::AUTO_AIM) {
        RCLCPP_INFO(get_logger(), "Not in OUTPOST_AIM or AUTO_AIM mode, skipping processing");
        return;
    }
    // 从机器人消息获取弹速并更新弹道 是否重复了？
    bool is_big_bullet = false;
    if (manager_.robot.bullet_velocity > 8.0) {
        is_big_bullet = params_.rmcv_id.robot_id == RobotId::ROBOT_HERO;
        // RCLCPP_INFO(get_logger(), "Bullet velocity updated: %.2f m/s", manager_.robot.bullet_velocity);
    } else {
        manager_.robot.bullet_velocity = is_big_bullet ? 12.0 : 22.0;
        // RCLCPP_WARN(get_logger(), "Invalid bullet velocity: %.2f", manager_.robot.bullet_velocity);
    }
    bac->refresh_velocity(is_big_bullet, manager_.robot.bullet_velocity);

    // 从机器人消息获取相机切换状态
    if (manager_.robot.switch_cam) {
        RCLCPP_INFO(get_logger(), "Camera switch requested");
        // 这里可以添加相机切换逻辑
    }
    
    // 从机器人消息获取自动射击率
    if (manager_.robot.autoshoot_rate > 0) {
        // RCLCPP_INFO(get_logger(), "Auto shoot rate: %d", manager_.robot.autoshoot_rate);
    }

    off_cmd.flag = 0;
    // 清空之前的检测结果
    std::vector<TrackedArmor> tracked_armors;
    
    // 转换检测消息为内部结构，并进行坐标变换
    for (const auto& detection : detection_msg->detections) {
        if (detection.results.empty()) {
            continue;
        }
        
        TrackedArmor armor;
        armor.fromDetectionMsg(detection, timestamp);
        
        // 获取camera系下的位姿
        const auto& pose_camera = detection.results[0].pose.pose;
        
        // 将相机坐标系下的位姿转换到odom坐标系
        Eigen::Vector3d pos_camera(
            pose_camera.position.x,
            pose_camera.position.y,
            pose_camera.position.z
        );
        
        // 注意：当前上游把 orientation.xyz 填成了 camera 系下的 RPY（rad），且 w 恒为 1.0
        // 这里需要先把 RPY 还原成四元数，再交给 tf2 做坐标系转换。
        const double ow = pose_camera.orientation.w;
        const double ox = pose_camera.orientation.x;
        const double oy = pose_camera.orientation.y;
        const double oz = pose_camera.orientation.z;

        Eigen::Quaterniond ori_camera;
        // 兼容 imagepipe 发来的 w=0 和其他节点可能的 w=1 的 RPY 编码情况
        if (std::abs(ow - 1.0) < 1e-3 || std::abs(ow) < 1e-3) {
            // imagepipe 发送的 orientation 顺序为 [roll, pitch, yaw]
            // ox(roll) 对应绕 Z 轴旋转 (光轴)
            // oy(pitch) 对应绕 X 轴旋转 (水平)
            // oz(yaw) 对应绕 Y 轴旋转 (垂直)
            // tf2::setRPY(r, p, y) 对应绕固定轴 X, Y, Z 的旋转
            
            // 理论推导：
            // ImagePipe PnP 解算出的 Z 轴可能指向 Cam Z 反方向（指向相机），即 Yaw=180。
            // 转换到 Odom 系下，Normal 指向 Vehicle X (前)，即背离车 (Yaw=0)。
            // 但物理上装甲板面向车，Normal 应指向 Vehicle -X (后)，即 Yaw=180。
            // 因此需要补偿 M_PI 来翻转 Normal 方向。
            const double rot_x = oy; // pitch
            const double rot_y = oz + M_PI; // yaw + 180 deg
            const double rot_z = ox; // roll
            
            tf2::Quaternion q;
            q.setRPY(rot_x, rot_y, rot_z);
            q.normalize();
            ori_camera = Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z());
        } else {
            ori_camera = Eigen::Quaterniond(ow, ox, oy, oz);
            ori_camera.normalize();
        }
        
        // 转换到odom系
        Eigen::Vector3d pos_odom;
        Eigen::Quaterniond ori_odom;
        
        if (convertCameraToOdom(pos_camera, ori_camera, detection_msg->header.stamp, pos_odom, ori_odom)) {
            armor.setOdomPose(pos_odom, ori_odom);
            
            RCLCPP_INFO(get_logger(), "Armor %d: camera (%.3f, %.3f, %.3f) -> odom (%.3f, %.3f, %.3f)",
                        armor.id,
                        pos_camera.x(), pos_camera.y(), pos_camera.z(),
                        pos_odom.x(), pos_odom.y(), pos_odom.z());
            
            tracked_armors.push_back(armor);
        } else {
            RCLCPP_WARN(get_logger(), "Failed to transform armor %d from camera to odom", armor.id);
        }
    }
    
    // 允许短时间无装甲板：仍然更新管理器以便处理Absent/超时/状态机
    if (tracked_armors.empty()) {
        RCLCPP_WARN(get_logger(), "No armor successfully transformed to odom frame");
    }

    // 更新管理器
    manager_.recv_detection.armors = tracked_armors;
    manager_.recv_detection.mode = params_.mode;
    manager_.updateArmors(tracked_armors, timestamp);
    manager_.updateOutpost();

    // 计算延迟（简化处理）
    tock = CLK.now();
    prev_latency = (tock - tick).seconds();

    ControlMsg now_cmd = get_command();

    // 设置消息头
    now_cmd.header.frame_id = "robot_cmd: " + std::to_string(manager_.robot.robot_id);
    now_cmd.header.stamp = detection_msg->header.stamp;
    control_pub->publish(now_cmd);

    // 可视化
    if (params_.enable_imshow && !current_image_.empty()) {
        cv::Mat vis_img = manager_.getVisualizationImage();
        if (!vis_img.empty()) {
            // 绘制光心
            if (manager_.camera_k_.size() >= 6) {
                cv::circle(vis_img, cv::Point(manager_.camera_k_[2], manager_.camera_k_[5]), 3, cv::Scalar(255, 0, 255), 2);
            }

            cv::imshow("Outpost Predictor", vis_img);
            cv::waitKey(1);
        } else {
            RCLCPP_WARN(get_logger(), "Image empty!!!");
        }
    }

    publishMarkers(timestamp);

}

void OutpostNode::publishMarkers(double timestamp) {
    if (!manager_.outpost.outpost_kf_init) return;

    visualization_msgs::msg::MarkerArray marker_array;
    
    // 公共头信息
    auto header = std_msgs::msg::Header();
    header.frame_id = params_.target_frame; // "odom"
    header.stamp = this->now();

    const auto& op = manager_.outpost;

    // 1. 中心位置 (Sphere)
    visualization_msgs::msg::Marker center_marker;
    center_marker.header = header;
    center_marker.ns = "outpost_center";
    center_marker.id = 0;
    center_marker.type = visualization_msgs::msg::Marker::SPHERE;
    center_marker.action = visualization_msgs::msg::Marker::ADD;
    center_marker.pose.position.x = op.now_position_.center_[0];
    center_marker.pose.position.y = op.now_position_.center_[1];
    center_marker.pose.position.z = op.now_position_.center_[2];
    center_marker.scale.x = 0.2;
    center_marker.scale.y = 0.2;
    center_marker.scale.z = 0.2;
    center_marker.color.a = 1.0;
    center_marker.color.r = 1.0; 
    center_marker.color.g = 1.0;
    center_marker.color.b = 0.0; // Yellow
    marker_array.markers.push_back(center_marker);

    // 2. 中心方向/角速度 (Arrow)
    visualization_msgs::msg::Marker center_arrow;
    center_arrow.header = header;
    center_arrow.ns = "outpost_center_yaw";
    center_arrow.id = 1;
    center_arrow.type = visualization_msgs::msg::Marker::ARROW;
    center_arrow.action = visualization_msgs::msg::Marker::ADD;
    center_arrow.pose.position = center_marker.pose.position;
    // 使用中心 Yaw
    tf2::Quaternion q;
    q.setRPY(0, 0, op.op_ckf.state_.yaw); 
    center_arrow.pose.orientation = tf2::toMsg(q);
    center_arrow.scale.x = 0.5; // Length
    center_arrow.scale.y = 0.05; 
    center_arrow.scale.z = 0.05;
    center_arrow.color.a = 1.0;
    center_arrow.color.r = 1.0; 
    center_arrow.color.g = 0.0;
    center_arrow.color.b = 1.0; // Magenta
    marker_array.markers.push_back(center_arrow);
    
    // 3. 估计的三块装甲板 (Sphere)
    for(int i=0; i<3; ++i) {
        visualization_msgs::msg::Marker armor_marker;
        armor_marker.header = header;
        armor_marker.ns = "estimated_armors";
        armor_marker.id = 10 + i;
        armor_marker.type = visualization_msgs::msg::Marker::SPHERE;
        armor_marker.action = visualization_msgs::msg::Marker::ADD;
        
        if (i < (int)op.now_position_.armors_xyz_.size()) {
            armor_marker.pose.position.x = op.now_position_.armors_xyz_[i][0];
            armor_marker.pose.position.y = op.now_position_.armors_xyz_[i][1];
            armor_marker.pose.position.z = op.now_position_.armors_xyz_[i][2];
            armor_marker.scale.x = 0.15;
            armor_marker.scale.y = 0.15;
            armor_marker.scale.z = 0.15;
            armor_marker.color.a = 0.8;
            armor_marker.color.r = 0.0;
            armor_marker.color.g = 1.0;
            armor_marker.color.b = 0.0; // Green
            marker_array.markers.push_back(armor_marker);
        }
    }

    // 4. 用于更新的观测装甲板 Yaw (Arrow)
    // 遍历 armors，找到 alive_ts_ 接近 current timestamp 的
    for(const auto& armor : op.armors) {
        if (std::abs(armor.alive_ts_ - timestamp) < 1e-4 && armor.status_ == Alive) {
            visualization_msgs::msg::Marker obs_arrow;
            obs_arrow.header = header;
            obs_arrow.ns = "measured_armor_yaw";
            // obs_arrow.id = 20 + armor.armor_id_; // Use armor_id (need to check if initialized, usually 0/1/2?)
            // wait, armor_id_ might not be 0-2 if not phase assigned?
            // In Outpost::update, phase is passed. OutpostArmor has phase_in_outpost_.
            // Let's use phase_in_outpost_ for ID if available, else standard iterator
            int marker_id = 20 + (armor.phase_in_outpost_ >= 0 ? armor.phase_in_outpost_ : 100);
            obs_arrow.id = marker_id;

            obs_arrow.type = visualization_msgs::msg::Marker::ARROW;
            obs_arrow.action = visualization_msgs::msg::Marker::ADD;
            
            // 起点是观测到的位置
            Eigen::Vector3d pos_xyz = armor.getPositionXyz(); 
            obs_arrow.pose.position.x = pos_xyz[0];
            obs_arrow.pose.position.y = pos_xyz[1];
            obs_arrow.pose.position.z = pos_xyz[2];

            // 这里的 Yaw 是 pc_result_.pose.yaw (Odom系下的观测Yaw)
            double obs_yaw = armor.pc_result_.pose.yaw;

            tf2::Quaternion q_obs;
            q_obs.setRPY(0, 0, obs_yaw);
            obs_arrow.pose.orientation = tf2::toMsg(q_obs);

            obs_arrow.scale.x = 0.4;
            obs_arrow.scale.y = 0.05;
            obs_arrow.scale.z = 0.05;
            obs_arrow.color.a = 1.0;
            obs_arrow.color.r = 1.0; // Red for observation
            obs_arrow.color.g = 0.0;
            obs_arrow.color.b = 0.0;
            
            marker_array.markers.push_back(obs_arrow);
        }
    }

    marker_pub_->publish(marker_array);
}

void OutpostNode::robotCallback(RmRobotMsg::SharedPtr robot_msg){
    manager_.robot = *robot_msg;
    // 更新自身ID
    manager_.self_id = RmcvId(static_cast<RobotIdDji>(robot_msg->robot_id));
    params_.rmcv_id = manager_.self_id;
    last_mode = string2vision_mode(robot_msg->vision_mode); 
    if (last_mode != VisionMode::OUTPOST_AIM && last_mode != VisionMode::AUTO_AIM && manager_.outpost.alive_ts > 0){
        RCLCPP_INFO(this->get_logger(), "Reset-Outpost");
        manager_.outpost = Outpost();
        manager_.outpost.alive_ts = -1;
        manager_.phase_initialized_ = false;
        manager_.last_active_tracker_id_ = -1;
        manager_.last_active_phase_ = -1;
        manager_.last_active_yaw_ = 0.0;
        manager_.last_active_ts_ = -1.0;
    }
    imu = manager_.robot.imu;

    // 更新弹速（如果机器人消息中有） 严查
    bool is_big_bullet = false;
    if (robot_msg->bullet_velocity > 8.) {
        is_big_bullet = params_.rmcv_id.robot_id == RobotId::ROBOT_HERO;
    }else {
        double bullet_velocity = is_big_bullet ? 12.0 : 22.9;
    }
    bac->refresh_velocity(is_big_bullet, robot_msg->bullet_velocity);
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(OutpostNode)