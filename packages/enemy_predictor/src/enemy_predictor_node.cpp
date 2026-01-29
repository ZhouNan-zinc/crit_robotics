#include "enemy_predictor/enemy_predictor_node.h"
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

EnemyPredictorNode::EnemyPredictorNode(const rclcpp::NodeOptions& options) 
    : Node("enemy_predictor", rclcpp::NodeOptions(options)),
      bac(create_ballistic_params()){
    RCLCPP_INFO(this->get_logger(),"AutoaimNode init"); 
    tf2_buffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf2_listener = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer);

    detector_sub=create_subscription<vision_msgs::msg::Detection2DArray>(
        "/vision/tracked", rclcpp::QoS(10), std::bind(&EnemyPredictorNode::detection_callback, this, std::placeholders::_1));
    //取决于电控给我发消息的频率, 要适配 HighFrequencyCallback()
    imu_sub=create_subscription<rm_msgs::msg::RmRobot>(
        "rm_robot", rclcpp::SensorDataQoS(), std::bind(&EnemyPredictorNode::robot_callback, this, std::placeholders::_1));

    camera_sub = image_transport::create_camera_subscription(
        this,
        "/hikcam/image_raw",
        [this](auto image_msg, auto camera_info_msg) {
            this->camera_callback(image_msg, camera_info_msg);
        },
        "raw"
    );
    control_pub = create_publisher<rm_msgs::msg::Control>("enemy_predictor", rclcpp::SensorDataQoS());
    //high_freq_timer_ = this->create_wall_timer(std::chrono::milliseconds(2),  // 2ms = 500Hz
    //                                           std::bind(&EnemyPredictorNode::HighFrequencyCallback, this));

     enemy_markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "enemy_center_markers", 10);
    
    this->declare_parameter<std::vector<double>>("armor_ekf.P",std::vector<double>{0.025, 0.049, 0.001, 0.01, 0.01, 0.01});
    auto p_diag = this->get_parameter("armor_ekf.P").as_double_array();
    ArmorXYYAWEKF::config_.config_P = Eigen::Map<Eigen::VectorXd>(p_diag.data(),6);
    
    this->declare_parameter<std::vector<double>>("armor_ekf.Q",std::vector<double>{0.025, 0.049, 0.001, 0.01, 0.01, 0.01});
    auto q_diag = this->get_parameter("armor_ekf.Q").as_double_array();
    ArmorXYYAWEKF::config_.config_Q = Eigen::Map<Eigen::VectorXd>(q_diag.data(),6);
    
    this->declare_parameter<std::vector<double>>("armor_ekf.R",std::vector<double>{0.00025, 0.00016, 0.0025});
    auto r_diag = this->get_parameter("armor_ekf.R").as_double_array();
    ArmorXYYAWEKF::config_.config_R = Eigen::Map<Eigen::VectorXd>(r_diag.data(),3);

    this->declare_parameter<std::vector<double>>("armor_z_ekf.Q", {0.01, 0.01});
    auto q_z = this->get_parameter("armor_z_ekf.Q").as_double_array();
    ZEKF::config_.config_Q = Eigen::Map<Eigen::VectorXd>(q_z.data(),2);
    
    // 改为单个double
    this->declare_parameter<double>("armor_z_ekf.R", 0.01);
    auto r_z = this->get_parameter("armor_z_ekf.R").as_double();
    ZEKF::config_.config_R = Eigen::VectorXd(1);
    ZEKF::config_.config_R(0) = r_z;

    this->declare_parameter<std::vector<double>>("enemy_ckf.Pe",
                                                std::vector<double>{
                                                    0.25,   0.10,   0.03,   0.01,   0.02,   0.00,
                                                    0.10,   1.00,   0.01,   0.02,   0.00,   0.00,
                                                    0.03,   0.01,   0.25,   0.10,   0.02,   0.00,
                                                    0.01,   0.02,   0.10,   1.00,   0.00,   0.00,
                                                    0.02,   0.00,   0.02,   0.00,   0.04,   0.05,
                                                    0.00,   0.00,   0.00,   0.00,   0.05,   0.25
                                                });
    auto pe = this->get_parameter("enemy_ckf.Pe").as_double_array();
    EnemyCKF::config_.config_Pe = Eigen::Map<Eigen::MatrixXd>(pe.data(),6,6);

    this->declare_parameter<double>("enemy_ckf.Q2_X", 0.01);
    EnemyCKF::config_.Q2_X = this->get_parameter("enemy_ckf.Q2_X").as_double();

    this->declare_parameter<double>("enemy_ckf.Q2_Y", 0.01);
    EnemyCKF::config_.Q2_Y = this->get_parameter("enemy_ckf.Q2_Y").as_double();

    this->declare_parameter<double>("enemy_ckf.Q2_YAW", 0.01);
    EnemyCKF::config_.Q2_YAW = this->get_parameter("enemy_ckf.Q2_YAW").as_double();

    this->declare_parameter<double>("enemy_ckf.R_XYZ", 0.01);
    EnemyCKF::config_.R_XYZ = this->get_parameter("enemy_ckf.R_XYZ").as_double();

    this->declare_parameter<double>("enemy_ckf.R_YAW", 0.01);
    EnemyCKF::config_.R_YAW = this->get_parameter("enemy_ckf.R_YAW").as_double();
    
    this->declare_parameter<double>("interframe_dis_thresh", 0.04617);
    interframe_dis_thresh = this->get_parameter("interframe_dis_thresh").as_double();

    this->declare_parameter<double>("robot_2armor_dis_thresh", 0.04617);
    robot_2armor_dis_thresh = this->get_parameter("robot_2armor_dis_thresh").as_double();

    this->declare_parameter<double>("high_spd_rotate_thresh", 0.30);
    cmd.high_spd_rotate_thresh = this->get_parameter("high_spd_rotate_thresh").as_double();

    this->declare_parameter<double>("yaw_thresh", 0.30);
    cmd.yaw_thresh = this->get_parameter("yaw_thresh").as_double();

    this->declare_parameter<double>("response_delay", 0.30);
    params_.response_delay = this->get_parameter("response_delay").as_double();

    this->declare_parameter<double>("shoot_delay", 0.30);
    params_.shoot_delay = this->get_parameter("shoot_delay").as_double();
    
    std::string shared_dir = ament_index_cpp::get_package_share_directory("enemy_predictor");
    bac.refresh_velocity(false, 30.0);

    RCLCPP_INFO(get_logger(), "shared_dir: %s", shared_dir.c_str());
    for(int i = 0; i < MAX_ENEMIES; i++){
        enemies_[i].class_id = i+1;
    }
}

void EnemyPredictorNode::camera_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg, 
                                        const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info_msg){
    if (!image_msg) {
        RCLCPP_ERROR(get_logger(), "Received null image message!");
        return;
    }
     // 检查图像数据是否为空
    if (image_msg->data.empty()) {
        RCLCPP_WARN(get_logger(), "Received empty image data!");
        return;
    }
    
    // 检查 CameraInfo 是否有效
    if (!camera_info_msg) {
        RCLCPP_ERROR(get_logger(), "Received null camera info message!");
        return;
    }
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_msg, 
                                                    sensor_msgs::image_encodings::BGR8);
    visualize_.armor_img = cv_ptr->image.clone();
    if(not visualize_.camera_matrix.empty()){
        return;
    }
    visualize_.camera_matrix = (cv::Mat_<double>(3, 3) <<
                               camera_info_msg->k[0], camera_info_msg->k[1], camera_info_msg->k[2],
                               camera_info_msg->k[3], camera_info_msg->k[4], camera_info_msg->k[5],
                               camera_info_msg->k[6], camera_info_msg->k[7], camera_info_msg->k[8]);
    std::vector<double> d_vec(camera_info_msg->d.begin(), camera_info_msg->d.end());
    visualize_.dist_coeffs = cv::Mat(d_vec).reshape(1, d_vec.size()).clone();  // Nx1 矩阵
}

Ballistic::BallisticParams EnemyPredictorNode::create_ballistic_params() {
    Ballistic::BallisticParams params;
    
    RCLCPP_INFO(this->get_logger(), "Creating ballistic parameters...");
    
    try {
        // 声明所有Ballistic参数
        this->declare_parameter("ballistic.noise_sigma", 0.00);
        this->declare_parameter("ballistic.params_found", true);
        this->declare_parameter("ballistic.ballistic_refresh", true);
        
        this->declare_parameter("ballistic.v9", 8.7);
        this->declare_parameter("ballistic.v15", 14.6);
        this->declare_parameter("ballistic.v16", 15.6);
        this->declare_parameter("ballistic.v18", 16.4);
        this->declare_parameter("ballistic.v30", 22.9);
        
        this->declare_parameter("ballistic.small_k", 0.002);
        this->declare_parameter("ballistic.big_k", 0.0008);
        this->declare_parameter("ballistic.l", 0.0);
        
        this->declare_parameter("ballistic.theta_l", -45.0);
        this->declare_parameter("ballistic.theta_r", 45.0);
        this->declare_parameter("ballistic.theta_d", 0.01);
        
        this->declare_parameter("ballistic.x_l", 0.0);
        this->declare_parameter("ballistic.x_r", 10.0);
        this->declare_parameter("ballistic.x_n", 1000);
        
        this->declare_parameter("ballistic.y_l", -5.0);
        this->declare_parameter("ballistic.y_r", 5.0);
        this->declare_parameter("ballistic.y_n", 1000);

        params.pitch2yaw_t = 
        this->declare_parameter("pitch2yaw_t", std::vector<double>{0.0, 0.0, 0.0});
        
        this->declare_parameter("ballistic.table_dir", "");
        
        // 获取参数值
        params.noise_sigma = this->get_parameter("ballistic.noise_sigma").as_double();
        params.params_found = this->get_parameter("ballistic.params_found").as_bool();
        params.ballistic_refresh = this->get_parameter("ballistic.ballistic_refresh").as_bool();
        
        params.v9 = this->get_parameter("ballistic.v9").as_double();
        params.v15 = this->get_parameter("ballistic.v15").as_double();
        params.v16 = this->get_parameter("ballistic.v16").as_double();
        params.v18 = this->get_parameter("ballistic.v18").as_double();
        params.v30 = this->get_parameter("ballistic.v30").as_double();
        
        params.small_k = this->get_parameter("ballistic.small_k").as_double();
        params.big_k = this->get_parameter("ballistic.big_k").as_double();
        params.l = this->get_parameter("ballistic.l").as_double();
        
        // 角度参数（度转弧度）
        params.theta_l = this->get_parameter("ballistic.theta_l").as_double();
        params.theta_r = this->get_parameter("ballistic.theta_r").as_double();
        params.theta_d = this->get_parameter("ballistic.theta_d").as_double();
        
        params.x_l = this->get_parameter("ballistic.x_l").as_double();
        params.x_r = this->get_parameter("ballistic.x_r").as_double();
        params.x_n = this->get_parameter("ballistic.x_n").as_int();
        
        params.y_l = this->get_parameter("ballistic.y_l").as_double();
        params.y_r = this->get_parameter("ballistic.y_r").as_double();
        params.y_n = this->get_parameter("ballistic.y_n").as_int();
        
        params.table_dir = this->get_parameter("ballistic.table_dir").as_string();
        
        
        // 设置默认值（如果没有从参数获取）
        if (params.table_dir.empty()) {
            params.table_dir = ament_index_cpp::get_package_share_directory("enemy_predictor");  // 空字符串表示不使用表格
        }
        
        // 设置默认的偏移量（二维数组）
        params.stored_cam2gun_offset = std::vector<std::vector<double>>(5, 
            std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0});
        
        // 设置默认的偏移量
        params.stored_yaw_offset = std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0};
        params.stored_pitch_offset = std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0};
     
        params.yaw2gun_offset = sqrt(params.pitch2yaw_t[0] * params.pitch2yaw_t[0] + params.pitch2yaw_t[1] * params.pitch2yaw_t[1]);    
        RCLCPP_INFO(this->get_logger(), "params.yaw2gun_offset = %lf",params.yaw2gun_offset);
        RCLCPP_INFO(this->get_logger(), 
                   "Ballistic params created: v15=%.1f m/s, theta=[%.1f°, %.1f°]",
                   params.v15, 
                   params.theta_l ,
                   params.theta_r);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), 
                    "Error creating ballistic params: %s. Using defaults.", e.what());
    }
    
    return params;
}

void EnemyPredictorNode::detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr detection_msg){
    if(visualize_.dist_coeffs.empty() || visualize_.armor_img.empty()){
        RCLCPP_WARN(get_logger(), "Empty Image Messages");
        return;
    }

    std::vector<Detection, Eigen::aligned_allocator<Detection>> current_detections_{};
    std::vector<int>active_enemies_idx{};
    std::vector<int>active_armor_idx{};
    std::vector<cv::Point3f> object_points;
    time_det = this -> now();
    double timestamp = time_det.seconds();
    const auto& detections = detection_msg->detections;
    time_image = detection_msg-> header.stamp;
    
    cmd.cmd_mode = -1;
    for(Enemy& enemy : enemies_){
        enemy.is_active = false;
    }

    for (const auto& detection : detections) {
        bool valid_det = true;
        if(detection.bbox.size_x / detection.bbox.size_y < 0.5 || detection.bbox.size_x / detection.bbox.size_y > 4){
            continue;
        }
        Detection det{};
        det.armor_idx = std::stoi(detection.id);
        
        const auto& results_ = detection.results;
       
        for(const auto& res : results_){
            const auto& pos = res.pose;
          
            det.position = Eigen::Vector3d(pos.pose.position.x, pos.pose.position.y, pos.pose.position.z) ;
            
            det.orientation = Eigen::Vector3d(pos.pose.orientation.x, pos.pose.orientation.y, pos.pose.orientation.z);
            
            if(abs(pos.pose.orientation.z) > 1.05){
                valid_det = false;
            }
            double current_yaw = getCurrentYaw(time_image);
           
            det.yaw = pos.pose.orientation.z - current_yaw;
           
            det.armor_class_id = std::stoi(res.hypothesis.class_id);
            
            if(det.armor_class_id % 10 == 1){
                object_points = large_object_points;
            }else{
                object_points = small_object_points;
            }
           //updateArmorDetection(object_points, det, time_image);
        }
        if(valid_det){
            updateArmorDetection(object_points, det, time_image);
            current_detections_.emplace_back(det);
        }

    }
    if(current_detections_.empty()){
        return;
    }
    //if (params_.mode != VisionMode::AUTO_AIM) {
     //    return;
     //}
    ToupdateArmors(current_detections_, timestamp, active_armor_idx);
   
    EnemyManage(timestamp, time_image, active_enemies_idx, active_armor_idx);

    rm_msgs::msg::Control control_msg{};
   
    if (cmd.cmd_mode == 0 || cmd.cmd_mode == 1){

        control_msg.pitch = cmd.cmd_pitch;
        //RCLCPP_INFO(get_logger(), "cmd.cmd_pitch =%d", cmd.cmd_pitch);
        control_msg.flag = 1;
        control_msg.vision_follow_id = enemies_[cmd.target_enemy_idx].class_id;
        if(cmd.cmd_mode == 0){
           control_msg.rate = 15; // adjust it later!!!
           control_msg.one_shot_num = 3;
        }
        else if(cmd.cmd_mode == 1){
           control_msg.rate = 18;
           control_msg.one_shot_num = 3;
        }
        //bool success = yaw_planner.setTargetYaw(cmd.cmd_yaw, imu_.current_yaw);
        bool success = false;    //先把自瞄调通
        if(!success){
            control_msg.yaw = cmd.cmd_yaw;
            publish_mode_ = PublishMode::FRAME_RATE_MODE;
            RCLCPP_INFO(get_logger(), "Publish Control Msgs!");
        }else{
            publish_mode_ = PublishMode::HIGH_FREQ_MODE;
        }
        control_pub-> publish(std::move(control_msg));
    }
    //else{
    //    control_msg.flag = 0;
    //    control_msg.yaw = 0.0;
    //    control_msg.pitch = 0.0;
    //    control_msg.rate = 0;
    //    control_msg.one_shot_num = 0;
//
    //    RCLCPP_INFO(get_logger(), "No valid Control Msgs");
    //}
    //control_pub-> publish(std::move(control_msg));

    if (!enemy_markers_.markers.empty()) {
        enemy_markers_pub_->publish(enemy_markers_);
    }
    cv::imshow("Armor_img", visualize_.armor_img);
    cv::waitKey(10);
}
void EnemyPredictorNode::robot_callback(const rm_msgs::msg::RmRobot::SharedPtr robot_msg){
    robot = *robot_msg;
    //params_.mode = robot.vision_mode;
    //params_.cam_mode = frame_info.cam_mode;
    params_.right_press = robot_msg->right_press;
    params_.right_press = false; // For Debug
    ImuData data;
    data.timestamp = this->now();
    data.current_yaw = robot_msg->imu.yaw;
    yaw_now = robot_msg->imu.yaw;
    imu_buffer_.push_back(data);

    if (imu_buffer_.size() > 500) {
        cleanOldImuData();
    }
    // RCLCPP_INFO_STREAM(get_logger(), "imu.yaw =" << imu_.current_yaw);
    //RCLCPP_INFO(this->get_logger(),  "IMU - Roll: %.3f°, Pitch: %.3f°, Yaw: %.3f°",robot.imu.roll, robot.imu.pitch, robot.imu.yaw);
}
/*void EnemyPredictorNode::HighFrequencyCallback() {
    if (cmd.aim_center.x() == -999 || publish_mode_ != PublishMode::HIGH_FREQ_MODE) {
        return;
    }
    
    auto next_cycle_time = std::chrono::steady_clock::now();
    
    while(true) {

        next_cycle_time = next_cycle_time + 
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(yaw_planner.config_.motor_control_period));
        
        double current_yaw = robot.imu.yaw;
        auto current_time = std::chrono::steady_clock::now();
        
        auto output = yaw_planner.getMotorTarget(current_yaw, current_time);
    
        if (control_msg) {
            control_msg -> yaw = output.target_position_abs;
            control_pub -> publish(*control_msg);
        }
        
        if(!output.is_active){
            publish_mode_ = PublishMode::FRAME_RATE_MODE;
            break;
        }
        
        std::this_thread::sleep_until(next_cycle_time);
    }
}*/
