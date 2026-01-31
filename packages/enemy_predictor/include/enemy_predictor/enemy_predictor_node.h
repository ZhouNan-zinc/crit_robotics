#ifndef ENEMY_PREDICTOR_NODE_H  
#define ENEMY_PREDICTOR_NODE_H  

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <cmath>
#include "rclcpp/rclcpp.hpp"
#include "Eigen/Dense"
#include <opencv2/opencv.hpp>
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include <opencv2/opencv.hpp>
#include "enemy_predictor/armor_filter.h"
#include "enemy_predictor/enemy_filter.h"
#include "enemy_predictor/enemy_ballistic.h"
#include <enemy_trajectoryControl.h>
#include <sensor_msgs/msg/image.hpp>
#include "datatypes.h"
#include "image_transport/image_transport.hpp"
#include <vision_msgs/msg/detection2_d_array.hpp>
#include "rm_msgs/msg/rm_robot.hpp"
#include "rm_msgs/msg/control.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"

class EnemyPredictorNode : public rclcpp::Node {

public:    
    static constexpr size_t MAX_ENEMIES = 8;
    enum class PublishMode{
        FRAME_RATE_MODE,    // 帧率模式：每次detection_callback直接发送
        HIGH_FREQ_MODE      // 高频模式：启动高频回调发送插值点
    }publish_mode_;

    struct ImuData{
        double current_yaw = 0.0;
        rclcpp::Time timestamp;
    }imu_;
    
    struct Detection {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d position;
        Eigen::Vector3d orientation; 
        int armor_class_id;         
        //double confidence;      
        int armor_idx;
        //cv::Rect rect;          
        double yaw;    
        double area_2d;      
        double dis_2d;   
        double dis_to_heart;
    
        Detection() = default;
        
        Detection(const Eigen::Vector3d& pos, int armor_class_id_, int armor_idx_, double y = 0)
            : position(pos), armor_class_id(armor_class_id_),armor_idx(armor_idx_),yaw(y) {}
    };
    // 装甲板跟踪器
    struct ArmorTracker {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        bool is_active = true;
        int tracker_idx;
        int armor_class_id;
  
        int missing_frames = 0;
        int phase_id_cnt = 0;
        
        Eigen::Vector3d position;
        Eigen::Vector3d last_position;
        Eigen::Vector3d predicted_position;
        std::vector<Eigen::Vector3d> position_history;
        
        ArmorXYYAWEKF ekf;
        ZEKF zekf;
        
        double dis_2d = 0.0;
        double area_2d = 0.0;
        cv::Rect rect;
        
        double last_update_time = 0.0;
        
        int assigned_enemy_idx = -1;
        int phase_id = -1;
        double phase_conf = 0.0;
        
        // 朝向相关
        double yaw = 0.0;
        double last_yaw = 0.0;
        int yaw_round = 0;

        ArmorTracker() = default;
        
        ArmorTracker(int armor_idx, int armor_class_id,
                    const Eigen::Vector3d& init_pos, double timestamp,
                    double armor_yaw = 0.0, double area_2d = 0.0);
        
        void update(const Eigen::Vector3d& new_position, 
                   int armor_class_id, double timestamp,
                   double armor_yaw = 0.0);
        
        std::string get_string_id() const;
    };
    
    
    // 敌人结构体
    struct Enemy {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int class_id = -1;
        int mode = -1;
        int best_armor = -1; // 最佳装甲板phase_id for ckf
        int best_armor_idx = -1;
        std::vector<int> armor_tracker_ids;
        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        std::vector<double> radius{0.25, 0.25};
        bool radius_cal = false;
        bool is_active = false;
        bool is_valid = false;
        int missing_frame = 0;
        double yaw = 0.0; 

        EnemyCKF enemy_ckf;
        
        double last_yaw = 0.0;
        
        Enemy() = default; 

        void add_armor(int tracker_id) {
            if (std::find(armor_tracker_ids.begin(), 
                         armor_tracker_ids.end(), tracker_id) == armor_tracker_ids.end()) {
                armor_tracker_ids.push_back(tracker_id);
            }
        }
        
        void remove_armor(int tracker_id) {
            auto it = std::find(armor_tracker_ids.begin(),
                               armor_tracker_ids.end(), tracker_id);
            if (it != armor_tracker_ids.end()) {
                armor_tracker_ids.erase(it);
            }
        }
        void reset(){
           mode = -1; 
           center = Eigen::Vector3d::Zero();
           radius_cal = false;
           is_active = false;
           is_valid = false;
           missing_frame = 0;
           last_yaw = 0.0;
           yaw = 0.0; 
           enemy_ckf.initializeCKF();
           std::vector<double> radius{0.25, 0.25}; // 保留历史数据，越打越准可行吗？？
        }
    };
    struct Command{
        rm_msgs::msg::RmRobot robot;
        double high_spd_rotate_thresh = 0.0;
        Eigen::Vector3d aim_center = Eigen::Vector3d(-999, -999, -999);
        double yaw_thresh = 0.0;
        double cmd_pitch = 0.0;
        double cmd_yaw = 0.0;
        int last_target_enemy_idx = -1;
        int target_enemy_idx = -1;
        bool right_press = false;
        int cmd_mode; //  0 -> 平动 , 1 -> 小陀螺
    }cmd;
    struct EnemyPredictorNodeParams{
        std::string detection_name;
        std::string robot_name;
        std::string target_frame;
        std::string camera_name;
        bool enable_imshow;
        bool debug;
        VisionMode mode;
        bool right_press;  // 按下右键
        CameraMode cam_mode;
        RobotIdDji robot_id;
        RmcvId rmcv_id;
    
        double size_ratio_thresh;  // 切换整车滤波跟踪装甲板的面积阈值/切换选择目标的面积阈值
        
        // 火控参数
        double change_armor_time_thresh;
        double dis_yaw_thresh;
        double gimbal_error_dis_thresh;            // 自动发弹阈值，限制云台误差的球面意义距离
        double pitch_error_dis_thresh;             // 自动发弹阈值，目标上下小陀螺时pitch限制
        bool choose_enemy_without_autoaim_signal;  // 在没有收到右键信号的时候也选择目标（调试用)
        // 延迟参数
        double response_delay;  // 系统延迟(程序+通信+云台响应)
        double shoot_delay;     // 发弹延迟
    
        bool test_ballistic;
        bool follow_without_fire;
    
        double pitch_offset_high_hit_low;  //deg
        double pitch_offset_low_hit_high;  //deg
    }params_;    

    // 数据容器
    ArmorTracker armor_tracker;
    std::vector<ArmorTracker> armor_trackers_;
    std::array<Enemy, MAX_ENEMIES> enemies_;
    
    
    Ballistic::BallisticResult ball_res;
    Ballistic bac;
    Ballistic::BallisticParams create_ballistic_params();
    RmcvId self_id;
    double yaw_now = 0.0;
    
    //YawTrajectoryPlanner yaw_planner;
   
    double timestamp;
    // 参数
    double interframe_dis_thresh = 0.5;
    double robot_2armor_dis_thresh= 1.0;
    double min_radius_ = 0.12;
    double max_radius_ = 0.30;
    
    // 可视化相关
    struct VisualizeData {
        cv::Mat armor_img{};
        cv::Mat camera_matrix{};
        cv::Mat dist_coeffs{};
        cv::Mat camera_rvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat camera_tvec = cv::Mat::zeros(3, 1, CV_64F);;
        Eigen::Isometry3d camara_to_odom{};
        Eigen::Vector3d pos_camera{};
        cv::Point2f camera_heart{};
        std::string image_frame;

    } visualize_;
    std::vector<cv::Point3f> small_object_points = {
        {-0.0675, 0.0275, 0.},
        {-0.0675, -0.0275, 0.},
        {0.0675, -0.0275, 0.},
        {0.0675, 0.0275, 0.}
    };

    std::vector<cv::Point3f> large_object_points = {
        {-0.115, 0.029, 0.},
        {-0.115, -0.029, 0.},
        {0.115, -0.029, 0.},
        {0.115, 0.029, 0.}
    };    
public:
    explicit EnemyPredictorNode(const rclcpp::NodeOptions& options);
    

    void updateArmorDetection(std::vector<cv::Point3f> object_points,
                              Detection& det,
                              rclcpp::Time timestamp_det);

    void ToupdateArmors(const std::vector<Detection, Eigen::aligned_allocator<Detection>>& detections,
                     double timestamp, std::vector<int>& active_armor_idx);
    
    Eigen::Isometry3d getTrans(const std::string& source_frame, 
                               const std::string& target_frame,
                               rclcpp::Time timestamp_det);
    void EnemyManage(double timestamp, rclcpp::Time timestamp_det, std::vector<int>& active_enemies_idx, std::vector<int>& active_armor_idx);
private:
    // 敌人分配和更新
    void updateEnemy(Enemy& enemy, double timestamp, std::vector<int>& active_armor_idx);

    void updateSingleEnemy(Enemy& enemy, double timestamp);
    void calculateEnemyCenterAndRadius(Enemy& enemy, double timestamp, std::vector<ArmorTracker*> active_armors_this_enemy);
    Eigen::Vector3d FilterManage(Enemy &enemy, double dt, ArmorTracker& tracker, double timestamp);
    // 相位处理
    //void updateArmorPhase(Enemy& enemy, ArmorTracker& tracker, double timestamp);
    void findBestPhaseForEnemy(Enemy& enemy, ArmorTracker& tracker,  std::vector<ArmorTracker*> active_armors_this_enemy);
    int estimatePhaseFromPosition(const Enemy& enemy, const ArmorTracker& tracker);
    // 决策 + 弹道
    std::pair<Ballistic::BallisticResult, Eigen::Vector3d> calc_ballistic_(double delay, rclcpp::Time timestamp_det, ArmorTracker& tracker, double timestamp, std::function<Eigen::Vector3d(ArmorTracker&, double, double)> _predict_func);
    void getCommand(Enemy& enemy, double timestamp, rclcpp::Time timestamp_det, std::vector<int>& active_armor_idx);
    int ChooseMode(Enemy &enemy, double timestamp);
    // tool
    void create_new_tracker(const Detection &detection, double timestamp, std::vector<int>& active_armor_idx);

    void useGeometricCenterSimple(Enemy& enemy, 
                                const std::vector<ArmorTracker*>& active_armors);

    //std::vector<cv::Point2f> EnemyPredictorNode::Reprojection(Eigen::Vector3d odom_3d);
    //同步imu messages(yaw)
    double getCurrentYaw(const rclcpp::Time & target_time);
    void cleanOldImuData();

    // 角度处理
    double normalize_angle(double angle);
    double angle_difference(double a, double b);
    
    // 可视化
    void visualizeAimCenter(const Eigen::Vector3d& armor_odom, const cv::Scalar& point_color = cv::Scalar(0, 0, 255));
    
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detector_sub;
    rclcpp::Subscription<rm_msgs::msg::RmRobot>::SharedPtr imu_sub;

    image_transport::CameraSubscriber camera_sub;
    rclcpp::Publisher<rm_msgs::msg::Control>::SharedPtr control_pub;

    rm_msgs::msg::Control::SharedPtr control_msg;
    std::mutex control_msg_mutex;

    std::shared_ptr<tf2_ros::Buffer> tf2_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener;
    rclcpp::TimerBase::SharedPtr high_freq_timer_;    // 高频定时器
    rm_msgs::msg::RmRobot robot;
    sensor_msgs::msg::Image::SharedPtr img_msg; 
    rclcpp::Time time_det;
    rclcpp::Time time_image;

    std::deque<ImuData> imu_buffer_;
    // 保护缓冲区的互斥锁
    std::mutex buffer_mutex_;
    // 缓存的最大时长（秒），用于清理旧数据
    double buffer_duration_ = 10.0;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr enemy_markers_pub_;
    visualization_msgs::msg::MarkerArray enemy_markers_;
    //Armor armor;
    void detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr detection_msg);
    void robot_callback(const rm_msgs::msg::RmRobot::SharedPtr robot_msg);
    void camera_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                         const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg);
    //void HighFrequencyCallback();
};
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(EnemyPredictorNode)  // 注册插件


#endif // _ENEMY_PREDICTOR_NODE_H