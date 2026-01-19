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

class EnemyPredictorNode : public rclcpp::Node {

public:    
    enum class PublishMode{
        FRAME_RATE_MODE,    // 帧率模式：每次detection_callback直接发送
        HIGH_FREQ_MODE      // 高频模式：启动高频回调发送插值点
    }publish_mode_;

    struct Imu{
        double current_yaw = 0.0;
    }imu_;
    
    struct Detection {
        Eigen::Vector3d position;
        Eigen::Vector3d orientation; 
        int armor_class_id;         
        //double confidence;      
        int armor_idx;
        //cv::Rect rect;          
        double yaw = 0.0;    
        double area_2d = 0.0;      
        double dis_2d = 0.0;   
        double dis_to_heart = 0.0;
        Detection() = default;
        
        Detection(const Eigen::Vector3d& pos, int armor_class_id_, int armor_idx_, double y = 0)
            : position(pos), armor_class_id(armor_class_id_),armor_idx(armor_idx_),yaw(y) {}
    };

    
    // 装甲板跟踪器
    struct ArmorTracker {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        //bool is_valid = true;
        bool is_active = true;
        int tracker_idx;
        int armor_class_id;
  
        int missing_frames = 0;
        
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
        int type = -1;
        int enemy_idx = -1;
        int mode = -1;
        int best_armor = -1; // 最佳装甲板phase_id for ckf
        int best_armor_idx = -1;
        std::vector<int> armor_tracker_ids;
        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        Eigen::Vector3d center_pre = Eigen::Vector3d::Zero();
        double radius = 0.28;
        bool radius_cal = false;
        bool is_alive = false;
        bool is_valid = true;
        int missing_frame = 0;
        double yaw = 0; 
        double omega = 0;
        EnemyCKF enemy_ckf;
        double alive_ts = 0;
        double min_dis_2d = INFINITY;
        
        // 滤波器系统（模仿上一版本）
        double last_yaw = 0;
        int yaw_round = 0;
        
<<<<<<< HEAD
        Enemy() = default; 
=======
>>>>>>> e7739be (feat: ros predictor node)
        // 构造函数
        Enemy(int enemy_idx, int type, double init_radius = 0.28)
            : enemy_idx(enemy_idx), type(type){
            // 初始化所有相位的滤波器
            //armor_radius_filters.resize(MathFilter(100, "harmonic_mean"));
            //armor_z_filters.resize(MathFilter(3, "arithmetic_mean"));
            
            //for (int i = 0; i < 4; ++i) {
            //    armor_radius_filters[i].update(init_radius);
            //    armor_z_filters[i].update(-0.1);  // 默认高度
            //}
        }
        
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
        
        bool empty() const { return armor_tracker_ids.empty(); }
        
    };
    struct Command{
        rm_msgs::msg::RmRobot robot;
        double high_spd_rotate_thresh = 0.0;
        Eigen::Vector3d aim_center = Eigen::Vector3d(-999, -999, -999);
        double yaw_thresh = 0.0;
        double cmd_pitch = 0.0;
        double cmd_yaw = 0.0;
        int last_target_enemy_id = -1;
        int target_enemy_id = -1;
        bool right_press = false;
        int cmd_mode; //  0 -> 平动 , 1 -> 小陀螺
    };
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
    };    

    // 数据容器
    ArmorTracker armor_tracker;
    std::vector<ArmorTracker> armor_trackers_;
    std::vector<Enemy> enemies_;
    std::vector<Detection> current_detections_; 
    //std::unordered_map<int, int> tracker_id_to_index_;
    Command cmd;
    EnemyPredictorNodeParams params_;
    Ballistic::BallisticResult ball_res;
    Ballistic bac;
    Ballistic::BallisticParams create_ballistic_params();
    RmcvId self_id;
    std::vector<int>active_enemies_idx;
<<<<<<< HEAD
    std::vector<int>active_armor_idx;
=======
>>>>>>> e7739be (feat: ros predictor node)
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
        cv::Mat camera_rvec{};
        cv::Mat camera_tvec{};
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
                              Detection& det);

    void ToupdateArmors(const std::vector<Detection>& detections,
                     double timestamp);
    
    Eigen::Isometry3d getTrans(const std::string& source_frame, 
                               const std::string& target_frame);
    
private:
    // 敌人分配和更新
    int assignToEnemy(ArmorTracker& tracker, double timestamp);
    void EnemyManage(double timestamp, Command& cmd, EnemyPredictorNodeParams& params_);
    void updateSingleEnemy(Enemy& enemy, double timestamp);
    void calculateEnemyCenterAndRadius(Enemy& enemy, double timestamp);
    Eigen::Vector3d FilterManage(Enemy &enemy, double dt, ArmorTracker& tracker);
    // 相位处理
    //void updateArmorPhase(Enemy& enemy, ArmorTracker& tracker, double timestamp);
    void findBestPhaseForEnemy(Enemy& enemy, ArmorTracker& tracker);
    int estimatePhaseFromPosition(const Enemy& enemy, const ArmorTracker& tracker);
    //bool check_left(const Eigen::Vector3d& pos1, const Eigen::Vector3d& pos2);
    
    // 半径计算
    double calculateRadiusFromTwoArmors(const ArmorTracker& armor1, 
                                       const ArmorTracker& armor2);
    //void updateRadiusFilters(Enemy& enemy, const ArmorTracker& armor1,
    //                        const ArmorTracker& armor2, double r1, double r2);
    std::pair<Ballistic::BallisticResult, Eigen::Vector3d> calc_ballistic_(double delay, Command& cmd, double timestamp, ArmorTracker& tracker, std::function<Eigen::Vector3d(ArmorTracker&, double)> _predict_func);
    void getCommand(Enemy& enemy, Command& cmd, double timestamp, EnemyPredictorNodeParams& params_);
    int ChooseMode(Enemy &enemy, double timestamp, Command& cmd);
    // tool
    void create_new_tracker(const Detection &detection, double timestamp);
    //ArmorTracker* getActiveArmorTrackerById(int tracker_id);

    void useGeometricCenterSimple(Enemy& enemy, 
                                const std::vector<ArmorTracker*>& active_armors);
    
    // 角度处理
    double normalize_angle(double angle);
    double angle_difference(double a, double b);
    
    // 可视化
    void visualizeAimCenter(const Eigen::Vector3d& armor_odom, const cv::Scalar& point_color = cv::Scalar(0, 0, 255));
    
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detector_sub;
    rclcpp::Subscription<rm_msgs::msg::RmRobot>::SharedPtr imu_sub;
    //std::shared_ptr<image_transport::CameraSubscriber> camera_sub;
    image_transport::CameraSubscriber camera_sub;
    rclcpp::Publisher<rm_msgs::msg::Control>::SharedPtr control_pub;

    rm_msgs::msg::Control::SharedPtr control_msg;
    std::mutex control_msg_mutex;

    std::shared_ptr<tf2_ros::Buffer> tf2_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener;
    rclcpp::TimerBase::SharedPtr high_freq_timer_;    // 高频定时器
    rm_msgs::msg::RmRobot robot;
    sensor_msgs::msg::Image::SharedPtr img_msg; 
    FrameInfo frame_info;
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