#ifndef _OUTPOST_NODE_H
#define _OUTPOST_NODE_H

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <rm_msgs/msg/rm_robot.hpp>
#include <rm_msgs/msg/control.hpp>
#include <rm_msgs/msg/rm_imu.hpp>
#include <rm_msgs/msg/state.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "outpostaim/outpost_estimator.h"
#include "outpostaim/ballistic.h"
#include "outpostaim/datatypes.h"
#include "outpostaim/filter.h"
#include "outpostaim/math_utils.h"
#include <functional>

typedef rm_msgs::msg::Control ControlMsg;
typedef rm_msgs::msg::RmImu RmImuMsg;
typedef rm_msgs::msg::RmRobot RmRobotMsg;
typedef rm_msgs::msg::State StateMsg;
typedef vision_msgs::msg::Detection2DArray DetectionMsg;  // 添加类型别名

inline ControlMsg createControlMsg(float _pitch, float _yaw, uint8_t _flag, uint8_t _one_shot_num, uint8_t _rate
                                    /*, uint8_t _vision_follow_id, uint8_t _cam_mode*/){
    ControlMsg now;
    now.pitch = (float)_pitch;
    now.yaw = (float)_yaw;
    now.flag = _flag;
    now.one_shot_num = _one_shot_num;
    now.rate = _rate;
    //now.vision_follow_id = _vision_follow_id;
    //now.cam_mode = _cam_mode;
    return now;
}


inline VisionMode string2vision_mode(const std::string& mode_str){
    if (mode_str == "NO_AIM")
        return NO_AIM;
    else if (mode_str == "AUTO_AIM")
        return AUTO_AIM;
    else if (mode_str == "OUTPOST_AIM")
        return OUTPOST_AIM;
    else if (mode_str == "OUTPOST_LOB")
        return OUTPOST_LOB;
    else if (mode_str == "S_WM")
        return S_WM;
    else if (mode_str == "B_WM")
        return B_WM;
    else if (mode_str == "LOB")
        return LOB;
    else if (mode_str == "HALT")
        return HALT;
    else if (mode_str == "AUTOLOB")
        return AUTOLOB;
    else if (mode_str == "Unknown")
        return Unknown;
    else
        return Unknown;
}

// 简化装甲板信息结构，从新识别节点接收
struct TrackedArmor {
    int id;  // 跟踪ID
    int class_id;  // 装甲板类型
    float score;  // 置信度
    cv::Rect_<float> bbox;  // 2D边界框
    Eigen::Vector3d position_odom;    // odom坐标系下的3D位置
    Eigen::Quaterniond orientation_odom;    // odom坐标系下的姿态
    double timestamp;  // 时间戳
    
    // 从vision_msgs::msg::Detection2D转换而来
    void fromDetectionMsg(const vision_msgs::msg::Detection2D& detection, double stamp) {
        id = std::stoi(detection.id);
        if (!detection.results.empty()) {
            class_id = std::stoi(detection.results[0].hypothesis.class_id);
            score = detection.results[0].hypothesis.score;
        }
        
        // 2D边界框
        bbox = cv::Rect_<float>(
            detection.bbox.center.position.x - detection.bbox.size_x / 2,
            detection.bbox.center.position.y - detection.bbox.size_y / 2,
            detection.bbox.size_x,
            detection.bbox.size_y
        );
        
        timestamp = stamp;
    }
    
    // 设置odom系下的位姿（从外部转换后传入）
    void setOdomPose(const Eigen::Vector3d& pos, const Eigen::Quaterniond& ori) {
        position_odom = pos;
        orientation_odom = ori;
    }
};

struct OutpostDetectMsg {
    double time_stamp;
    std::vector<TrackedArmor> armors;  // 跟踪的装甲板列表
    VisionMode mode;
};

struct ArmorPoseResult {
    NgxyPose pose;
    double reproject_error;
    Eigen::Vector3d normal_vec;
    
    // 构造函数
    ArmorPoseResult() {}
    ArmorPoseResult(const Eigen::Vector3d& position, double yaw_val) {
        pose.x = position[0];
        pose.y = position[1];
        pose.z = position[2];
        
        // 从 position 计算 pitch 和 yaw
        Eigen::Vector3d pyd = xyz2pyd(position);
        pose.pitch = pyd[0];
        pose.yaw = yaw_val;  // 使用传入的 yaw
        // 其他成员保持默认
        reproject_error = 0.0;
        normal_vec = Eigen::Vector3d::Zero();
    }
};

struct OpNewSelectArmor{
    //整车建模的选择策略
    int armors_index_in_outpost; // 选择的装甲板在outpost_position.armors_xyz_中的索引
    Eigen::Vector3d xyz;
    double yaw_distance_predict;
};

struct OutpostNodeParams {
    // ROS话题参数
    std::string detection_topic = "vision/tracked";
    std::string robot_topic = "rm_robot";
    std::string control_topic = "control";
    std::string image_topic = "hikcam/image_raw";
    std::string camera_info_topic = "hikcam/camera_info";
    std::string target_frame = "odom";
    std::string camera_frame = "camera_optical_frame";

    bool enable_imshow;
    bool debug;
    VisionMode mode;
    bool right_press;  // 按下右键
    CameraMode cam_mode;
    RobotIdDji robot_id;
    RmcvId rmcv_id;

    // 火控参数
    double timestamp_thresh;               // 开火时间差阈值
    double change_armor_time_thresh;
    double midshot_period;
    double dis_yaw_thresh;
    double gimbal_error_dis_thresh;            // 自动发弹阈值，限制云台误差的球面意义距离
    // 延迟参数
    double response_delay;  // 系统延迟(程序+通信+云台响应)
    double shoot_delay;     // 发弹延迟
    double gimbal_adjust_delay;  // 云台调整延迟时间 仅用于英雄选择目标

    int hero_fixed_armor_id;
    bool enable_hero_dynamic_selection;  // 是否启用英雄动态选择

    //--------------Manager-----------------------------
    // 传统方法感知陀螺/前哨站相关参数
    double census_period_max;
    double anti_outpost_census_period;  // 过中时间的过时阈值, 之前的过中时间与当前时间的差值小于该period, 就认为未过时

    // 装甲目标过滤/选择
    double top_pitch_thresh;         // 判定建筑顶端装甲板的pitch阈值
    double sight_limit;         // 过滤距离过远装甲板
    double high_limit;          // 过滤距离过高过低的装甲板
    double size_limit;          // 按面积过滤装甲板（太小）
    double bound_limit;         // 过滤图像边缘的装甲板（单位为像素）
    double aspect_limit_small;  // 当小装甲板处于40度时宽高比
    double reset_time;         // 若在视野中消失 reset_time秒，认为目标丢失
    cv::Point2d collimation;   // 二维图像上的准星
    // 帧间匹配
    double interframe_dis_thresh;    // 两帧间装甲板的最大移动距离（用于帧间匹配）
};

enum Status { Alive = 0, Absent};

class OutpostArmor{
public:
    OutpostArmor() : status_(Alive) {}
    Eigen::Vector3d getPositionXyz() const{ return Eigen::Vector3d(pc_result_.pose.x, pc_result_.pose.y, pc_result_.pose.z); };
    Eigen::Vector3d getPositionPyd() const{ return xyz2pyd(getPositionXyz()); };

    void init(const ArmorPoseResult &pc_result, double _timestamp);
    void update(ArmorPoseResult &pc_result, double _timestamp);
    void predict(double _timestamp);
    void zeroCrossing(double datum);
    double getYaw() const{ return yaw_round_ * M_PI * 2 + getPositionPyd()[1]; }
    double getYawSpd() { return armor_kf_.getX()[4]; }

    Status status_ = Absent, last_status_ = Absent; // last_status_貌似也无用
    bool in_follow;
    ArmorId armor_id_;
    float score = 0.0f;  // 添加 score 成员

    double alive_ts_ = -1.0;  // 上次Alive的时间戳, 首次出现时间戳, first_ts_ = -1.
    double dis_2d_ = INFINITY;               // 在二维图像中距离准星的距离（仅Alive时有效）
    double area_2d_ = 0.;                    // 在二维图像中的面积（仅Alive时有效）
    int yaw_round_ = 0;  // yaw定义为:世界坐标系下目标相对于车的yaw
    double last_yaw_ = 0;
    int ori_yaw_round_ = 0;  // yaw定义为:世界坐标系下目标相对于车的yaw
    double last_ori_yaw_ = 0;
    
    bool matched_ = false;  // 帧间匹配标志位（这个可以不用放在类里面）
    int tracker_id = -1;  // 添加：Tracker ID，用于帧间匹配，不用于相位分配
    int phase_in_outpost_ = -1; // 装甲板id初始化为-1，表示未分配ID
    ArmorPoseResult pc_result_;  // 滤波前位姿
    OutpostArmorEkf armor_kf_;
    OutpostYawEkf yaw_kf_;
};


class Outpost{
public:
    bool is_rotate = false, is_high_spd_rotate = false, is_move = false;
    struct OutpostPosition {
        Eigen::Vector3d center_;               // 车体中心二维xyz坐标
        std::vector<Eigen::Vector3d> armors_xyz_;  // 四个装甲板的xyz坐标
        std::vector<double> armor_yaws_;       // 每一个装甲板对应的yaw值
        OutpostPosition(int armor_cnt = 3) : armors_xyz_(armor_cnt), armor_yaws_(armor_cnt) {}
    };

    Outpost(){}
    explicit Outpost(ArmorId _id, bool _outpost_kf_init = false, bool _in_follow = false, int _armor_cnt = 3);

    // void add_armor(OutpostArmor armor);
    OutpostPosition predict_positions(double _timestamp);
    void reset(const OutpostCkf::Observe &_observe, int _phase_id, int _armor_cnt, double _timestamp, double _z);
    void update(OutpostCkf::Observe _observe, double _timestamp, int _phase_id);
    // void set_unfollowed();
    
    double get_rotate_spd() { return op_ckf.getState().omega; }
    double get_move_spd() { return sqrt(op_ckf.getState().vx * op_ckf.getState().vx + op_ckf.getState().vy * op_ckf.getState().vy); }

    // 高度相关函数
    void update_armor_height(int armor_id, double observed_height);
    double get_armor_height_by_id(int armor_id);

    inline static OutpostNodeParams params;

    // 高度映射状态
    bool is_hero = false;
    bool height_mapping_initialized_ = false;  // 是否已收集到三块装甲板的高度数据
    double armor_heights_[3] = {1.416, 1.516, 1.616};  // 默认高度，会被实际观测覆盖

    //double target_dis_before_init_ = INFINITY; // 初始化完成前暂时使用的目标距离（保证ROI正常） 已取消
    
    // outpost状态
    Status status = Status::Absent;
    OutpostPosition now_position_;
    double alive_ts = -1;
    ArmorId id;

    // 滤波
    // MathFilter common_rotate_spd = MathFilter(5); 好像并未用上
    MathFilter common_middle_dis, common_middle_pitch, common_yaw_spd = MathFilter(10);
    MathFilter const_z_filter = MathFilter(20, ArithmeticMean);
    MathFilter center_pos_filter[3];
    // MathFilter top_pos_filter[3];
    MathFilter armor_height_filter[3];  // 分别对三个装甲板高度进行滤波 英雄
    // MathFilter aiming_z_filter = MathFilter(20); //对瞄准点的z滤波 步兵

    OutpostCkf op_ckf;

    // 过中时间线性拟合器
    step_fitter T_solver;

    bool outpost_kf_init = false;

    // armor相关
    bool in_follow;
    std::vector<OutpostArmor> armors; 
    int armor_cnt = 3;
    // 其他数据
    double ori_diff;
    double last_yaw;
    int yaw_round = 0;
    double min_dis_2d = INFINITY;

    // timestamp, yaw
    std::deque<std::pair<double, double>> yaw_increase_history, yaw_decrease_history;// yaw的增加历史 和 yaw的减少历史

    // std::queue<double> dis_yaw_queue;
    // std::multiset<double> dis_yaw_set;
    // int max_yaw_num = 100;

    
};

class OutpostManager{
public:
    OutpostManager() = default;
    ~OutpostManager() = default;

    // void init(rclcpp::Node* _node, std::string visual_topic = "outpost_visual"){
    //     outpost_visual.init(_node, visual_topic);
    //     data_visual.init(_node, visual_topic+"_data");
    // }

    void updateArmors(const std::vector<TrackedArmor>& tracked_armors, double timestamp);
    void updateOutpost();
    
    // 设置相机内参（用于可视化）
    void setCameraInfo(const std::vector<double>& k, const std::vector<double>& d) {
        camera_k_ = k;
        camera_d_ = d;
    }
    // 设置图像用于可视化
    void setImage(const cv::Mat& img) {
        result_img_ = img.clone();
    }
    // 获取可视化图像
    cv::Mat getVisualizationImage() const {
        return result_img_;
    }
    //外部加载
    inline static OutpostNodeParams params;

    // 机器人自身信息
    RmcvId self_id;
    RmRobotMsg robot;
    // outpost 管理（存为对象以简化使用）
    Outpost outpost;
    // int target_outpost_id;
    // 接收到的检测消息
    OutpostDetectMsg recv_detection;

    // 可视化相关
    cv::Mat result_img_;
    std::vector<double> camera_k_;
    std::vector<double> camera_d_;
    // 可注册的重投影函数，由 OutpostNode 构造时注入
    std::function<cv::Point2d(const Eigen::Vector3d&)> projector;

    // 上一帧装甲板信息（用于过中检测）
    std::map<int, TrackedArmor> last_frame_armors_;
    double last_frame_timestamp_ = -1.0;

    // 相位分配状态：仅在前哨站自瞄启动后第一块装甲板初始化为0
    bool phase_initialized_ = false;
    int last_active_tracker_id_ = -1;
    int last_active_phase_ = -1;
    double last_active_yaw_ = 0.0;
    double last_active_ts_ = -1.0;

};

class OutpostNode : public rclcpp::Node{
public:
    explicit OutpostNode(const rclcpp::NodeOptions &options);
    ~OutpostNode() override = default;

    void loadAllParams();

private:
    OutpostNodeParams params_;
    OutpostManager manager_;
    Outpost outpost;  // 添加Outpost成员
    RmImuMsg imu;
    std::shared_ptr<Ballistic> bac;

    ControlMsg off_cmd;
    // 位姿解算与变换相关
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener;

    // communicate
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection_sub; // 修改为新的检测消息
    rclcpp::Subscription<RmRobotMsg>::SharedPtr robot_sub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub; // 添加相机信息订阅 需要吗？
    image_transport::CameraSubscriber camera_sub;
    rclcpp::Publisher<ControlMsg>::SharedPtr control_pub;
    // rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr target_dis_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    // Latency
    rclcpp::Clock CLK;
    rclcpp::Time tick, tock;
    double prev_latency;

    double last_shoot_t = 0;
    double next_middle_time = -1;
    double last_next_middle_time = -1;
    double yaw_min;
    double yaw_max;
    VisionMode last_mode = VisionMode::AUTO_AIM;
    cv::Mat current_image_;
    // 添加相机内参
    std::vector<double> camera_k_;
    std::vector<double> camera_d_;
    bool camera_info_received_ = false;

    Eigen::Isometry3d getTrans(const std::string& source_frame, 
                               const std::string& target_frame);
     // 坐标变换辅助函数
    bool transformPoseToOdom(const geometry_msgs::msg::Pose& pose_camera, 
                            const std_msgs::msg::Header& header,
                            geometry_msgs::msg::Pose& pose_odom);        
    // 将相机系下的位姿转换到odom系
    bool convertCameraToOdom(const Eigen::Vector3d& pos_camera,
                           const Eigen::Quaterniond& ori_camera,
                           const builtin_interfaces::msg::Time& stamp,
                           Eigen::Vector3d& pos_odom,
                           Eigen::Quaterniond& ori_odom);
    // 重投影
    cv::Point2d projectPointToImage(const Eigen::Vector3d& point_odom);
    Eigen::Vector3d transPoint(const Eigen::Vector3d& source_point, 
                            const std::string& source_frame, 
                            const std::string& target_frame);

    Ballistic::BallisticResult center_ballistic(Eigen::Vector3d &predict_center, double armor_z);
    Ballistic::BallisticResult calc_ballistic(int armor_phase, double delay, Eigen::Vector3d &predict_pos, double armor_height);

    OpNewSelectArmor select_armor_directly();

    //英雄动态选择装甲板瞄准
    int select_armor_dynamically(double &selected_time_diff, double &selected_total_delay, 
                                 double &selected_system_delay, std::vector<double> &time_diffs,
                                 std::vector<double> &total_delays, std::vector<bool> &adjust_flags);

    ControlMsg get_command();

    void publishMarkers(double timestamp);

    void detectionCallback(vision_msgs::msg::Detection2DArray::UniquePtr detection_msg); // 修改回调函数
    void camera_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                         const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg);
    void robotCallback(RmRobotMsg::SharedPtr robot_msg);
};

#endif // _OUTPOST_NODE_H