#pragma once
#include <Eigen/Dense>
#include <mutex>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

//class YawTrajectoryPlanner {
//public:
//    struct Config {
//        double max_velocity = 3.0;          // 最大角速度 (rad/s) ~ 172°/s
//        double max_acceleration = 10.0;     // 最大角加速度 (rad/s²) ~ 573°/s²
//        double max_jerk = 50.0;             // 最大加加速度 (rad/s³) ~ 2865°/s³
//        
//        // 电控相关参数
//        double motor_control_period = 0.002;  // 电控周期 (s) 500Hz
//        double lookahead_time = 0.010;       // 发送给电控的位置超前时间 (s)
//        double interpolation_factor = 0.5;   // 插值因子 (0-1), 用于平滑
//        
//        // 规划参数
//        double tolerance = 0.01;            // 收敛容差 (rad) ~ 0.57°
//        double min_update_threshold = 0.001; // 最小更新阈值 (rad) ~ 0.057°
//        
//        // 调试参数
//        bool enable_debug = false;
//        double debug_interval = 0.1;        // 调试输出间隔 (s)
//
//        Config() = default;
//    }config_;
//    
//    struct State {
//        double position;      // 当前位置 (rad)
//        double velocity;      // 当前速度 (rad/s)
//        double acceleration;  // 当前加速度 (rad/s²)
//        
//        // 转换为 Eigen 向量
//        Eigen::Vector3d toVector() const;
//        
//        // 从 Eigen 向量设置状态
//        void fromVector(const Eigen::Vector3d& vec);
//        
//        // 重置
//        void reset();
//    };
//    
//    // 控制输出结构 - 简化版，只发送位置
//    struct ControlOutput {
//        double target_position_abs;  // 发送给电控的目标位置 (rad)
//        bool is_finished;            // 是否完成规划
//        bool is_active;              // 是否正在规划
//        double progress;             // 进度 0-1
//        double time_remaining;       // 预计剩余时间 (s)
//        
//        // 内部状态信息（用于调试）
//        double planned_position;     // 规划的目标位置 (rad)
//        double planned_velocity;     // 规划的目标速度 (rad/s)
//        double planned_acceleration; // 规划的目标加速度 (rad/s²)
//    };
//    
//    // 规划状态结构
//    struct PlannerStatus {
//        bool is_active;
//        double start_yaw;      // 规划起点 (绝对)
//        double target_yaw;     // 规划目标 (绝对)
//        double current_yaw;    // 当前yaw (绝对)
//        double progress;       // 进度 0-1
//        double time_remaining; // 剩余时间
//        double max_velocity;   // 规划的最大速度
//        double total_time;     // 总规划时间
//        double next_target;    // 下一个要发送给电控的目标
//    };
//    
//    // 构造函数
//    YawTrajectoryPlanner();
//    explicit YawTrajectoryPlanner(const Config& config);
//    
//    // 主接口：设置目标yaw（绝对角度）
//    bool setTargetYaw(double target_yaw_abs, double current_yaw_abs);
//    
//    // 获取下一个要发送给电控的目标位置
//    // 输入：当前云台yaw（绝对角度），当前时间
//    // 输出：电控应该到达的目标位置（绝对角度）
//    ControlOutput getMotorTarget(double current_yaw_abs);
//    
//    // 带时间戳的版本（更精确）
//    ControlOutput getMotorTarget(double current_yaw_abs, 
//                                std::chrono::steady_clock::time_point current_time);
//    
//    // 获取规划状态
//    PlannerStatus getStatus(double current_yaw_abs);
//    
//    // 紧急停止
//    void emergencyStop();
//    
//    // 重置
//    void reset();
//    
//    // 更新配置
//    void updateConfig(const Config& config);
//    
//    // 设置电控周期（动态调整）
//    void setMotorControlPeriod(double period);
//    
//private:
//   
//    State current_state_abs_;   // 绝对坐标系状态
//    State current_state_rel_;   // 相对坐标系状态
//    
//    // 规划参数
//    bool is_active_;
//    double start_yaw_abs_;     // 规划起点 (绝对)
//    double target_yaw_abs_;    // 规划目标 (绝对)
//    double target_delta_yaw_;  // 需要转动的角度差
//    double target_direction_;  // 转动方向: 1.0 正向, -1.0 反向
//    
//    // 轨迹参数
//    double distance_;           // 需要转动的距离
//    double max_velocity_reached_; // 能达到的最大速度
//    double acc_time_;           // 加速段时间
//    double const_time_;         // 匀速段时间
//    double dec_time_;           // 减速段时间
//    double total_time_;         // 总时间
//    
//    // 时间相关
//    std::chrono::steady_clock::time_point start_time_;
//    std::chrono::steady_clock::time_point last_update_time_;
//    std::chrono::steady_clock::time_point last_motor_send_time_;
//    std::chrono::steady_clock::time_point last_debug_time_;
//    mutable std::mutex mutex_;
//    
//    // 私有方法
//    void calculateTrajectoryParams();
//    double solveTriangleTime(double v0, double a, double s);
//    State getPlannedState(double t) const;
//    double normalizeAngle(double angle) const;
//    
//    // 插值方法
//    double interpolatePosition(double current_pos, double planned_pos, 
//                              double current_vel, double planned_vel,
//                              double alpha) const;
//    
//    // 调试输出
//    void debugPrint(const std::string& message) const;
//    void debugPrintStatus(double current_yaw_abs, const ControlOutput& output) const;
//};