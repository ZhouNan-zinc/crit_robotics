#ifndef _ENEMY_CKF_H
#define _ENEMY_CKF_H
#include <iostream>
#include <Eigen/Dense>
#include <numeric> 


class EnemyCKF {
public:
    static const int STATE_NUM = 8;  // [x, vx, y, vy, yaw, vyaw, r1, r2]
    static const int OBSERVE_NUM = 3; // [x, y, yaw]
    
    using Vx = Eigen::Vector<double, STATE_NUM>;
    using Vz = Eigen::Vector<double, OBSERVE_NUM>;
    using Mxx = Eigen::Matrix<double, STATE_NUM, STATE_NUM>;
    using Mzz = Eigen::Matrix<double, OBSERVE_NUM, OBSERVE_NUM>;
    using Mxz = Eigen::Matrix<double, STATE_NUM, OBSERVE_NUM>;

    // 状态变量
    Vx Xe;  // 当前状态估计 [x, vx, y, vy, yaw, vyaw, r1, r2]
    Vx Xp;  // 预测状态
    
    // CKF变量
    Mxx Pe, Pp, Q;
    Mzz R, Pzz;
    Mxz Pxz, K;
    Vz Zp;
    
    int sample_num_;
    std::vector<double> weights_;
    std::vector<Vx> samples_;
    std::vector<Vx> sample_X;
    std::vector<Vz> sample_Z;
    std::vector<double> initial_radius{0.25, 0.25};
    //double radius;
    
    double last_timestamp_;
    bool is_initialized_;

    // 机器人特定参数
    double angle_dis_;

    // 配置结构体
    struct CKFConfig {
        double Q2_X = 0.1;
        double Q2_Y = 0.1;
        double Q2_YAW = 0.01;
        double Q_r = 0.01;
        double R_XYZ = 0.01;
        double R_YAW = 0.001;
        Mxx config_Pe;
    };
    
    static CKFConfig config_;

    // 构造函数
    EnemyCKF();

    // 重置滤波器
    void reset(const Eigen::Vector3d& position, double yaw, int _phase_id, 
               double _timestamp);
    
    // 更新函数
    void update(const Eigen::Vector3d& position, double yaw, 
                double _timestamp, int _phase_id);
    
    // 预测特定装甲板位置
    Eigen::Vector3d predictArmorPosition(double z, int phase_id, double dt, double timestamp);
    Eigen::Vector3d predictCenterPosition(double z, double dt, double timestamp);
    // 初始化CKF
    void initializeCKF();
    
    // CKF核心算法
    void SRCR_sampling(const Vx& x, const Mxx& P);
    void predict(double timestamp);
    void measure(const Vz& z, int phase_id);
    void correct(const Vz& z);
    
    // 系统模型和观测模型
    Vx f(const Vx& x, double timestamp) const;
    Vz h(const Vx& x, int phase_id) const;
    
    // 噪声计算
    void calcQ(double dt);
    void calcR(const Vz& z);
    
    // 工具函数
    double get_average(const std::vector<double>& vec) const;
    bool is_initialized() const { return is_initialized_; }

};
#endif