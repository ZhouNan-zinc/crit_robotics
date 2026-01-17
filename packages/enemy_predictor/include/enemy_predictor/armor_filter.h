#ifndef _ARMOR_FILTER_H
#define _ARMOR_FILTER_H

#include <iostream>
#include <Eigen/Dense>


/*class ArmorEKF {
public:
    static const int STATE_NUM = 6;  // [x, y, z, vx, vy, vz]
    static const int OBSERVE_NUM = 3; // [x, y, z]
    
    using Vx = Eigen::Matrix<double, STATE_NUM, 1>;
    using Vz = Eigen::Matrix<double, OBSERVE_NUM, 1>;
    using Mxx = Eigen::Matrix<double, STATE_NUM, STATE_NUM>;
    using Mzz = Eigen::Matrix<double, OBSERVE_NUM, OBSERVE_NUM>;
    using Mzx = Eigen::Matrix<double, OBSERVE_NUM, STATE_NUM>; 
    using Mxz= Eigen::Matrix<double, STATE_NUM, OBSERVE_NUM>; 

    struct EKFConfig{
        Vx config_Q;  // 过程噪声
        Vz config_R;  // 观测噪声
    };
    Vx Xe;          // [x, y, z, vx, vy, vz]
    Mxx P;          // 状态协方差
    Mxx Q;          // 过程噪声协方差
    Mzz R;          // 观测噪声协方差
    Mxx F;          // 状态转移矩阵
    Mzx H;       
    double last_timestamp;
    bool is_initialized_;
    static EKFConfig config_;

   
    // 构造函数
    ArmorEKF();
    
    // 预测位置（外部获取预测值）
    Vz predict_position(double _timestamp) const;
    
    // EKF预测
    Vx predict(double _timestamp);
    
    // 观测模型
    Vz h(Vx _X);
    
    void init(Eigen::Vector3d measured_xyz, double _timestamp, const EKFConfig& config);
    
    // EKF更新
    Vx update(Vz observe_Zk, double timestamp);
    
    Vx get_state() const { return Xe; }
    Eigen::Vector3d get_position() const { return Eigen::Vector3d(Xe[0], Xe[1], Xe[2]); }
    Eigen::Vector3d get_velocity() const { return Eigen::Vector3d(Xe[3], Xe[4], Xe[5]); }
    bool is_initialized() const { return is_initialized_; }
};*/
class ArmorXYYAWEKF { 
    public:
    static const int STATE_NUM = 6;  // [x, y, yaw, vx, vy, omega]
    static const int OBSERVE_NUM = 3; // [x, y, yaw]
    
    using Vx = Eigen::Matrix<double, STATE_NUM, 1>;
    using Vz = Eigen::Matrix<double, OBSERVE_NUM, 1>;
    using Mxx = Eigen::Matrix<double, STATE_NUM, STATE_NUM>;
    using Mzz = Eigen::Matrix<double, OBSERVE_NUM, OBSERVE_NUM>;
    using Mzx = Eigen::Matrix<double, OBSERVE_NUM, STATE_NUM>; 
    using Mxz= Eigen::Matrix<double, STATE_NUM, OBSERVE_NUM>; 

    struct XYYAWEKFConfig{
        Vx config_Q;  // 过程噪声
        Vz config_R;  // 观测噪声
        Vx config_P;
    };
    Vx Xe;          // [x, y, z, vx, vy, yaw, omega]
    Mxx P;          // 状态协方差
    Mxx Q;          // 过程噪声协方差
    Mzz R;          // 观测噪声协方差
    Mxx F;          // 状态转移矩阵
    Mzx H;       
    double last_timestamp;
    bool is_initialized_;
    static XYYAWEKFConfig config_;

   
    // 构造函数
    ArmorXYYAWEKF();
    
    // 预测位置（外部获取预测值）
    Vz predict_position(double _timestamp);
    
    // EKF预测
    Vx predict(double _timestamp);
    
    // 观测模型
    Vz h(Vx _X);
    
    void init(Eigen::Vector3d measured_xyyaw, double _timestamp);
    
    // EKF更新
    Vx update(Vz observe_Zk, double timestamp);
    
    Vx get_state() const { return Xe; }
    Eigen::Vector3d get_position() const { return Eigen::Vector3d(Xe[0], Xe[1],Xe[2]); }
    Eigen::Vector3d get_velocity() const { return Eigen::Vector3d(Xe[3], Xe[4], Xe[5]); }
    bool is_initialized() const { return is_initialized_; }
};
class ZEKF { 
    public:
    static const int STATE_NUM = 2;  // [z, vz]
    static const int OBSERVE_NUM = 1; // [z]
    
    using Vx = Eigen::Matrix<double, STATE_NUM, 1>;
    using Vz = Eigen::Matrix<double, OBSERVE_NUM, 1>;
    using Mxx = Eigen::Matrix<double, STATE_NUM, STATE_NUM>;
    using Mzz = Eigen::Matrix<double, OBSERVE_NUM, OBSERVE_NUM>;
    using Mzx = Eigen::Matrix<double, OBSERVE_NUM, STATE_NUM>; 
    using Mxz= Eigen::Matrix<double, STATE_NUM, OBSERVE_NUM>; 

    struct ZEKFConfig{
        Vx config_Q;  // 过程噪声
        Vz config_R;  // 观测噪声
    };
    Vx Xe;          // [z, vz]
    Mxx P;          // 状态协方差
    Mxx Q;          // 过程噪声协方差
    Mzz R;          // 观测噪声协方差
    Mxx F;          // 状态转移矩阵
    Mzx H;       
    double last_timestamp;
    bool is_initialized_;
    static ZEKFConfig config_;

   
    // 构造函数
    ZEKF();
    
    // 预测位置（外部获取预测值）
    Vz predict_position(double _timestamp);
    
    // EKF预测
    Vx predict(double _timestamp);
    
    // 观测模型
    Vz h(Vx _X);
    
    void init(double z, double _timestamp);
    
    // EKF更新
    Vx update(Vz observe_Zk, double timestamp);
    
    //Vx get_state() const { return Xe; }
    //Eigen::Vector2d get_position() const { return Eigen::Vector2d(Xe[0], Xe[1]); }
    //Eigen::Vector2d get_velocity() const { return Eigen::Vector2d(Xe[2], Xe[3]); }
    //bool is_initialized() const { return is_initialized_; }
};

#endif