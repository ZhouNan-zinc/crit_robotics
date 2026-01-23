#include "enemy_predictor/armor_filter.h"

/*ArmorEKF::EKFConfig ArmorEKF::config_;
ArmorEKF::ArmorEKF(){
    H.setZero();
    H(0, 0) = H(1, 1) = H(2, 2) = 1;
    P = Mxx::Identity();
    Q = Mxx::Identity();
    R = Mzz::Identity();
    F = Mxx::Identity();
    Xe.setZero();
    last_timestamp = 0.0;
    bool is_initialized_ = false;
}
ArmorEKF::Vz ArmorEKF::predict_position(double _timestamp) const{
        Vz res;
        double dt = _timestamp - last_timestamp;
        res[0] = Xe[0] + Xe[3] * 0.01*dt;
        res[1] = Xe[1] + Xe[4] * 0.01*dt;
        res[2] = Xe[2] + Xe[5] * 0.01*dt;
        return res;
    }
ArmorEKF::Vx ArmorEKF::predict(double _timestamp){
       double _dt = _timestamp - last_timestamp;
       F(0,3)=F(1,4)=F(2,5)=_dt;
       return F*Xe ;
    }
ArmorEKF::Vz ArmorEKF::h(Vx _X){
        Vz z_ans = H*_X;
        return z_ans;
    }
void ArmorEKF::init(Eigen ::Vector3d measured_xyz,double _timestamp,const EKFConfig& config){
     if (!is_initialized_) {
        config_ = config;
        Xe << measured_xyz.x(), measured_xyz.y(), measured_xyz.z(), 0, 0, 0;
        P = Mxx::Identity();
        Q = config.config_Q.asDiagonal();
        R = config.config_R.asDiagonal();
        last_timestamp = _timestamp;
        is_initialized_ = true;
     }
}
ArmorEKF::Vx ArmorEKF::update(Vz observe_Zk,double timestamp){
    if (!is_initialized_) {
        if (!is_initialized_) {
            
            std::cerr << "EKF not initialized!" << std::endl;
            return Xe;
        }
    }
    double dt = timestamp - last_timestamp;
    if (dt <= 0) { 
        std::cerr << "Invalid dt: " << dt << ", using default 0.01" << std::endl;
        dt = 0.01;
    }

    Vx Xp = predict (timestamp);
    Mxx Pp = F* P* F.transpose()+Q;
    Mzz S = H* Pp* H.transpose()+R;
    Mxz K = Pp* H.transpose()*S.inverse();
    double min_gain = 0.01;
    double max_gain = 0.5; 
    for (int i = 0; i < K.rows(); ++i) {
        for (int j = 0; j < K.cols(); ++j) {
            if (std::abs(K(i,j)) < min_gain) {
                K(i,j) = min_gain;
            } else if (std::abs(K(i,j)) > max_gain) {
                K(i,j) = max_gain;
            }
        }
    }
    Xe = Xp + K* (observe_Zk- h(Xp));
    P = (Mxx::Identity()-K* H)* Pp;
    last_timestamp = timestamp;
    return Xe;
    //for (int i = 0; i < Xe.size(); ++i) {
    //std::cout << Xe(i);
    //if (i < Xe.size() - 1) std::cout << ", ";
    //}

}*/
ArmorXYYAWEKF::XYYAWEKFConfig ArmorXYYAWEKF::config_;
ArmorXYYAWEKF::ArmorXYYAWEKF(){
    H.setZero();
    H(0, 0) = H(1, 1) = H(2, 2) = 1;
    P = Mxx::Identity();
    Q = Mxx::Identity();
    R = Mzz::Identity();
    F = Mxx::Identity();
    Xe.setZero();
    last_timestamp = 0.0;
    bool is_initialized_ = false;
}
ArmorXYYAWEKF::Vz ArmorXYYAWEKF::predict_position(double dt) {
        Vz res;
        //double dt = _timestamp - last_timestamp ;
        //last_timestamp = _timestamp;
        //std::cout << "predict time: " << dt << std::endl;
        res[0] = Xe[0] + Xe[3] * dt;
        res[1] = Xe[1] + Xe[4] * dt;
        res[2] = Xe[2] + Xe[5] * dt;
        return res;
    }
ArmorXYYAWEKF::Vx ArmorXYYAWEKF::predict(double _timestamp){
       double _dt = _timestamp - last_timestamp;
       F(0,3)=F(1,4)=F(2,5)=_dt;
       return F*Xe ;
    }
ArmorXYYAWEKF::Vz ArmorXYYAWEKF::h(Vx _X){
        Vz z_ans = H*_X;
        return z_ans;
    }
void ArmorXYYAWEKF::init(Eigen ::Vector3d measured_xyyaw,double _timestamp){
     if (!is_initialized_) {
        
        Xe << measured_xyyaw.x(), measured_xyyaw.y(),measured_xyyaw.z(), 0, 0, 0;
        Q = config_.config_Q.asDiagonal();
        R = config_.config_R.asDiagonal();
        P = config_.config_P.asDiagonal();
        last_timestamp = _timestamp;
        is_initialized_ = true;
     }
}
ArmorXYYAWEKF::Vx ArmorXYYAWEKF::update(Vz observe_Zk,double timestamp){
    if (!is_initialized_) {
        if (!is_initialized_) {
            
            std::cerr << "EKF not initialized!" << std::endl;
            return Xe;
        }
    }
    double dt = timestamp - last_timestamp;
    if (dt <= 0) { 
        std::cerr << "XYYAWEKF:Invalid dt: " << dt << ", using default 0.01" << std::endl;
        dt = 0.01;
    }

    Vx Xp = predict (timestamp);
    Mxx Pp = F* P* F.transpose()+Q;
    Mzz S = H* Pp* H.transpose()+R;
    Mxz K = Pp* H.transpose()*S.inverse();
    double min_gain = 0.01;
    double max_gain = 0.5; 
    for (int i = 0; i < K.rows(); ++i) {
        for (int j = 0; j < K.cols(); ++j) {
            if (std::abs(K(i,j)) < min_gain) {
                K(i,j) = min_gain;
            } else if (std::abs(K(i,j)) > max_gain) {
                K(i,j) = max_gain;
            }
        }
    }
    Xe = Xp + K* (observe_Zk- h(Xp));
    P = (Mxx::Identity()-K* H)* Pp;
    last_timestamp = timestamp;
    return Xe;
    //for (int i = 0; i < Xe.size(); ++i) {
    //std::cout << Xe(i);
    //if (i < Xe.size() - 1) std::cout << ", ";
    //}

};


ZEKF::ZEKFConfig ZEKF::config_;
ZEKF::ZEKF(){
    H.setZero();
    H(0, 0) = 1;
    P = Mxx::Identity();
    Q = Mxx::Identity();
    R = Mzz::Identity();
    F = Mxx::Identity();
    Xe.setZero();
    last_timestamp = 0.0;
    is_initialized_ = false;
}
ZEKF::Vz ZEKF::predict_position(double dt) {
        Vz res;
        res[0] = Xe[0] + Xe[1] * dt;
        return res;
    }
ZEKF::Vx ZEKF::predict(double _timestamp){
       double _dt = _timestamp - last_timestamp;
       F(0,1) = _dt;
       return F*Xe ;
    }
ZEKF::Vz ZEKF::h(Vx _X){
        Vz z_ans = H*_X;
        return z_ans;
    }
void ZEKF::init(double _z, double _timestamp){
        H(0, 0) = 1;
        last_timestamp = _timestamp;
        Xe << _z, 0;
        P = Mxx::Identity();
        Q = config_.config_Q.asDiagonal();
        R = config_.config_R.asDiagonal();
        is_initialized_ = true;
    }
ZEKF::Vx ZEKF::update(Vz observe_Zk,double timestamp){
    if (!is_initialized_) {
        if (!is_initialized_) {
            
            std::cerr << "ZEKF not initialized!" << std::endl;
            return Xe;
        }
    }
    double dt = timestamp - last_timestamp;
    if (dt <= 0) { 
        std::cerr << "ZEKF:Invalid dt: " << dt << ", using default 0.01" << std::endl;
        dt = 0.01;
    }

    Vx Xp = predict (timestamp);
    Mxx Pp = F* P* F.transpose()+Q;
    Mzz S = H* Pp* H.transpose()+R;
    Mxz K = Pp* H.transpose()*S.inverse();
    double min_gain = 0.01;
    double max_gain = 0.5; 
    for (int i = 0; i < K.rows(); ++i) {
        for (int j = 0; j < K.cols(); ++j) {
            if (std::abs(K(i,j)) < min_gain) {
                K(i,j) = min_gain;
            } else if (std::abs(K(i,j)) > max_gain) {
                K(i,j) = max_gain;
            }
        }
    }
    Xe = Xp + K* (observe_Zk- h(Xp));
    P = (Mxx::Identity()-K* H)* Pp;
    last_timestamp = timestamp;
    return Xe;
    //for (int i = 0; i < Xe.size(); ++i) {
    //std::cout << Xe(i);
    //if (i < Xe.size() - 1) std::cout << ", ";
    //}

}



