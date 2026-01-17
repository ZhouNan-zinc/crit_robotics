#include "enemy_predictor/enemy_filter.h"

EnemyCKF::CKFConfig EnemyCKF::config_;
EnemyCKF::EnemyCKF() : sample_num_(2 * STATE_NUM), is_initialized_(false) {
   
    samples_.resize(sample_num_);
    sample_X.resize(sample_num_);
    sample_Z.resize(sample_num_);
    weights_.resize(sample_num_);
    
    double weight = 1.0 / (2 * STATE_NUM);
    for (int i = 0; i < sample_num_; ++i) {
        weights_[i] = weight;
    }
    
    
    Xe = Vx::Zero();
    Pe = Mxx::Identity();
    Pp = Mxx::Identity();
    Q = Mxx::Identity() * 0.1;
    R = Mzz::Identity() * 0.01;
    
}
    void EnemyCKF::reset(const Eigen::Vector3d& position, double yaw, int _phase_id, 
               int _armor_cnt, double _timestamp, 
               std::vector<double> _radius, std::vector<double> _z) {
        armor_cnt_ = _armor_cnt;
        angle_dis_ = 2 * M_PI / armor_cnt_;
        
        const_radius_ = _radius;
        const_z_ = _z;
        assert(const_radius_.size() == armor_cnt_);
        assert(const_z_.size() == armor_cnt_);
        
        // 直接初始化状态向量
        Xe = Vx::Zero();
        Xe[0] = position.x() - const_radius_[_phase_id] * cos(yaw + _phase_id * angle_dis_); // x
        Xe[1] = 0;  // vx
        Xe[2] = position.y() - const_radius_[_phase_id] * sin(yaw + _phase_id * angle_dis_); // y
        Xe[3] = 0;  // vy
        Xe[4] = yaw;  // yaw
        Xe[5] = 0;  // omega
        
        Pe = config_.config_Pe;
        Pp = Mxx::Identity();
        last_timestamp_ = _timestamp;
        is_initialized_ = true;
    }

    // 更新函数 - 直接接受位置和yaw，不再需要Observe结构体
    void EnemyCKF::update(const Eigen::Vector3d& position, double yaw, 
                double _timestamp, int _phase_id) {
        double dT = _timestamp - last_timestamp_;
        
        Vz z;
        z << position.x(), position.y(), yaw;
        
        predict(_timestamp);
        measure(z, _phase_id);
        correct(z);
        
        last_timestamp_ = _timestamp;
    }

    // 预测特定!!装甲板位置
    Eigen::Vector3d EnemyCKF:: predictArmorPosition(int phase_id, double predict_time) {
        double dt = predict_time;
        Vx predicted_state = f(Xe, dt);
        
        double pred_x = predicted_state[0] + const_radius_[phase_id] * cos(predicted_state[4] + phase_id * angle_dis_);
        double pred_y = predicted_state[2] + const_radius_[phase_id] * sin(predicted_state[4] + phase_id * angle_dis_);
        double pred_z = const_z_[phase_id];
        double pred_yaw = predicted_state[4];
        
        return Eigen::Vector3d(pred_x, pred_y, pred_z);
    }
    void EnemyCKF::initializeCKF() {
        if (!is_initialized_){
           samples_.resize(sample_num_);
           sample_X.resize(sample_num_);
           sample_Z.resize(sample_num_);
           weights_.resize(sample_num_);
           
           double weight = 1.0 / (2 * STATE_NUM);
           for (int i = 0; i < sample_num_; ++i) {
               weights_[i] = weight;
           }
           Pe = config_.config_Pe;
           Pp = Mxx::Identity();
           Xe = Vx::Zero();
           Q = Mxx::Identity() * 0.1;
           R = Mzz::Identity() * 0.01;
           is_initialized_ = true;
        }
    }

    // CKF核心算法
    void EnemyCKF::SRCR_sampling(const Vx& x, const Mxx& P) {
        double sqrtn = sqrt(STATE_NUM);
        Eigen::LLT<Mxx> cholesky(P);
        Mxx S = cholesky.matrixL();
        
        for (int i = 0; i < STATE_NUM; ++i) {
            samples_[i] = x + sqrtn * S.col(i);
            samples_[i + STATE_NUM] = x - sqrtn * S.col(i);
        }
    }

    void EnemyCKF::predict(double timestamp) {
        if (!is_initialized_) {
            std::cerr << "CKF not initialized!" << std::endl;
            return;
        }
        
        double dt = timestamp - last_timestamp_;
        if (dt <= 0) dt = 0.01;
        
        calcQ(dt);
        SRCR_sampling(Xe, Pe);
        
        Xp = Vx::Zero();
        for (int i = 0; i < sample_num_; ++i) {
            sample_X[i] = f(samples_[i], dt);
            Xp += weights_[i] * sample_X[i];
        }

        Pp = Mxx::Zero();
        for (int i = 0; i < sample_num_; ++i) {
            Pp += weights_[i] * (sample_X[i] - Xp) * (sample_X[i] - Xp).transpose();
        }
        Pp += Q;
        
        Xe = Xp;
        Pe = Pp;
    }

    void EnemyCKF::measure(const Vz& z, int phase_id) {
        calcR(z);
        
        Zp = Vz::Zero();
        for (int i = 0; i < sample_num_; ++i) {
            sample_Z[i] = h(samples_[i], phase_id);
            Zp += weights_[i] * sample_Z[i];
        }

        Pzz = Mzz::Zero();
        for (int i = 0; i < sample_num_; ++i) {
            Pzz += weights_[i] * (sample_Z[i] - Zp) * (sample_Z[i] - Zp).transpose();
        }
        Pzz += R;
    }

    void EnemyCKF::correct(const Vz& z) {
        Pxz = Mxz::Zero();
        for (int i = 0; i < sample_num_; ++i) {
            Pxz += weights_[i] * (sample_X[i] - Xp) * (sample_Z[i] - Zp).transpose();
        }
        
        K = Pxz * Pzz.inverse();
        Xe = Xp + K * (z - Zp);
        Pe = Pp - K * Pzz * K.transpose();
    }

    // 系统模型
    EnemyCKF::Vx EnemyCKF::f(const Vx& x, double timestamp) const {
        Vx ans = x;
        double dt = timestamp - last_timestamp_;
        ans[0] += x[1] * dt;  // x += vx*dt
        ans[2] += x[3] * dt;  // y += vy*dt
        ans[4] += x[5] * dt;  // yaw += w*dt
        return ans;
    }

    EnemyCKF::Vz EnemyCKF::h(const Vx& x, int phase_id) const {
        Vz result;
        result[0] = x[0] + const_radius_[phase_id] * cos(x[5] + phase_id * angle_dis_);  // x
        result[1] = x[2] + const_radius_[phase_id] * sin(x[5] + phase_id * angle_dis_);  // y
        result[2] = x[5];  // yaw
        return result;
    }

    void EnemyCKF::calcQ(double dt) {
        static double dTs[4];
        dTs[0] = dt;
        for (int i = 1; i < 4; ++i) {
            dTs[i] = dTs[i - 1] * dt;
        }
        
        double q_x_x = dTs[3] / 3 * config_.Q2_X;      // dt³/3
        double q_x_vx = dTs[2] / 2 * config_.Q2_X;     // dt²/2
        double q_vx_vx = dTs[1] * config_.Q2_X;        // dt
        
        double q_y_y = dTs[3] / 3 * config_.Q2_Y;      // 区分X和Y方向
        double q_y_vy = dTs[2] / 2 * config_.Q2_Y;
        double q_vy_vy = dTs[1] * config_.Q2_Y;
        
        // 角度使用不同的噪声强度
        double q_yaw_yaw = dTs[3] / 3 * config_.Q2_YAW;
        double q_yaw_omega = dTs[2] / 2 * config_.Q2_YAW;
        double q_omega_omega = dTs[1] * config_.Q2_YAW;
        
        Q = Mxx::Zero();
        Q.block(0, 0, 2, 2) << q_x_x, q_x_vx, q_x_vx, q_vx_vx;
        Q.block(2, 2, 2, 2) << q_y_y, q_y_vy, q_y_vy, q_vy_vy;
        Q.block(4, 4, 2, 2) << q_yaw_yaw, q_yaw_omega, q_yaw_omega, q_omega_omega;
    }

    void EnemyCKF::calcR(const Vz& z) {
        Vz R_vec;
        R_vec << abs(config_.R_XYZ * z[0]), abs(config_.R_XYZ * z[1]),config_.R_YAW;
        R = R_vec.asDiagonal();
    }

    double EnemyCKF:: get_average(const std::vector<double>& vec) const {
        return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
    }
