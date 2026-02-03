#include "outpostaim/outpost_estimator.h"
#include "outpostaim/outpost_node.h"
#include <cmath>

Eigen::Vector3d operator*(const Eigen::Isometry3d &T, const Eigen::Vector3d &v) {
    Eigen::Vector4d V(v[0], v[1], v[2], 1);
    return (T * V).block<3, 1>(0, 0);
}

//----------------------------------------OutpostArmor类---------------------------------------

void OutpostArmor::init(const ArmorPoseResult &pc_result, double _timestamp){
    pc_result_ = pc_result;
    // 重置滤波器
    Eigen::Matrix<double, 3, 1> new_pyd = pc_result_.pose.get_pyd_vec();

    status_ = Alive;
    alive_ts_ = _timestamp;
    yaw_round_ = 0;
    ori_yaw_round_ = 0;
    last_yaw_ = new_pyd[1];
    last_ori_yaw_ = new_pyd[1];

    RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "OutpostArmor init, ID not assigned yet");
    
    armor_kf_.init(new_pyd, _timestamp);
    OutpostYawEkf::Vz z;
    z << pc_result_.pose.yaw;
    yaw_kf_.init(z, _timestamp);
}

void OutpostArmor::update(ArmorPoseResult &pc_result, double _timestamp){
    // 更新点坐标
    pc_result_ = pc_result;
    Eigen::Matrix<double, 3, 1> new_pyd = pc_result_.pose.get_pyd_vec();
    // 此处 pc_result_.pose.yaw 已修改为位置方位角(Position Yaw)，而非法向量yaw
    // new_ori_yaw 变量名虽含 ori (orientation)，但现在实际存储的是 Position Yaw
    double new_ori_yaw = pc_result_.pose.yaw;
    // 进行yaw区间过零处理
    if (new_pyd[1] - last_yaw_ < -M_PI * 1.5)
        yaw_round_++;
    else if (new_pyd[1] - last_yaw_ > M_PI * 1.5)
        yaw_round_--;
    if (new_ori_yaw - last_ori_yaw_ < -M_PI * 1.5)
        ori_yaw_round_++;
    else if (new_ori_yaw - last_ori_yaw_ > M_PI * 1.5)
        ori_yaw_round_--;

    last_yaw_ = new_pyd[1];
    new_pyd[1] += yaw_round_ * M_PI * 2;
    last_ori_yaw_ = new_ori_yaw;
    new_ori_yaw += ori_yaw_round_ * M_PI * 2;

    OutpostYawEkf::Vz z;
    z << new_ori_yaw;
    armor_kf_.update(new_pyd, _timestamp);
    yaw_kf_.update(z, _timestamp);
    // 更新时间戳和状态
    alive_ts_ = _timestamp;
    status_ = Alive;
}

void OutpostArmor::zeroCrossing(double datum){
    double yaw = armor_kf_.getX()[1];
    yaw += yaw_round_ * M_PI * 2;
    while (yaw - datum < -M_PI * 1.5) {
        yaw_round_++;
        yaw += M_PI * 2;
    }
    while (yaw - datum > M_PI * 1.5) {
        yaw_round_--;
        yaw -= M_PI * 2;
    }
    armor_kf_.getX()[1] = yaw;  // 利用协方差与均值无关的性质，对SCKF的点进行yaw轴上的平移
}

//---------------------------------------OutpostArmor类---------------------------------------

//---------------------------------------Outpost类---------------------------------------

Outpost::Outpost(ArmorId _id, bool _outpost_kf_init, bool _in_follow, int _armor_cnt)
                : id(_id), outpost_kf_init(_outpost_kf_init), in_follow(_in_follow), armor_cnt(_armor_cnt){
    for (int i = 0; i < 3; ++i) {
        center_pos_filter[i] = MathFilter(5);
        // top_pos_filter[i] = MathFilter(1);
        armor_height_filter[i] = MathFilter(20);  // 初始化高度滤波器
    }
    T_solver.reset();
}

// 更新装甲板高度
void Outpost::update_armor_height(int armor_id, double observed_height){
    if (armor_id < 0 || armor_id >= 3) {
        RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Invalid armor_id %d for height update", armor_id);
        return;
    }
    // 使用MathFilter进行滤波
    armor_height_filter[armor_id].update(observed_height);
    armor_heights_[armor_id] = armor_height_filter[armor_id].get();
    
    RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Updated armor %d height: observed=%.3f, filtered=%.3f", armor_id, observed_height, armor_heights_[armor_id]);
}

// 获取装甲板高度
double Outpost::get_armor_height_by_id(int armor_id){
    if (armor_id < 0 || armor_id >= 3) {
        RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Invalid armor_id: %d, using default height", armor_id);
        // 返回默认高度
        return 1.516;
    }
    // 如果高度映射已初始化，返回滤波后的高度
    if (height_mapping_initialized_) {
        return armor_heights_[armor_id];
    }
    // 其他情况返回默认高度
    return 1.516;
}

Outpost::OutpostPosition Outpost::predict_positions(double _timestamp){

    OutpostPosition result(4);

    // 中心状态
    OutpostCkf::State state_pre = op_ckf.predictState(_timestamp);
    result.center_ = OutpostCkf::get_center(state_pre);
    //若高度映射已建立，用装甲板的平均高度作为中心高度。 中心高度貌似暂时用不上
    if(height_mapping_initialized_){
        double height[3];
        for(int i = 0; i < 3; ++i){
            height[i] = get_armor_height_by_id(i);
        }
        result.center_[2] = (height[0] + height[1] + height[2]) / 3.0;
    }

    //每块装甲板的状态
    for (int i = 0; i < armor_cnt; ++i) {
        double armor_height = result.center_[2];
        if(height_mapping_initialized_){
            // 使用实际装甲板高度
            armor_height = get_armor_height_by_id(i);
        }
        OutpostCkf::Vx tmp_state_vec = state_pre.toVx();
        // yaw 表示 0 号装甲板相位角，其他装甲板按 +i*angle_dis_ 递增
        // 计算预测的位置坐标 (Center -> Armor)
        OutpostCkf::Observe observe_pre(op_ckf.h(Eigen::Ref<const OutpostCkf::Vx>(tmp_state_vec), i, armor_height));
        result.armors_xyz_[i] = Eigen::Vector3d(observe_pre.x, observe_pre.y, observe_pre.z);
        
        // 外部选板逻辑需要装甲板的【法向量朝向】(Normal Yaw)，而非位置方位角(Position Yaw)
        // state.yaw 是 0号装甲板的相位角
        OutpostCkf::State temp_state; temp_state.fromVx(tmp_state_vec);
        result.armor_yaws_[i] = temp_state.yaw + i * op_ckf.angle_dis_;
    }
    return result;
}

void Outpost::reset(const OutpostCkf::Observe &_observe, int _phase_id, int _armor_cnt, double _timestamp, double _z){

    armor_cnt = _armor_cnt;

    outpost_kf_init = true;

    op_ckf.reset(_observe, _phase_id, _armor_cnt, _timestamp, _z);

    now_position_.center_ = OutpostCkf::get_center(op_ckf.state_);

    for (int i = 0; i < armor_cnt; ++i) {
        // 使用实际装甲板高度
        double armor_height = get_armor_height_by_id(i);
        OutpostCkf::Observe observe(op_ckf.h(Eigen::Ref<const OutpostCkf::Vx>(op_ckf.Xe), i, armor_height));
        now_position_.armors_xyz_[i] = Eigen::Vector3d(observe.x, observe.y, observe.z);
        
        // reset 时的状态显示，使用【法向量朝向】
        OutpostCkf::State temp_state; temp_state.fromVx(Eigen::Ref<const OutpostCkf::Vx>(op_ckf.Xe));
        now_position_.armor_yaws_[i] = temp_state.yaw + i * op_ckf.angle_dis_;
    }
    RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Outpost reset: phase=%d, armor_cnt=%d, Center=[%.3f, %.3f, %.3f], Phase0=%.3f", 
        _phase_id, armor_cnt, now_position_.center_[0], now_position_.center_[1], now_position_.center_[2], op_ckf.state_.yaw);
}

void Outpost::update(OutpostCkf::Observe _observe, double _timestamp, int _phase_id){
    // 安全检查
    if (_phase_id < 0 || _phase_id >= armor_cnt) {
        RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Invalid phase_id %d in outpost update", _phase_id);
        return;
    }
    // 修改观测值的高度为实际装甲板高度
    _observe.z = get_armor_height_by_id(_phase_id);
    
    // --------由于CKF不稳定，这里强行修正中心及角速度--------- 若打前哨站时自身不动，可以考虑使用这个逻辑？
    // 当前 非无人机使用此逻辑
    if (!is_aerial && !yaw_increase_history.empty() && !yaw_decrease_history.empty()) {
         // --- 替换 CKF 逻辑 ---
        
        // 取左右边界的中线Yaw
        double yaw_l = yaw_increase_history.front().second;
        double yaw_r = yaw_decrease_history.front().second;
        double yaw_center_direction = angle_middle(yaw_l, yaw_r);
        // 取过中时的Pitch和Dis
        double pitch = common_middle_pitch.get();
        double dis_middle_armor = common_middle_dis.get();
        // 计算中心Dis
        double dis_center = (dis_middle_armor * cos(pitch) + OutpostCkf::const_dis_) / cos(pitch);
        // 计算固定中心坐标XYZ
        Eigen::Vector3d center_pyd(pitch, yaw_center_direction, dis_center);
        Eigen::Vector3d center_xyz = pyd2xyz(center_pyd);
        // 确定固定Omega
        double yaw_spd_val = common_yaw_spd.get();
        double fixed_omega = (yaw_spd_val < 0) ? 2.5 : -2.5;

        // 简易的状态更新 (Geometry + Complementary Filter)
        
        // 根据时间差预测 Yaw
        double dt = _timestamp - op_ckf.last_timestamp_;
        if (op_ckf.last_timestamp_ < 0) dt = 0; // 首次
        double predicted_yaw = op_ckf.state_.yaw + fixed_omega * dt;
        // 根据固定中心和当前装甲板坐标反解 Yaw
        // Vector Center -> Armor
        double dx = _observe.x - center_xyz[0];
        double dy = _observe.y - center_xyz[1];
        // 算出的 yaw 是该块装甲板的 yaw
        double measured_armor_yaw = atan2(dy, dx);
        // 换算回 0 号装甲板的相位 (yaw)
        double measured_base_yaw = measured_armor_yaw - _phase_id * op_ckf.angle_dis_;
        
        // 对预测和观测进行加权融合
        // 观测位置越准，alpha 越大。由于中心固定，位置观测相对可信。
        // 为了解决“旋转忽快忽慢且反向”的问题，必须大幅降低测量权值 alpha，
        // 使滤波器更多地信赖平滑的预测模型 (Fixed Omega)，仅利用观测缓慢修正相位偏差。
        double alpha = 0.01; // 增强平滑性
        double diff = measured_base_yaw - predicted_yaw;
        // Angle Wrap (-PI, PI)
        while (diff > M_PI) diff -= 2 * M_PI;
        while (diff < -M_PI) diff += 2 * M_PI;

        double final_yaw = predicted_yaw + alpha * diff;

        // 写回 State (完全绕过 CKF_update)
        op_ckf.state_.x = center_xyz[0];
        op_ckf.state_.y = center_xyz[1];
        op_ckf.state_.z = center_xyz[2];
        op_ckf.state_.vx = 0; // 固定中心，无速度
        op_ckf.state_.vy = 0;
        op_ckf.state_.yaw = final_yaw;
        op_ckf.state_.omega = fixed_omega;
        
        // 更新 Xe 向量和时间戳，保持一致性
        op_ckf.Xe = op_ckf.state_.toVx();
        op_ckf.last_timestamp_ = _timestamp;
        op_ckf.const_z_ = center_xyz[2];

        RCLCPP_INFO(rclcpp::get_logger("outpostaim"), 
            "Simplified Model: Center=[%.3f, %.3f], Omega=%.1f | GeomYaw=%.3f, PredYaw=%.3f -> FinalYaw=%.3f",
            center_xyz[0], center_xyz[1], fixed_omega, measured_base_yaw, predicted_yaw, final_yaw);

    } else {
        // 其他情况使用原 CKF 进行更新
        op_ckf.CKF_update(_observe, _timestamp, _phase_id);
    }
    // -------------------------------------------------

    now_position_.center_ = OutpostCkf::get_center(op_ckf.state_);

    RCLCPP_INFO(rclcpp::get_logger("outpostaim"), 
        "Outpost Update [ts=%.4f]: Phase=%d | Center=[%.3f, %.3f, %.3f] | Yaw=%.3f, Omega=%.3f",
        _timestamp, _phase_id, 
        now_position_.center_[0], now_position_.center_[1], now_position_.center_[2], 
        op_ckf.state_.yaw, op_ckf.state_.omega);
    
    for (int i = 0; i < armor_cnt; ++i) {
        double armor_height = get_armor_height_by_id(i);
        OutpostCkf::Observe observe(op_ckf.h(Eigen::Ref<const OutpostCkf::Vx>(op_ckf.Xe), i, armor_height));
        
        // 状态更新后显示，使用【法向量朝向】
        OutpostCkf::State temp_state; temp_state.fromVx(Eigen::Ref<const OutpostCkf::Vx>(op_ckf.Xe));
        now_position_.armor_yaws_[i] = temp_state.yaw + i * op_ckf.angle_dis_;

        RCLCPP_INFO(rclcpp::get_logger("outpostaim"), 
            " -> Armor %d: Pos=[%.3f, %.3f, %.3f], Yaw=%.3f", 
            i, now_position_.armors_xyz_[i][0], now_position_.armors_xyz_[i][1], now_position_.armors_xyz_[i][2], now_position_.armor_yaws_[i]);
    }
}

//---------------------------------------Outpost类---------------------------------------

//---------------------------------------OutpostCkf---------------------------------------

OutpostCkf::OutpostCkf(){
    sample_num_ = 2 * state_num;
    samples_ = std::vector<Vx>(sample_num_);
    weights_ = std::vector<double>(sample_num_);
    Pe = Pp = Mxx::Identity();
    sample_X = std::vector<Vx>(sample_num_);
    const_dis_ = 0.2765; // 前哨站半径 
}

void OutpostCkf::reset(const Observe &_observe, int phase_id, int _armor_cnt, double _timestamp, double _z) {
    armor_cnt_ = _armor_cnt;
    angle_dis_ = 2 * M_PI / armor_cnt_;
    //_z是观测到的装甲板高度，用它初始化滤波器
    const_z_ = _z;
    // 使用观测到的位置方位角来粗略初始化状态
    // 注意：_observe.yaw 是 Azimuth, state_.yaw 是 Normal。
    // 两者存在几何偏差，仅作为初值。
    state_ = State(_observe.x - const_dis_ * cos(_observe.yaw + phase_id * angle_dis_), 0.,
                  _observe.y - const_dis_ * sin(_observe.yaw + phase_id * angle_dis_), 0., 
                  const_z_,
                  _observe.yaw, 0.);  // omega 初始化为anti_clock_wise * M_PI * 2 * 0.4 更快，然无法预先判断anti_clock_wise
    Xe = state_.toVx();
    Pe = Pp = config_.init_P.asDiagonal();
    last_timestamp_ = _timestamp;
}

void OutpostCkf::CKF_update(Observe _observe, double _timestamp, int _phase_id){
    double dT = _timestamp - last_timestamp_;
    // // NGXY_INFO("OutpostCKF update: dT=%.4f, phase_id=%d", dT, _phase_id);
    Xe = state_.toVx();
    Vz z = _observe.toVz();
    CKF_predict(dT);
    SRCR_sampling_3(Eigen::Ref<const OutpostCkf::Vx>(Xp), Pp);
    CKF_measure(z, _phase_id, _observe.z);
    CKF_correct(z);
    last_timestamp_ = _timestamp;
}

OutpostCkf::State OutpostCkf::predictState(double _timestamp){
    State ans;
    OutpostCkf::Vx tmp = f(Eigen::Ref<const OutpostCkf::Vx>(Xe), _timestamp - last_timestamp_);
    ans.fromVx(tmp);
    ans.z = const_z_;
    return ans;
}

Eigen::Vector3d OutpostCkf::get_center(State _state){
    return Eigen::Vector3d(_state.x, _state.y, _state.z);
}

Eigen::Vector3d OutpostCkf::get_center(){
    return Eigen::Vector3d(state_.x, state_.y, const_z_);
}

OutpostCkf::Vx OutpostCkf::f(const Eigen::Ref<const OutpostCkf::Vx> &_x, double _dt) const{
    Vx ans = _x;// x vx y vy yaw omega
    ans[0] += _x[1] * _dt;
    ans[2] += _x[3] * _dt;
    ans[4] += _x[5] * _dt;
    return ans;
}

OutpostCkf::Vz OutpostCkf::h(const Eigen::Ref<const OutpostCkf::Vx> &_x, int _phase_id, double armor_height) const{
    State X_state;
    X_state.fromVx(_x);
    Observe ans;
    // 相位角转位置坐标
    ans.x = X_state.x + const_dis_ * cos(X_state.yaw + _phase_id * angle_dis_);
    ans.y = X_state.y + const_dis_ * sin(X_state.yaw + _phase_id * angle_dis_);

    // 这里的 ans.yaw 对应观测向量 z 中的 yaw (即 atan2(y, x))
    // 注意：不要输出法向量 yaw，否则会与输入的观测值含义冲突导致残差计算错误。
    // 先算方位角，用于和测量值(也是方位角)做差
    ans.yaw = atan2(ans.y, ans.x);
    
    ans.z = armor_height; 
    return ans.toVz();
}

void OutpostCkf::SRCR_sampling_3(const Eigen::Ref<const OutpostCkf::Vx> &_x, const Mxx &_P){
    double sqrtn = sqrt(state_num);
    double weight = 1.0 / (2 * state_num);
    Eigen::LLT<Eigen::MatrixXd> get_S(_P);
    Eigen::MatrixXd S = get_S.matrixL();
    for (int i = 0; i < state_num; ++i) {
        samples_[i] = _x + sqrtn * S.col(i);
        weights_[i] = weight;

        samples_[i + state_num] = _x - sqrtn * S.col(i);
        weights_[i + state_num] = weight;
    }
}

void OutpostCkf::calcQ(double _dt){
    static double dTs[4];
    dTs[0] = _dt;
    for (int i = 1; i < 4; ++i) {
        dTs[i] = dTs[i - 1] * _dt;
    }
    double q_x_x = dTs[3] / 4 * config_.Q2_XY, q_x_vx = dTs[2] / 2 * config_.Q2_XY, q_vx_vx = dTs[1] * config_.Q2_XY;
    double q_y_y = dTs[3] / 4 * config_.Q2_YAW, q_y_vy = dTs[2] / 2 * config_.Q2_YAW, q_vy_vy = dTs[1] * config_.Q2_YAW;
    Q = Mxx::Zero();
    Q.block(0, 0, 2, 2) << q_x_x, q_x_vx,  //
        q_x_vx, q_vx_vx;                   //
    Q.block(2, 2, 2, 2) << q_x_x, q_x_vx,  //
        q_x_vx, q_vx_vx;                   //
    Q.block(4, 4, 2, 2) << q_y_y, q_y_vy,  //
        q_y_vy, q_vy_vy;     
}

void OutpostCkf::calcR(Vz _z){
    Vz R_vec;
    R_vec << abs(config_.R_XYZ * _z[0]), abs(config_.R_XYZ * _z[1]), abs(config_.R_XYZ * _z[2]), config_.R_YAW;
    R = R_vec.asDiagonal();
}

void OutpostCkf::CKF_predict(double _dt){
    calcQ(_dt);
    SRCR_sampling_3(Eigen::Ref<const OutpostCkf::Vx>(Xe), Pe);
    Xp = Vx::Zero();
    for (int i = 0; i < sample_num_; ++i) {
        sample_X[i] = f(samples_[i], _dt);
        Xp += weights_[i] * sample_X[i];
    }

    Pp = Mxx::Zero();
    for (int i = 0; i < sample_num_; ++i) {
        Pp += weights_[i] * (sample_X[i] - Xp) * (sample_X[i] - Xp).transpose();
    }
    Pp += Q;
}

void OutpostCkf::CKF_measure(Vz _z, int phase_id, double armor_height){
    sample_Z = std::vector<Vz>(sample_num_);  // 修正
    Zp = Vz::Zero();
    for (int i = 0; i < sample_num_; ++i) {
        sample_Z[i] = h(samples_[i], phase_id, armor_height);
        Zp += weights_[i] * sample_Z[i];
    }

    Pzz = Mzz::Zero();
    for (int i = 0; i < sample_num_; ++i) {
        Vz diff = sample_Z[i] - Zp;
        while (diff[3] > M_PI) diff[3] -= 2 * M_PI;
        while (diff[3] < -M_PI) diff[3] += 2 * M_PI;
        Pzz += weights_[i] * diff * diff.transpose();
    }

    // 根据dis计算自适应R
    calcR(_z);
    Pzz += R;
}

void OutpostCkf::CKF_correct(Vz _z){
    Pxz = Mxz::Zero();
    for (int i = 0; i < sample_num_; ++i) {
        Vz diff_z = sample_Z[i] - Zp;
        while (diff_z[3] > M_PI) diff_z[3] -= 2 * M_PI;
        while (diff_z[3] < -M_PI) diff_z[3] += 2 * M_PI;

        Pxz += weights_[i] * (sample_X[i] - Xp) * diff_z.transpose();
    }
    K = Pxz * Pzz.inverse();

    Vz residual = _z - Zp;
    while (residual[3] > M_PI) residual[3] -= 2 * M_PI;
    while (residual[3] < -M_PI) residual[3] += 2 * M_PI;

    Xe = Xp + K * residual;
    Pe = Pp - K * Pzz * K.transpose();

    state_.fromVx(Eigen::Ref<const OutpostCkf::Vx>(Xe));
    state_.z = const_z_;
}
//---------------------------------------OutpostCkf---------------------------------------