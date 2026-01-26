#include "outpostaim/outpost_estimator.h"
#include "outpostaim/outpost_node.h"
#include <cmath>

Eigen::Vector3d operator*(const Eigen::Isometry3d &T, const Eigen::Vector3d &v) {
    Eigen::Vector4d V(v[0], v[1], v[2], 1);
    return (T * V).block<3, 1>(0, 0);
}

//——————————————————————————————————————Armor状态————————————————————————————————————————

void OutpostArmor::init(const ArmorPoseResult &pc_result, double _timestamp){
    pc_result_ = pc_result;
    // 重置滤波器
    Eigen::Matrix<double, 3, 1> new_pyd = pc_result_.pose.get_pyd_vec();
    // Eigen::Vector3d xyz = pc_result_.pose.get_xyz_vec(); 

    status_ = Alive;
    alive_ts_ = _timestamp;
    yaw_round_ = 0;
    ori_yaw_round_ = 0;
    last_yaw_ = new_pyd[1];
    last_ori_yaw_ = new_pyd[1];

    // phase_in_outpost_ = -1;  // -1表示未分配ID
    
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
    // _new_pb.pose.yaw = yaw_kf_.getX()[0];

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

//------------------------Armor状态----------------------------

//-----------------------Outpost状态---------------------

Outpost::Outpost(ArmorId _id, bool _outpost_kf_init, bool _in_follow, int _armor_cnt)
                : id(_id), outpost_kf_init(_outpost_kf_init), in_follow(_in_follow), armor_cnt(_armor_cnt){
    for (int i = 0; i < 3; ++i) {
        center_pos_filter[i] = MathFilter(1); //？？严查
        top_pos_filter[i] = MathFilter(1);
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

// 是否要判断两块装甲板是否属于同一辆车？但是前哨站只有一个。。。

// void Outpost::add_armor(OutpostArmor armor){
//     static std::vector<int> alive_indexs(armor_cnt);
//     alive_indexs.clear();
//     for (int i = 0; i < (int)armors.size(); ++i) {
//         if (armors[i].status_ == Status::Alive) {
//             alive_indexs.push_back(static_cast<int>(i));
//         }
//     }
//     // 在所有装甲板中寻找tracking armor并更新phase   待修改，直接接收tracker的id就行了吧
//     double nearest_ts = 0;
//     int nearest_id = 0;
//     for (int i = 0; i < (int)armors.size(); ++i) {
//         if (armors[i].alive_ts_ > nearest_ts) {
//             nearest_ts = armors[i].alive_ts_;
//             nearest_id = i;
//         }
//     }
//     if (armors.size() > 0) { // 是否应该删去
//         if (check_left(armor.getPositionPyd(), armors[nearest_id].getPositionPyd())) {
//             armor.phase_in_outpost_ = (armors[nearest_id].phase_in_outpost_ - 1 + armor_cnt) % armor_cnt;
//         } else {
//             armor.phase_in_outpost_ = (armors[nearest_id].phase_in_outpost_ + 1 + armor_cnt) % armor_cnt;
//         }
//     } else {
//         armor.phase_in_outpost_ = 0;
//     }
//     // NGXY_INFO("Assigned ID %d to new armor", armor.phase_in_outpost_);

//     // 没有活动装甲板
//     if (alive_indexs.size() == 0) {
//         armors.clear();
//         armors.push_back(armor);
//     } else if (alive_indexs.size() == 1) {
//         // 有一个原有的装甲板
//         OutpostArmor previous_armor = armors[alive_indexs[0]];
//         // 求解两个装甲板之间的位置坐标距离
//         double armor_dis = calc_surface_dis_xyz(previous_armor.getPositionXyz(), armor.getPositionXyz());

//         if (armor_dis < params.robot_2armor_dis_thresh) {
//             // 成功组成一对装甲板
//             armors.clear();
//             // 保证在左边装甲板位于数组的前位 待修改
//             if (check_left(previous_armor.getPositionPyd(), armor.getPositionPyd())) {
//                 armors.push_back(previous_armor);
//                 armors.push_back(armor);
//             } else {
//                 armors.push_back(armor);
//                 armors.push_back(previous_armor);
//             }
//             // NGXY_INFO("Added armor pair: heights=[%.3f, %.3f], phases=[%d, %d]",
//                     //  armors[0].getPositionXyz()[2], armors[1].getPositionXyz()[2],
//                     //  armors[0].phase_in_outpost_, armors[1].phase_in_outpost_);
//         } else {
//             if (previous_armor.getPositionXyz().norm() > armor.getPositionXyz().norm()) {
//                 armors.clear();
//                 armors.push_back(armor);
//             }
//         }
//     } else if (alive_indexs.size() == 2) {
//         // add_armor_logger.warn("3 armors");
//         // TODO
//         // NGXY_INFO("[Outpost add_armor] 3 armors!");
//         return;
//     } else {
//         // NGXY_ERROR("[Outpost add_armor] impossible armor amount: %d!", armor_cnt);
//         // 异常情况
//         armors.clear();
//         armors.push_back(armor);
//     }
// }

Outpost::OutpostPosition Outpost::predict_positions(double _timestamp){

    OutpostPosition result(4);

    // 安全检查：确保滤波器已初始化
    // if (!outpost_kf_init) {
    //     //// NGXY_WARN("Outpost CKF not initialized, returning default position");
    //     result.center_ = Eigen::Vector3d(0, 0, 1.516);  // 默认中心高度
    //     for (int i = 0; i < armor_cnt; ++i) {
    //         result.armors_xyz_[i] = Eigen::Vector3d(0, 0, get_armor_height_by_id(i));
    //         result.armor_yaws_[i] = 0;
    //     }
    //     return result;
    // }
    //double center_z = op_ckf.state_.z;

    // 中心状态
    OutpostCkf::State state_pre = op_ckf.predictState(_timestamp);
    result.center_ = OutpostCkf::get_center(state_pre);
    //若高度映射已建立，用装甲板的平均高度作为中心高度
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
        OutpostCkf::Observe observe_pre(op_ckf.h(Eigen::Ref<const OutpostCkf::Vx>(tmp_state_vec), i, armor_height));
        result.armors_xyz_[i] = Eigen::Vector3d(observe_pre.x, observe_pre.y, observe_pre.z);
        result.armor_yaws_[i] = observe_pre.yaw + i * op_ckf.angle_dis_;
    }
    return result;
}

void Outpost::reset(const OutpostCkf::Observe &_observe, int _phase_id, int _armor_cnt, double _timestamp, double _z){

    armor_cnt = _armor_cnt;

    // 更新中心高度滤波器
    // const_z_filter.update(_z);
    // double center_z = const_z_filter.get();

    outpost_kf_init = true;

    op_ckf.reset(_observe, _phase_id, _armor_cnt, _timestamp, _z);

    now_position_.center_ = OutpostCkf::get_center(op_ckf.state_);

    for (int i = 0; i < armor_cnt; ++i) {
        // 使用实际装甲板高度
        double armor_height = get_armor_height_by_id(i);
        OutpostCkf::Observe observe(op_ckf.h(Eigen::Ref<const OutpostCkf::Vx>(op_ckf.Xe), i, armor_height));

        now_position_.armors_xyz_[i] = Eigen::Vector3d(observe.x, observe.y, observe.z);
        now_position_.armor_yaws_[i] = observe.yaw + i * op_ckf.angle_dis_;
    }
    RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Outpost reset: phase=%d, armor_cnt=%d", _phase_id, armor_cnt);
}

void Outpost::update(OutpostCkf::Observe _observe, double _timestamp, int _phase_id){
    // 安全检查
    if (_phase_id < 0 || _phase_id >= armor_cnt) {
        RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Invalid phase_id %d in outpost update", _phase_id);
        return;
    }

    // 更新中心高度滤波器
    //const_z_filter.update(_z);
    //double center_z = const_z_filter.get();
    
    // 修改观测值的高度为实际装甲板高度
    _observe.z = get_armor_height_by_id(_phase_id);
    
    op_ckf.CKF_update(_observe, _timestamp, _phase_id);
    
    //op_ckf.state_.z = center_z;

    now_position_.center_ = OutpostCkf::get_center(op_ckf.state_);
    
    for (int i = 0; i < armor_cnt; ++i) {

        double armor_height = get_armor_height_by_id(i);
        OutpostCkf::Observe observe(op_ckf.h(Eigen::Ref<const OutpostCkf::Vx>(op_ckf.Xe), i, armor_height));

        now_position_.armors_xyz_[i] = Eigen::Vector3d(observe.x, observe.y, observe.z);
        now_position_.armor_yaws_[i] = observe.yaw + i * op_ckf.angle_dis_;
    }
}

void Outpost::set_unfollowed(){
    in_follow = false;
}
//-----------------------Outpost状态---------------------

//---------------------OutpostCkf-------------------------

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
    ans.yaw = X_state.yaw;  // id为0的装甲板的yaw

    ans.x = X_state.x + const_dis_ * cos(X_state.yaw + _phase_id * angle_dis_);
    ans.y = X_state.y + const_dis_ * sin(X_state.yaw + _phase_id * angle_dis_);
    ans.z = armor_height;
    Vz result = ans.toVz();
    return result;

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
        Pzz += weights_[i] * (sample_Z[i] - Zp) * (sample_Z[i] - Zp).transpose();
    }

    // 根据dis计算自适应R
    calcR(_z);
    Pzz += R;
}

void OutpostCkf::CKF_correct(Vz _z){
    Pxz = Mxz::Zero();
    for (int i = 0; i < sample_num_; ++i) {
        Pxz += weights_[i] * (sample_X[i] - Xp) * (sample_Z[i] - Zp).transpose();
    }
    K = Pxz * Pzz.inverse();

    Xe = Xp + K * (_z - Zp);
    Pe = Pp - K * Pzz * K.transpose();

    state_.fromVx(Eigen::Ref<const OutpostCkf::Vx>(Xe));
    state_.z = const_z_;
}
//--------------------------OutpostCkf------------------------------------------