#include <enemy_trajectoryControl.h>

/*
// ==================== State 结构体方法实现 ====================
Eigen::Vector3d YawTrajectoryPlanner::State::toVector() const {
    return Eigen::Vector3d(position, velocity, acceleration);
}

void YawTrajectoryPlanner::State::fromVector(const Eigen::Vector3d& vec) {
    position = vec(0);
    velocity = vec(1);
    acceleration = vec(2);
}
void YawTrajectoryPlanner::State::reset() {
    position = 0.0;
    velocity = 0.0;
    acceleration = 0.0;
}

// ==================== 构造函数和公共方法实现 ====================
YawTrajectoryPlanner::YawTrajectoryPlanner(const Config& config)
    : config_(config), 
      is_active_(false),
      start_time_(std::chrono::steady_clock::now()),
      last_update_time_(std::chrono::steady_clock::now()),
      last_motor_send_time_(std::chrono::steady_clock::now()),
      last_debug_time_(std::chrono::steady_clock::now()) {
    
    reset();
}

bool YawTrajectoryPlanner::setTargetYaw(double target_yaw_abs, double current_yaw_abs) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 计算最短路径的角度差（处理角度环绕）
    double delta_yaw = normalizeAngle(target_yaw_abs - current_yaw_abs);
    
    // 如果已经在容差范围内，不需要规划
    if (std::abs(delta_yaw) < config_.tolerance) {
        if (is_active_) {
            if (config_.enable_debug) {
                debugPrint("目标已在容差范围内，停止规划");
            }
            is_active_ = false;
        }
        return false;
    }
    
    // 2. 保存起点和目标点
    start_yaw_abs_ = current_yaw_abs;
    target_yaw_abs_ = target_yaw_abs;
    target_delta_yaw_ = delta_yaw;
    
    // 3. 准备当前状态（相对坐标系）
    current_state_abs_.position = current_yaw_abs;
    current_state_rel_.reset();
    current_state_rel_.position = 0.0;  // 相对位置起点为0
    current_state_rel_.velocity = 0.0;  // 假设初始相对速度为0
    current_state_rel_.acceleration = 0.0;
    
    // 4. 计算轨迹参数
    calculateTrajectoryParams();
    
    // 5. 开始规划
    is_active_ = true;
    start_time_ = std::chrono::steady_clock::now();
    last_update_time_ = start_time_;
    last_motor_send_time_ = start_time_;
    
    if (config_.enable_debug) {
        auto now = std::chrono::steady_clock::now();
        last_debug_time_ = now;
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3)
            << "新目标设置: "
            << "当前=" << current_yaw_abs*57.3 << "°, "
            << "目标=" << target_yaw_abs_*57.3 << "°, "
            << "差值=" << delta_yaw*57.3 << "°, "
            << "方向=" << (target_direction_ > 0 ? "正向" : "反向") << ", "
            << "总时间=" << total_time_ << "s, "
            << "最大速度=" << max_velocity_reached_*57.3 << "°/s";
        debugPrint(oss.str());
    }
    
    return true;
}

YawTrajectoryPlanner::ControlOutput YawTrajectoryPlanner::getMotorTarget(double current_yaw_abs) {
    return getMotorTarget(current_yaw_abs, std::chrono::steady_clock::now());
}

YawTrajectoryPlanner::ControlOutput YawTrajectoryPlanner::getMotorTarget(
    double current_yaw_abs, 
    std::chrono::steady_clock::time_point current_time) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    ControlOutput output;
    
    // 如果没有激活的规划，返回当前yaw
    if (!is_active_) {
        output.target_position_abs = current_yaw_abs;
        output.is_finished = true;
        output.is_active = false;
        output.progress = 1.0;
        output.time_remaining = 0.0;
        output.planned_position = current_yaw_abs;
        output.planned_velocity = 0.0;
        output.planned_acceleration = 0.0;
        return output;
    }
    
    // 更新当前绝对yaw
    current_state_abs_.position = current_yaw_abs;
    
    // 计算从规划开始到现在的时间
    double elapsed_time = std::chrono::duration<double>(current_time - start_time_).count();
    
    // 计算应该发送给电控的时间点（考虑系统延迟）
    // 电控会在 lookahead_time 后执行这个指令
    double motor_execution_time = elapsed_time + config_.lookahead_time;
    
    // 获取在该时间点的规划状态
    State planned_state_rel = getPlannedState(motor_execution_time);
    
    // 计算绝对目标位置 = 起点 + 相对位移
    double planned_abs_position = normalizeAngle(start_yaw_abs_ + planned_state_rel.position);
    
    // 插值平滑（避免跳跃）
    double target_for_motor = planned_abs_position;
    
    if (config_.interpolation_factor > 0) {
        // 获取当前时刻的规划位置
        State current_planned_state = getPlannedState(elapsed_time);
        double current_planned_position = normalizeAngle(start_yaw_abs_ + current_planned_state.position);
        
        // 在当前位置和计划位置之间插值
        target_for_motor = interpolatePosition(
            current_yaw_abs,
            planned_abs_position,
            current_planned_state.velocity,
            planned_state_rel.velocity,
            config_.interpolation_factor
        );
    }
    
    // 7. 检查是否完成
    bool is_finished = (elapsed_time >= total_time_) || 
                      (std::abs(normalizeAngle(target_yaw_abs_ - current_yaw_abs)) < config_.tolerance);
    
    if (is_finished) {
        if (config_.enable_debug) {
            debugPrint("规划完成");
        }
        is_active_ = false;
        output.target_position_abs = target_yaw_abs_;  // 直接给最终目标
        output.is_finished = true;
        output.is_active = false;
        output.progress = 1.0;
        output.time_remaining = 0.0;
        output.planned_position = target_yaw_abs_;
        output.planned_velocity = 0.0;
        output.planned_acceleration = 0.0;
    } else {
        // 检查是否需要进行小更新（避免频繁发送微小变化）
        static double last_sent_position = target_for_motor;
        double position_change = std::abs(normalizeAngle(target_for_motor - last_sent_position));
        
        if (position_change > config_.min_update_threshold) {
            output.target_position_abs = target_for_motor;
            last_sent_position = target_for_motor;
        } else {
            // 变化太小，保持上次发送的位置
            output.target_position_abs = last_sent_position;
        }
        
        output.is_finished = false;
        output.is_active = true;
        output.progress = std::min(1.0, elapsed_time / total_time_);
        output.time_remaining = total_time_ - elapsed_time;
        output.planned_position = planned_abs_position;
        output.planned_velocity = planned_state_rel.velocity;
        output.planned_acceleration = planned_state_rel.acceleration;
        
        // 更新最后发送时间
        last_motor_send_time_ = current_time;
        
        // 调试输出
        if (config_.enable_debug) {
            debugPrintStatus(current_yaw_abs, output);
        }
    }
    
    last_update_time_ = current_time;
    return output;
}

YawTrajectoryPlanner::PlannerStatus YawTrajectoryPlanner::getStatus(double current_yaw_abs) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    PlannerStatus status;
    status.is_active = is_active_;
    status.start_yaw = start_yaw_abs_;
    status.target_yaw = target_yaw_abs_;
    status.current_yaw = current_yaw_abs;
    
    if (is_active_) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time_).count();
        status.progress = std::min(1.0, elapsed / total_time_);
        status.time_remaining = total_time_ - elapsed;
        status.max_velocity = max_velocity_reached_;
        status.total_time = total_time_;
        
        // 计算下一个要发送给电控的目标
        double motor_time = elapsed + config_.lookahead_time;
        State motor_state = getPlannedState(motor_time);
        status.next_target = normalizeAngle(start_yaw_abs_ + motor_state.position);
    } else {
        status.progress = 1.0;
        status.time_remaining = 0.0;
        status.max_velocity = 0.0;
        status.total_time = 0.0;
        status.next_target = current_yaw_abs;
    }
    
    return status;
}

void YawTrajectoryPlanner::emergencyStop() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_active_ = false;
    if (config_.enable_debug) {
        debugPrint("紧急停止");
    }
}

void YawTrajectoryPlanner::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_active_ = false;
    start_yaw_abs_ = 0.0;
    target_yaw_abs_ = 0.0;
    target_delta_yaw_ = 0.0;
    current_state_abs_.reset();
    current_state_rel_.reset();
    target_direction_ = 1.0;
    total_time_ = 0.0;
}

void YawTrajectoryPlanner::updateConfig(const Config& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

void YawTrajectoryPlanner::setMotorControlPeriod(double period) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.motor_control_period = period;
    // 可以根据控制周期调整其他参数
    config_.lookahead_time = std::max(period * 2.0, 0.005);  // 至少2个控制周期
}

// ==================== 私有方法实现 ====================
void YawTrajectoryPlanner::calculateTrajectoryParams() {
    // 使用梯形速度剖面
    distance_ = std::abs(target_delta_yaw_);
    target_direction_ = (target_delta_yaw_ > 0) ? 1.0 : -1.0;
    
    double max_vel = config_.max_velocity;
    double max_accel = config_.max_acceleration;
    double v0 = current_state_rel_.velocity;  // 初始相对速度
    
    // 计算加速段和减速段
    double t_acc = (max_vel - std::abs(v0)) / max_accel;
    double s_acc = v0 * t_acc + 0.5 * max_accel * t_acc * t_acc;
    
    double t_dec = max_vel / max_accel;
    double s_dec = 0.5 * max_accel * t_dec * t_dec;
    
    double total_acc_dec_distance = s_acc + s_dec;
    
    if (distance_ <= total_acc_dec_distance) {
        // 三角形速度剖面（达不到最大速度）
        if (std::abs(v0) < 1e-6) {
            // 初始速度为0
            t_acc = std::sqrt(distance_ / max_accel);
            t_dec = t_acc;
        } else {
            // 需要解方程
            // 使用数值解法
            t_acc = solveTriangleTime(v0, max_accel, distance_);
            t_dec = (std::abs(v0) + max_accel * t_acc) / max_accel;
        }
        
        max_velocity_reached_ = std::abs(v0) + max_accel * t_acc;
        acc_time_ = t_acc;
        const_time_ = 0.0;
        dec_time_ = t_dec;
    } else {
        // 梯形速度剖面
        max_velocity_reached_ = max_vel;
        acc_time_ = t_acc;
        const_time_ = (distance_ - total_acc_dec_distance) / max_vel;
        dec_time_ = t_dec;
    }
    
    total_time_ = acc_time_ + const_time_ + dec_time_;
    
    if (config_.enable_debug) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3)
            << "轨迹参数: "
            << "加速=" << acc_time_ << "s, "
            << "匀速=" << const_time_ << "s, "
            << "减速=" << dec_time_ << "s, "
            << "总时间=" << total_time_ << "s, "
            << "最大速度=" << max_velocity_reached_*57.3 << "°/s";
        debugPrint(oss.str());
    }
}

double YawTrajectoryPlanner::solveTriangleTime(double v0, double a, double s) {
    // 解方程: v0*t + 0.5*a*t² + 0.5*a*((v0 + a*t)/a)² = s
    
    // 使用数值解法（二分法）
    double t_low = 0.0;
    double t_high = 2.0 * std::sqrt(s / a);  // 上界
    
    for (int i = 0; i < 20; i++) {
        double t_mid = (t_low + t_high) / 2.0;
        double s_mid = v0 * t_mid + 0.5 * a * t_mid * t_mid + 
                      0.5 * std::pow(v0 + a * t_mid, 2) / a;
        
        if (s_mid > s) {
            t_high = t_mid;
        } else {
            t_low = t_mid;
        }
    }
    
    return (t_low + t_high) / 2.0;
}

YawTrajectoryPlanner::State YawTrajectoryPlanner::getPlannedState(double t) const {
    State state;
    state.position = 0.0;
    state.velocity = current_state_rel_.velocity;
    state.acceleration = 0.0;
    
    if (t <= acc_time_) {
        // 加速段
        double a = config_.max_acceleration * target_direction_;
        state.position = current_state_rel_.position + 
                       state.velocity * t + 0.5 * a * t * t;
        state.velocity += a * t;
        state.acceleration = a;
    } else if (t <= acc_time_ + const_time_) {
        // 匀速段
        double t_acc = acc_time_;
        double a = config_.max_acceleration * target_direction_;
        double v_max = current_state_rel_.velocity + a * t_acc;
        
        double s_acc = current_state_rel_.position + 
                      current_state_rel_.velocity * t_acc + 0.5 * a * t_acc * t_acc;
        
        state.position = s_acc + v_max * (t - t_acc);
        state.velocity = v_max;
        state.acceleration = 0.0;
    } else if (t <= total_time_) {
        // 减速段
        double t_acc = acc_time_;
        double t_const = const_time_;
        double a_acc = config_.max_acceleration * target_direction_;
        double v_max = current_state_rel_.velocity + a_acc * t_acc;
        double a_dec = -config_.max_acceleration * target_direction_;
        
        double s_acc = current_state_rel_.position + 
                      current_state_rel_.velocity * t_acc + 0.5 * a_acc * t_acc * t_acc;
        double s_const = v_max * t_const;
        
        double t_dec = t - t_acc - t_const;
        state.position = s_acc + s_const + 
                       v_max * t_dec + 0.5 * a_dec * t_dec * t_dec;
        state.velocity = v_max + a_dec * t_dec;
        state.acceleration = a_dec;
    } else {
        // 已到达终点
        state.position = target_delta_yaw_;
        state.velocity = 0.0;
        state.acceleration = 0.0;
    }
    
    return state;
}

double YawTrajectoryPlanner::normalizeAngle(double angle) const{
    angle = std::fmod(angle + M_PI, 2.0 * M_PI);
    if (angle < 0) angle += 2.0 * M_PI;
    return angle - M_PI;
}

double YawTrajectoryPlanner::interpolatePosition(
    double current_pos, double planned_pos, 
    double current_vel, double planned_vel,
    double alpha) const {
    
    // 简单的线性插值
    // alpha = 0: 完全用当前位置
    // alpha = 1: 完全用规划位置
    // 实际可以使用更复杂的插值方法,有没有用我也并不知道
    
    // 计算角度差（考虑环绕）
    double delta = normalizeAngle(planned_pos - current_pos);
    
    // 插值
    double interpolated_pos = current_pos + delta * alpha;
    
    // 归一化
    return normalizeAngle(interpolated_pos);
}

// ==================== 调试方法实现 ====================
void YawTrajectoryPlanner::debugPrint(const std::string& message) const {
    if (config_.enable_debug) {
        std::cout << "[YawPlanner] " << message << std::endl;
    }
}

void YawTrajectoryPlanner::debugPrintStatus(double current_yaw_abs, const ControlOutput& output) const {
    auto now_time = std::chrono::steady_clock::now();
    double debug_elapsed = std::chrono::duration<double>(now_time - last_debug_time_).count();
    
    if (debug_elapsed >= config_.debug_interval) {
        const_cast<YawTrajectoryPlanner*>(this)->last_debug_time_ = now_time;
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3)
            << "进度: " << output.progress* 100 << "% | "
            << "当前: " << current_yaw_abs* 57.3 << "° | "
            << "电控目标: " << output.target_position_abs* 57.3 << "° | "
            << "规划位置: " << output.planned_position* 57.3 << "° | "
            << "速度: " << output.planned_velocity* 57.3 << "°/s | "
            << "剩余: " << output.time_remaining << "s";
        debugPrint(oss.str());
    }
}*/