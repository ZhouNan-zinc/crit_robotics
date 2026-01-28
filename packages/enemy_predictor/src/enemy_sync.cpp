#include "enemy_predictor/enemy_predictor_node.h"
#include <algorithm>
#include <cmath>

double EnemyPredictorNode::getCurrentYaw(const rclcpp::Time & target_time) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    // 默认返回 0.0
    double output_yaw = 0.0;
    
    //RCLCPP_INFO(this->get_logger(), "getCurrentYaw called: target=%.6f, buffer_size=%zu", 
    //            target_time.seconds(), imu_buffer_.size());

    if (imu_buffer_.empty()) {
        RCLCPP_INFO(this->get_logger(), "IMU buffer is empty, returning 0.0");
        return 0.0;
    }

    // 即使只有一帧数据也尝试使用，扩大时间容差
    if (imu_buffer_.size() == 1) {
        // 放宽到0.2秒容差
        if (std::fabs((imu_buffer_.front().timestamp - target_time).seconds()) < 0.2) {
            output_yaw = imu_buffer_.front().current_yaw;
            RCLCPP_INFO(this->get_logger(), "Using single frame, yaw=%.3f, diff=%.3f", 
                        output_yaw, std::fabs((imu_buffer_.front().timestamp - target_time).seconds()));
            return output_yaw;
        }
        RCLCPP_INFO(this->get_logger(), "Single frame but diff too large: %.3f, returning 0.0", 
                    std::fabs((imu_buffer_.front().timestamp - target_time).seconds()));
        return 0.0;
    }

    const double TIME_TOLERANCE = 0.2;  // 从0.01改为0.2秒

    // 寻找时间戳刚好早于目标时间的数据
    auto it_after = std::lower_bound(
        imu_buffer_.begin(), imu_buffer_.end(), target_time,
        [](const ImuData & data, const rclcpp::Time & time) {
            return data.timestamp < time;
        });

    // 边界情况处理
    if (it_after == imu_buffer_.begin()) {
        // 目标时间早于所有缓存数据
        double time_diff = std::fabs((it_after->timestamp - target_time).seconds());
        if (time_diff < TIME_TOLERANCE) {
            output_yaw = it_after->current_yaw;
            RCLCPP_INFO(this->get_logger(), "Using first frame, yaw=%.3f, diff=%.3f", 
                        output_yaw, time_diff);
            return output_yaw;
        }
        RCLCPP_INFO(this->get_logger(), "Target time too early: diff=%.3f > %.3f, returning 0.0", 
                    time_diff, TIME_TOLERANCE);
        return 0.0;
    }

    if (it_after == imu_buffer_.end()) {
        // 目标时间晚于所有缓存数据，使用最后一帧
        auto it_before = imu_buffer_.end() - 1;
        double time_diff = (target_time).seconds() - (it_before->timestamp).seconds();
        if (time_diff < TIME_TOLERANCE) {
            output_yaw = it_before->current_yaw;
            RCLCPP_INFO(this->get_logger(), "Using last frame, yaw=%.3f, diff=%.3f", 
                        output_yaw, time_diff);
            return output_yaw;
        }
        RCLCPP_INFO(this->get_logger(), "Target time too late: diff=%.3f > %.3f, returning 0.0", 
                    time_diff, TIME_TOLERANCE);
        return 0.0;
    }

    // 找到目标时间前后的两帧数据
    auto it_before = it_after - 1;
    rclcpp::Time t1 = it_before->timestamp;
    rclcpp::Time t2 = it_after->timestamp;
    double yaw1 = it_before->current_yaw;
    double yaw2 = it_after->current_yaw;

    //RCLCPP_INFO(this->get_logger(), "Found frames: t1=%.6f, t2=%.6f, target=%.6f", 
    //            t1.seconds(), t2.seconds(), target_time.seconds());

    // 处理角度环绕问题
    double diff = yaw2 - yaw1;
    if (diff > M_PI) {
        yaw2 -= 2.0 * M_PI;
    } else if (diff < -M_PI) {
        yaw2 += 2.0 * M_PI;
    }

    // 计算插值比例 (0~1)
    double ratio = (target_time - t1).seconds() / (t2 - t1).seconds();
    ratio = std::clamp(ratio, 0.0, 1.0);

    // 线性插值
    double interpolated = yaw1 + ratio * (yaw2 - yaw1);

    // 将结果归一化到[-π, π]
    output_yaw = normalize_angle(interpolated);

    //RCLCPP_INFO(this->get_logger(), "Interpolated yaw=%.3f rad (%.1f deg)", 
    //            output_yaw, output_yaw * 180.0 / M_PI);

    return output_yaw;
}
void EnemyPredictorNode::cleanOldImuData() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    // ========== 关键修改: 减少清理强度 ==========
    // 只有当缓冲区非常大时才清理
    if (imu_buffer_.size() < 1000) {  // 从500提高到1000
        return;
    }

    rclcpp::Time now = this->now();
    rclcpp::Time cutoff_time = now - rclcpp::Duration::from_seconds(buffer_duration_);

    size_t old_size = imu_buffer_.size();

    // 找到第一个晚于截止时间的数据
    auto it = std::lower_bound(
        imu_buffer_.begin(), imu_buffer_.end(), cutoff_time,
        [](const ImuData & data, const rclcpp::Time & time) {
            return data.timestamp < time;
        });

    // ========== 关键修改: 保留更多数据 ==========
    // 保留至少500帧数据（原来200帧）
    size_t min_keep_frames = 500;

    if (it - imu_buffer_.begin() > (int)min_keep_frames) {
        // 只清理超出min_keep_frames的部分
        size_t erase_count = (it - imu_buffer_.begin()) - min_keep_frames;
        if (erase_count > 0) {
            imu_buffer_.erase(imu_buffer_.begin(), 
                             imu_buffer_.begin() + erase_count);

            size_t new_size = imu_buffer_.size();
        }
    }
    // 如果it在min_keep_frames内，什么都不做，保留所有数据
}