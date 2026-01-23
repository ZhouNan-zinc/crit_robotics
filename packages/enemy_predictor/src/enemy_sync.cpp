#include "enemy_predictor/enemy_predictor_node.h"
#include <algorithm>
#include <cmath>

bool EnemyPredictorNode::getCurrentYaw(const rclcpp::Time & target_time, double & output_yaw) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    if (imu_buffer_.size() < 2) {
        return false;
    }

    // 寻找时间戳刚好早于目标时间的数据
    auto it_after = std::lower_bound(
        imu_buffer_.begin(), imu_buffer_.end(), target_time,
        [](const ImuData & data, const rclcpp::Time & time) {
            return data.timestamp < time;
        });

    // 边界情况处理
    if (it_after == imu_buffer_.begin()) {
        // 目标时间早于所有缓存数据
        if (std::fabs((it_after->timestamp - target_time).seconds()) < 0.01) {
            output_yaw = it_after->current_yaw;
            return true;
        }
        return false;
    }

    if (it_after == imu_buffer_.end()) {
        // 目标时间晚于所有缓存数据，使用最后一帧
        auto it_before = imu_buffer_.end() - 1;
        if ((target_time - it_before->timestamp).seconds() < 0.01) {
            output_yaw = it_before->current_yaw;
            return true;
        }
        return false;
    }
    // 找到目标时间前后的两帧数据
    auto it_before = it_after - 1;
    rclcpp::Time t1 = it_before->timestamp;
    rclcpp::Time t2 = it_after->timestamp;
    double yaw1 = it_before->current_yaw;
    double yaw2 = it_after->current_yaw;

    // 处理角度环绕问题
    // 如果两个yaw角差值大于π，则调整其中一个使其成为最短路径
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

    return true;
}

void EnemyPredictorNode::cleanOldImuData() {
    if (imu_buffer_.empty()) return;

    rclcpp::Time now = this->now();
    rclcpp::Time cutoff_time = now - rclcpp::Duration::from_seconds(buffer_duration_);

    // 移除所有早于截止时间的数据，但保留至少一帧（如果数据很旧也保留一帧用于边界情况）
    auto it = std::lower_bound(
        imu_buffer_.begin(), imu_buffer_.end(), cutoff_time,
        [](const ImuData & data, const rclcpp::Time & time) {
            return data.timestamp < time;
        });

    if (it != imu_buffer_.begin()) {
        // 保留一帧数据，确保如果新数据是第一个，我们还有前一帧可以插值
        size_t keep_from = (it - imu_buffer_.begin() > 1) ? (it - imu_buffer_.begin() - 1) : 0;
        imu_buffer_.erase(imu_buffer_.begin(), imu_buffer_.begin() + keep_from);
    }
}