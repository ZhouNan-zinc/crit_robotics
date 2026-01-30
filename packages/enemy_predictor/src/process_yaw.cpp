/*#include "enemy_predictor/enemy_predictor_node.h"
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>


static const Eigen::Matrix3d R_fixed = []() {
    const double pitch_rad = 15.0 * M_PI / 180.0;
    const double cp = std::cos(pitch_rad);
    const double sp = std::sin(pitch_rad);
    
    Eigen::Matrix3d R;
    R << cp, 0, sp,
         0, 1, 0,
         -sp, 0, cp;
    return R;
}();

// 黄金分割比例 (sqrt(5)-1)/2
static const double PHI = 0.6180339887498949;

double computeErrorForYaw(
    double yaw,
    const std::vector<Eigen::Vector3d>& armor_3d_points,
    const std::vector<Eigen::Vector2d>& detected_2d_points,
    const Eigen::Vector3d& tvec,
    const Eigen::Matrix3d& camera_matrix
) {
    double total_error = 0.0;
    
    // 预计算yaw的sin/cos
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);
    
    // yaw旋转矩阵 R_z(yaw)
    Eigen::Matrix3d R_z;
    R_z << cy, -sy, 0,
           sy,  cy, 0,
           0,   0,  1;
    
    // 完整旋转矩阵：R = R_z(yaw) * R_fixed
    Eigen::Matrix3d R = R_z * R_fixed;
    
    // 预计算内参分量
    const double fx = camera_matrix(0, 0);
    const double fy = camera_matrix(1, 1);
    const double cx = camera_matrix(0, 2);
    const double cy_in = camera_matrix(1, 2);
    
    // 计算4个角点的重投影误差
    for (int i = 0; i < 4; ++i) {
        // 从装甲板坐标系变换到相机坐标系
        Eigen::Vector3d point_camera = R * armor_3d_points[i] + tvec;
        
        // 检查深度是否有效
        if (std::abs(point_camera[2]) < 1e-6) {
            return 1e9; // 返回一个大值表示无效
        }
        
        // 投影到图像平面
        double u_proj = fx * point_camera[0] / point_camera[2] + cx;
        double v_proj = fy * point_camera[1] / point_camera[2] + cy_in;
        
        // 计算误差
        double du = u_proj - detected_2d_points[i](0);
        double dv = v_proj - detected_2d_points[i](1);
        total_error += du * du + dv * dv;
    }
    
    return total_error;
}
double optimizeYawByPhiSearch(
    double initial_yaw,
    const std::vector<Eigen::Vector3d>& armor_3d_points,
    const std::vector<Eigen::Vector2d>& detected_2d_points,
    const Eigen::Vector3d& tvec,
    const Eigen::Matrix3d& camera_matrix,
    int max_iter = 12
) {
    // 搜索范围：初始yaw ± 30°
    const double search_range = M_PI / 6.0;
    double a = initial_yaw - search_range;
    double b = initial_yaw + search_range;
    
    // 确保在[-π, π]范围内
    a = std::fmod(a + M_PI, 2 * M_PI) - M_PI;
    b = std::fmod(b + M_PI, 2 * M_PI) - M_PI;
    if (a > b) std::swap(a, b);
    
    // 计算初始内点
    double c = b - PHI * (b - a);
    double d = a + PHI * (b - a);
    
    double fc = computeErrorForYaw(c, armor_3d_points, detected_2d_points, tvec, camera_matrix);
    double fd = computeErrorForYaw(d, armor_3d_points, detected_2d_points, tvec, camera_matrix);

    // φ优选法迭代
    for (int iter = 0; iter < max_iter; ++iter) {
        // 收敛检查
        if (std::abs(b - a) < 1e-6) {
            break;
        }
        
        if (fc < fd) {
            // 最小值在[a, d]区间
            b = d;
            d = c;
            fd = fc;
            c = b - PHI * (b - a);
            fc = computeErrorForYaw(c, armor_3d_points, detected_2d_points, tvec, camera_matrix);
        } else {
            // 最小值在[c, b]区间
            a = c;
            c = d;
            fc = fd;
            d = a + PHI * (b - a);
            fd = computeErrorForYaw(d, armor_3d_points, detected_2d_points, tvec, camera_matrix);
        }
    }
    
    // 返回最优yaw
    double optimal_yaw = (a + b) / 2.0;
    
    // 归一化到[-π, π]
    optimal_yaw = std::fmod(optimal_yaw + M_PI, 2 * M_PI) - M_PI;
    
    return optimal_yaw;
}
double ProcessYaw(
    double pnp_yaw,
    const std::vector<cv::Point3f>& armor_3d_points_cv,
    const std::vector<cv::Point2f>& detected_corners_cv,
    const cv::Mat& tvec_cv,
    const cv::Mat& camera_matrix_cv
) {
    // 参数检查
    if (armor_3d_points_cv.size() != 4 || detected_corners_cv.size() != 4) {
        RCLCPP_ERROR(this->get_logger(),"Yaw优化需要4个角点,实际得到,3D点=%zu, 2D点=%zu", armor_3d_points_cv.size(), detected_corners_cv.size());
        return pnp_yaw; // 返回原始值
    }
    
    // 转换为Eigen格式
    std::vector<Eigen::Vector3d> armor_3d_points(4);
    std::vector<Eigen::Vector2d> detected_2d_points(4);
    Eigen::Vector3d tvec;
    Eigen::Matrix3d camera_matrix;
    
    // 转换装甲板3D角点
    for (int i = 0; i < 4; ++i) {
        armor_3d_points[i] = Eigen::Vector3d(
            armor_3d_points_cv[i].x,
            armor_3d_points_cv[i].y,
            armor_3d_points_cv[i].z
        );
    }
    // 转换检测到的2D角点
    for (int i = 0; i < 4; ++i) {
        detected_2d_points[i] = Eigen::Vector2d(
            detected_corners_cv[i].x,
            detected_corners_cv[i].y
        );
    }
    
    // 转换tvec
    tvec = Eigen::Vector3d(
        tvec_cv.at<double>(0, 0),
        tvec_cv.at<double>(1, 0),
        tvec_cv.at<double>(2, 0)
    );
    
    // 转换相机内参矩阵
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            camera_matrix(i, j) = camera_matrix_cv.at<double>(i, j);
        }
    }
     // 优化yaw角
    double optimized_yaw = optimizeYawByPhiSearch(
        pnp_yaw,
        armor_3d_points,
        detected_2d_points,
        tvec,
        camera_matrix,
        12  // 迭代12次
    );
    
    return optimized_yaw;
}*/