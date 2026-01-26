#ifndef _MATH_UTILS_H
#define _MATH_UTILS_H

#include <Eigen/Dense>
#include "ballistic.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <random>


inline double get_disAngle(double ag1, double ag2) {
    double diff = fmod(ag1 - ag2, M_PI * 2);
    if (diff > M_PI) {
        return diff - 2 * M_PI;
    } else if (diff < -M_PI) {
        return diff + 2 * M_PI;
    } else {
        return diff;
    }
}

// 计算三维点距离
inline double get_dis3d(Eigen::Vector3d A, Eigen::Vector3d B = Eigen::Vector3d::Zero()) {
    return sqrt((A[0] - B[0]) * (A[0] - B[0]) + (A[1] - B[1]) * (A[1] - B[1]) +
                (A[2] - B[2]) * (A[2] - B[2]));
}
// 计算二维点距离
inline double get_dis2d(cv::Point2d a, cv::Point2d b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// 叉乘 计算三角形面积的两倍
inline double cross(cv::Point2d a, cv::Point2d b) { return a.x * b.y - a.y * b.x; }
inline double get_S_triangle(cv::Point2d a, cv::Point2d b, cv::Point2d c) {
    return fabs(cross(b - a, c - a)) / 2;
}
inline double get_area_armor(cv::Point2f pts[5]) {
    return get_S_triangle(pts[0], pts[1], pts[2]) + get_S_triangle(pts[3], pts[0], pts[2]);
}

/// 计算三维向量之间的夹角
inline double calc_diff_angle_xyz(const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
    return fabs(acos(a.dot(b) / a.norm() / b.norm()));
}
/// 计算球面距离（减少distance误差影响）
inline double calc_surface_dis_xyz(const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
    return calc_diff_angle_xyz(a, b) * (a.norm() + b.norm()) / 2;
}

inline bool check_left(const Eigen::Vector3d &pyd_left, const Eigen::Vector3d &pyd_right) {
    double yaw_l = pyd_left[1];
    double yaw_r = pyd_right[1];
    double yaw_dis = fmod(yaw_l - yaw_r, M_PI * 2);
    if (fabs(yaw_dis) > M_PI)
        return yaw_dis < 0;
    else
        return yaw_dis > 0;
}

inline Eigen::Vector3d xyz2pyd(const Eigen::Vector3d &xyz) {
    Eigen::Vector3d pyd;
    // pitch向上为负
    pyd[0] = -atan2(xyz[2], sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]));
    pyd[1] = atan2(xyz[1], xyz[0]);
    pyd[2] = sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);
    return pyd;
}

inline Eigen::Vector3d pyd2xyz(const Eigen::Vector3d &pyd) {
    Eigen::Vector3d xyz;
    xyz[2] = -pyd[2] * sin(pyd[0]);
    double tmplen = pyd[2] * cos(pyd[0]);
    xyz[1] = tmplen * sin(pyd[1]);
    xyz[0] = tmplen * cos(pyd[1]);
    return xyz;
}

inline Eigen::Vector3d xyz2dyz(const Eigen::Vector3d &xyz) {
    Eigen::Vector3d dyz;
    dyz[0] = sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);
    dyz[1] = atan2(xyz[1], xyz[0]);
    dyz[2] = xyz[2];
    return dyz;
}

inline Eigen::Vector3d dyz2xyz(const Eigen::Vector3d &dyz) {
    Eigen::Vector3d xyz;
    xyz[0] = dyz[0] * cos(dyz[1]);
    xyz[1] = dyz[0] * sin(dyz[1]);
    xyz[2] = dyz[2];
    return xyz;
}

inline double rad2deg(double rad) { return rad * 180.0 / M_PI; }

inline double deg2rad(double deg) { return deg * M_PI / 180.0; }

// 转换x至[0, 2*pi]
inline double angle_normalize(double x) {
    x -= int(x / (M_PI * 2.0)) * M_PI * 2.0;
    if (x < 0) x += M_PI * 2.0;
    return x;
}

inline void angle_serialize(double &x, double &y) {
    x = angle_normalize(x);
    y = angle_normalize(y);
    if (fabs(x - y) > M_PI) {
        if (x < y)
            x += M_PI * 2.0;
        else
            y += M_PI * 2.0;
    }
}

inline double angle_middle(double x, double y) {
    angle_serialize(x, y);
    return angle_normalize((x + y) / 2);
}

inline bool angle_between(double l, double r, double x) {
    angle_serialize(l, r);
    if (l > r) std::swap(l, r);
    angle_serialize(l, x);
    return l <= x && x <= r;
}

inline double get_average(const std::vector<double> &array) {
    return std::accumulate(array.begin(), array.end(), 0.) / array.size();
}

inline double clamp(double lower, double upper, double x) {
    assert(lower <= upper);
    return std::min(std::max(lower, x), upper);
}

// 非内联函数,cpp中实现
Eigen::Vector3d operator*(const Eigen::Isometry3d &T, const Eigen::Vector3d &v);

#endif // _MATH_UTILS_H