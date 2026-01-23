#ifndef _NEW_BALLISTIC_H
#define _NEW_BALLISTIC_H

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <vector>

#include <Eigen/Dense>

class Ballistic {
    // ballistic calculate result
public:
    struct BallisticTable {
#define TABLE_WIDTH 1000
#define TABLE_HEIGHT 1000
       public:
        bool valid = false;
        double l, k, m, v, g;  // 枪长，空气阻力，弹丸质量，弹速，重力加速度
        int v_max;             // 弹速上限
        int x_n, y_n;
        // 下界，上界，步长
        double theta_l, theta_r, theta_d;  
        double x_l, x_r, x_d;
        double y_l, y_r, y_d;
        double sol_theta[TABLE_HEIGHT][TABLE_WIDTH],
            sol_t[TABLE_HEIGHT][TABLE_WIDTH];  // 解得的角度和时间，时间为 -1. 表示无解
#undef TABLE_WIDTH
#undef TABLE_HEIGHT
        bool contain(int x, int y) const { return 0 <= x && x < x_n && 0 <= y && y < y_n; }
    };

    struct BallisticParams {
        double noise_sigma;
        bool params_found;
        bool ballistic_refresh;
        std::vector<double> pitch2yaw_t;       // yaw转轴与pitch转轴相对位置
        double yaw2gun_offset;             // yaw转轴与发射机构的偏移距离
        double v9, v15, v16, v18, v30;     // 弹速
        double small_k, big_k;             // 空气阻力
        double l;                          // 枪长
        double theta_r, theta_l, theta_d;  // 仰角(上界、下界、步长)
        double x_r, x_l;                   // 距离(上界、下界)
        double y_r, y_l;                   // 高度(上界、下界)
        int x_n, y_n;                      // 步数(距离、高度)
        std::vector<double> stored_yaw_offset;
        std::vector<double> stored_pitch_offset;
        std::vector<std::vector<double>> stored_cam2gun_offset;
        std::string table_dir;
    };

    struct BallisticResult {
        bool fail;  // fail to solve
        double pitch, yaw, t;
        BallisticResult(double _p, double _y, double _t) : fail(false), pitch(_p), yaw(_y), t(_t) {}
        BallisticResult() : fail(true), pitch(0.), yaw(0.), t(0.) {}
    };
    Ballistic(const BallisticParams &param);
    Ballistic(rclcpp::Node *node, const std::string &table_dir);

    static bool dumpTable(const BallisticTable &table, const std::string &path);
    static bool loadTable(BallisticTable &table, const std::string &path);
    static BallisticParams declareParamsByNode(rclcpp::Node *node,
                                                               const std::string &table_dir);

    void reinit(const BallisticParams &params_);
    void refresh_velocity(bool isBig, double velocity);

    BallisticResult final_ballistic(Eigen::Vector3d p);
    BallisticResult final_ballistic(Eigen::Isometry3d T, Eigen::Vector3d p);
    std::pair<double, double> table_ballistic(const BallisticTable *tar, double x, double y);

    double rad2deg(double rad) { return rad * 180.0 / M_PI; }

    double deg2rad(double deg) { return deg * M_PI / 180.0; }

    bool feq(double a, double b) { 
        return rad2deg(fabs(a - b)) < 0.1; 
    }

    using Ptr = std::shared_ptr<Ballistic>;

private:
    double v, yaw_offset, pitch_offset;  // pitch_offset 向下为正  yaw_offset 向左为正
    Eigen::Matrix<double, 3, 1> cam2gun_offset;  // 发射装置与相机的偏移量
    const double g = 9.832, PI = 3.14159265358979;

    const std::vector<double> seg_x{1.395, 1.92, 2.73, 3.61, 5.07, 6.39};
    const std::vector<double> seg_s{-0.1, -0.07, -0.05, 0., 0.2, 0.3};

    // rclcpp::Logger logger;
    BallisticParams params;
    std::string table9_path, table15_path, table16_path, table18_path, table30_path;
    BallisticTable Table9, Table15, Table16, Table18, Table30;
    BallisticTable *pTable = nullptr;
    void calculateTable(bool is_big, int v_max);
};



#endif // _BALLISTIC_H