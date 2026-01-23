#include "enemy_predictor/enemy_ballistic.h"

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <fstream>

Ballistic::Ballistic(const BallisticParams &param){
    reinit(param);
}

Ballistic::Ballistic(rclcpp::Node *node, const std::string &table_dir){
    params = declareParamsByNode(node, table_dir);
    reinit(params);
}

Ballistic::BallisticParams Ballistic::declareParamsByNode(rclcpp::Node *node,const std::string &table_dir){
    BallisticParams params;
    params.params_found = node->declare_parameter("ballistic.params_found", false);
    // params_found为false说明加载失败
    assert(params.params_found && "Cannot found valid ballistic params");

    params.pitch2yaw_t = node->declare_parameter("pitch2yaw_t", std::vector<double>{0., 0., 0.});  // 没读进去就报错
    assert(params.pitch2yaw_t.size() == 3 && "pitch_to_yaw size must be 3!");
    params.yaw2gun_offset = sqrt(params.pitch2yaw_t[0] * params.pitch2yaw_t[0] + params.pitch2yaw_t[1] * params.pitch2yaw_t[1]);    
    
    params.stored_yaw_offset =
        node->declare_parameter("ballistic.stored_yaw_offset", std::vector<double>({}));
    params.stored_pitch_offset =
        node->declare_parameter("ballistic.stored_pitch_offset", std::vector<double>({}));    
    std::string offset_s_string = node->declare_parameter("ballistic.stored_cam2gun_offset", "");
    // 使用yaml-cpp 处理string
    YAML::Node v = YAML::Load(offset_s_string);
    params.stored_cam2gun_offset = v.as<std::vector<std::vector<double>>>();
    assert(params.stored_cam2gun_offset.size() == 5 && "stored_cam2gun_offset size must be 5!");
    assert(params.stored_yaw_offset.size() == 5 && "stored_yaw_offset size must be 5!");
    assert(params.stored_pitch_offset.size() == 5 && "stored_pitch_offset size must be 5!");

    // NGXY_DEBUG("stored_pitch_offset, [3]: %lf, [4]: %lf", params.stored_pitch_offset[3], params.stored_pitch_offset[4]);
    // NGXY_DEBUG("stored_yaw_offset, [3]: %lf, [4]: %lf", params.stored_yaw_offset[3], params.stored_yaw_offset[4]);

    params.ballistic_refresh = node->declare_parameter("ballistic.ballistic_refresh", false);
    params.small_k = node->declare_parameter("ballistic.small_k", 0.0);
    params.big_k = node->declare_parameter("ballistic.big_k", 0.0);
    params.l = node->declare_parameter("ballistic.l", 0.0);
    params.theta_l = node->declare_parameter("ballistic.theta_l", 0.0);
    params.theta_r = node->declare_parameter("ballistic.theta_r", 0.0);
    params.theta_d = node->declare_parameter("ballistic.theta_d", 0.0);
    params.x_l = node->declare_parameter("ballistic.x_l", 0.0);
    params.x_r = node->declare_parameter("ballistic.x_r", 0.0);
    params.x_n = node->declare_parameter("ballistic.x_n", 0);
    params.y_l = node->declare_parameter("ballistic.y_l", 0.0);
    params.y_r = node->declare_parameter("ballistic.y_r", 0.0);
    params.y_n = node->declare_parameter("ballistic.y_n", 0);
    params.v9 = node->declare_parameter("ballistic.v9", 0.0);
    params.v16 = node->declare_parameter("ballistic.v16", 0.0);
    params.v15 = node->declare_parameter("ballistic.v15", 0.0);
    params.v18 = node->declare_parameter("ballistic.v18", 0.0);
    params.v30 = node->declare_parameter("ballistic.v30", 0.0);
    params.table_dir = table_dir;
    params.noise_sigma = node->declare_parameter("ballistic.noise_sigma", 0.0);
    bool print_params = node->declare_parameter("ballistic.print_params", false);
    if (print_params) {
        RCLCPP_INFO(rclcpp::get_logger("Ballestic"), "Ballistic velocities: v9:%.2f v16:%.2f / v15:%.2f v:18%.2f v30:%.2f",
                                    params.v9, params.v16, params.v15, params.v18, params.v30);
    }
    return params;
}

void Ballistic::reinit(const BallisticParams &_params){
    params = _params;
    table9_path = params.table_dir + "/" + "big_9.dat";
    table16_path = params.table_dir + "/" + "big_16.dat";
    table15_path = params.table_dir + "/" + "small_15.dat";
    table18_path = params.table_dir + "/" + "small_18.dat";
    table30_path = params.table_dir + "/" + "small_30.dat";
    if (params.ballistic_refresh) {
        calculateTable(true, 9);
        calculateTable(true, 16);
        calculateTable(false, 15);
        calculateTable(false, 18);
        calculateTable(false, 30);
    }
    if (!loadTable(Table9, table9_path)) {
        RCLCPP_WARN(rclcpp::get_logger("Ballestic"), "Ballistic Table 9 load failed.");
        calculateTable(true, 9);
        if (!loadTable(Table9, table9_path)) {
            RCLCPP_WARN(rclcpp::get_logger("Ballestic"), "Ballistic Table 9 reload failed.");
        }
    }
    if (!loadTable(Table16, table16_path)) {
        RCLCPP_WARN(rclcpp::get_logger("Ballestic"), "Ballistic Table 16 load failed.");
        calculateTable(true, 16);
        if (!loadTable(Table16, table16_path)) {
            RCLCPP_WARN(rclcpp::get_logger("Ballestic"), "Ballistic Table 16 reload failed.");
        }
    }
    if (!loadTable(Table15, table15_path)) {
        RCLCPP_WARN(rclcpp::get_logger("Ballestic"), "Ballistic Table 15 load failed.");
        calculateTable(false, 15);
        if (!loadTable(Table15, table16_path)) {
            RCLCPP_WARN(rclcpp::get_logger("Ballestic"), "Ballistic Table 15 reload failed.");
        }
    }
    if (!loadTable(Table18, table18_path)) {
        RCLCPP_WARN(rclcpp::get_logger("Ballestic"), "Ballistic Table 18 load failed.");
        calculateTable(false, 18);
        if (!loadTable(Table18, table16_path)) {
            RCLCPP_WARN(rclcpp::get_logger("Ballestic"), "Ballistic Table 18 reload failed.");
        }
    }
    if (!loadTable(Table30, table30_path)) {
        RCLCPP_WARN(rclcpp::get_logger("Ballestic"), "Ballistic Table 30 load failed.");
        calculateTable(false, 30);
        if (!loadTable(Table30, table16_path)) {
            RCLCPP_WARN(rclcpp::get_logger("Ballestic"), "Ballistic Table 30 reload failed.");
        }
    }
}

bool Ballistic::dumpTable(const BallisticTable &table, const std::string &path){
    std::ofstream fout(path, std::ofstream::binary);
    if (!fout) return false;
    fout.write(reinterpret_cast<const char *>(&table), sizeof(table));
    fout.close();
    return true;
}

bool Ballistic::loadTable(BallisticTable &table, const std::string &path){
    std::ifstream fin(path, std::ifstream::binary);
    if (!fin) return false;
    fin.read(reinterpret_cast<char *>(&table), sizeof(table));
    fin.close();
    return true;
}


void Ballistic::refresh_velocity(bool isBig, double velocity){
    v = velocity;
    if (isBig) {
        if (v < 9.) {
            pTable = &Table9;
            cam2gun_offset = Eigen::Vector3d(params.stored_cam2gun_offset[0].data());
            yaw_offset = params.stored_yaw_offset[0];
            pitch_offset = params.stored_pitch_offset[0];
        } else {
            pTable = &Table16;
            cam2gun_offset = Eigen::Vector3d(params.stored_cam2gun_offset[2].data());
            yaw_offset = params.stored_yaw_offset[2];
            pitch_offset = params.stored_pitch_offset[2];
        }
    } else {
        if (v < 15.) {
            pTable = &Table15;
            cam2gun_offset = Eigen::Vector3d(params.stored_cam2gun_offset[1].data());
            yaw_offset = params.stored_yaw_offset[1];
            pitch_offset = params.stored_pitch_offset[1];
        } else if (v < 18.) {
            pTable = &Table18;
            cam2gun_offset = Eigen::Vector3d(params.stored_cam2gun_offset[3].data());
            yaw_offset = params.stored_yaw_offset[3];
            pitch_offset = params.stored_pitch_offset[3];
        } else {
            pTable = &Table30;
            cam2gun_offset = Eigen::Vector3d(params.stored_cam2gun_offset[4].data());
            yaw_offset = params.stored_yaw_offset[4];
            pitch_offset = params.stored_pitch_offset[4];
        }
    }
}

Ballistic::BallisticResult Ballistic::final_ballistic(Eigen::Vector3d p){

    double x = sqrt(p[0] * p[0] + p[1] * p[1]), y = p[2];
    RCLCPP_INFO(rclcpp::get_logger("Ballestic"), "final_ballistic: x:%lf y:%lf", x, y);
    auto [pitch, t] = table_ballistic(
        pTable, sqrt(x * x - params.yaw2gun_offset * params.yaw2gun_offset), y + cam2gun_offset[2]);
  
    RCLCPP_INFO_STREAM(rclcpp::get_logger("Ballestic: t"), "t: " << t);
    if (t < 0) return BallisticResult();
    double yaw = atan2(p[1], p[0]);
    return BallisticResult(-pitch + deg2rad(pitch_offset),
                      yaw + asin(params.yaw2gun_offset / x) + deg2rad(yaw_offset), t);
}

Ballistic::BallisticResult Ballistic::final_ballistic(Eigen::Isometry3d T, Eigen::Vector3d p){
    return final_ballistic(T * p);
}
//table_ballistic ------> 改为双线性差值
std::pair<double, double> Ballistic::table_ballistic(const BallisticTable *tar, double x, double y) {
    if (tar == nullptr) {
        RCLCPP_ERROR(rclcpp::get_logger("rm_base"), "You must call refresh_velocity() before using table_ballistic()");
        return std::make_pair(0., -1.);
    }

    // 1. 计算网格索引（浮点数）
    double i_float = (x - tar->x_l) / tar->x_d;
    double j_float = (y - tar->y_l) / tar->y_d;

    // 2. 扩展边界检查 - 允许外推
    // 2.1 如果索引完全超出表格范围，尝试外推
    if (i_float < 0 && j_float < 0) {
        // 两个维度都超出下限，使用最接近的点
        return std::make_pair(tar->sol_theta[0][0], tar->sol_t[0][0]);
    }
    if (i_float >= tar->x_n - 1 && j_float >= tar->y_n - 1) {
        // 两个维度都超出上限，使用最接近的点
        return std::make_pair(tar->sol_theta[tar->x_n - 1][tar->y_n - 1], 
                            tar->sol_t[tar->x_n - 1][tar->y_n - 1]);
    }

    // 3. 边界点处理 - 单维度超出
    // 3.1 x维度超出，y维度正常
    if (i_float < 0 || i_float >= tar->x_n - 1) {
        // 固定y维度进行外推
        int i_clamp = std::clamp(static_cast<int>(std::round(i_float)), 0, tar->x_n - 1);
        int j = static_cast<int>(std::clamp(j_float, 0.0, static_cast<double>(tar->y_n - 1)));

        if (i_float < 0) {
            // 近距离外推：使用最小的两个x点进行线性外推
            if (tar->x_n >= 2) {
                double dx_ratio = (x - tar->x_l) / (tar->x_l + tar->x_d - tar->x_l);
                double theta = tar->sol_theta[0][j] + 
                             (tar->sol_theta[1][j] - tar->sol_theta[0][j]) * dx_ratio;
                double t = tar->sol_t[0][j] + 
                          (tar->sol_t[1][j] - tar->sol_t[0][j]) * dx_ratio;
                return std::make_pair(std::max(theta, deg2rad(tar->theta_l)), 
                                    std::max(t, 0.01));
            }
        } else {
            // 远距离外推：使用最大的两个x点进行线性外推
            if (tar->x_n >= 2) {
                double dx_ratio = (x - (tar->x_l + (tar->x_n - 2) * tar->x_d)) / tar->x_d;
                double theta = tar->sol_theta[tar->x_n - 2][j] + 
                             (tar->sol_theta[tar->x_n - 1][j] - tar->sol_theta[tar->x_n - 2][j]) * dx_ratio;
                double t = tar->sol_t[tar->x_n - 2][j] + 
                          (tar->sol_t[tar->x_n - 1][j] - tar->sol_t[tar->x_n - 2][j]) * dx_ratio;
                return std::make_pair(std::min(theta, deg2rad(tar->theta_r)), t);
            }
        }
        // 如果表格太小，返回最接近的点
        return std::make_pair(tar->sol_theta[i_clamp][j], tar->sol_t[i_clamp][j]);
    }

    // 3.2 y维度超出，x维度正常
    if (j_float < 0 || j_float >= tar->y_n - 1) {
        // 固定x维度进行外推
        int i = static_cast<int>(std::clamp(i_float, 0.0, static_cast<double>(tar->x_n - 1)));
        int j_clamp = std::clamp(static_cast<int>(std::round(j_float)), 0, tar->y_n - 1);

        if (j_float < 0) {
            // 低高度外推
            if (tar->y_n >= 2) {
                double dy_ratio = (y - tar->y_l) / (tar->y_l + tar->y_d - tar->y_l);
                double theta = tar->sol_theta[i][0] + 
                             (tar->sol_theta[i][1] - tar->sol_theta[i][0]) * dy_ratio;
                double t = tar->sol_t[i][0] + 
                          (tar->sol_t[i][1] - tar->sol_t[i][0]) * dy_ratio;
                return std::make_pair(std::max(theta, deg2rad(tar->theta_l)), 
                                    std::max(t, 0.01));
            }
        } else {
            // 高高度外推
            if (tar->y_n >= 2) {
                double dy_ratio = (y - (tar->y_l + (tar->y_n - 2) * tar->y_d)) / tar->y_d;
                double theta = tar->sol_theta[i][tar->y_n - 2] + 
                             (tar->sol_theta[i][tar->y_n - 1] - tar->sol_theta[i][tar->y_n - 2]) * dy_ratio;
                double t = tar->sol_t[i][tar->y_n - 2] + 
                          (tar->sol_t[i][tar->y_n - 1] - tar->sol_t[i][tar->y_n - 2]) * dy_ratio;
                return std::make_pair(theta, t);
            }
        }
        // 如果表格太小，返回最接近的点
        return std::make_pair(tar->sol_theta[i][j_clamp], tar->sol_t[i][j_clamp]);
    }

    // 4. 正常范围内 - 使用双线性插值
    int i = static_cast<int>(i_float);
    int j = static_cast<int>(j_float);

    double dx = i_float - i;
    double dy = j_float - j;

    // 双线性插值权重
    double w00 = (1-dx) * (1-dy);
    double w10 = dx * (1-dy);
    double w01 = (1-dx) * dy;
    double w11 = dx * dy;

    // 获取四个点的值
    double theta = 
        tar->sol_theta[i][j] * w00 +
        tar->sol_theta[i+1][j] * w10 +
        tar->sol_theta[i][j+1] * w01 +
        tar->sol_theta[i+1][j+1] * w11;

    double t = 
        tar->sol_t[i][j] * w00 +
        tar->sol_t[i+1][j] * w10 +
        tar->sol_t[i][j+1] * w01 +
        tar->sol_t[i+1][j+1] * w11;

    // 5. 放宽验证条件
    if (t <= 0) {
        // 即使计算出的t为负，也尝试返回一个最小合理值
        return std::make_pair(theta, 0.01);
    }

    if (t > 10.0) {  // 放宽最大飞行时间限制
        t = 10.0;  // 设置为最大合理值
    }

    double theta_deg = rad2deg(theta);
    if (theta_deg < tar->theta_l - 10.0 || theta_deg > tar->theta_r + 10.0) {
        // 即使角度超出范围，也返回结果
        theta_deg = std::clamp(theta_deg, tar->theta_l - 10.0, tar->theta_r + 10.0);
        theta = deg2rad(theta_deg);
    }

    return std::make_pair(theta, t);
}

//inline bool feq(double a, double b) { return rad2deg(fabs(a - b)) < 0.1; }

void Ballistic::calculateTable(bool is_big, int v_max){
    std::shared_ptr<BallisticTable> tar = std::make_shared<BallisticTable>();
    
    // 1. 基本参数
    tar->v_max = v_max;
    tar->g = 9.7925;
    tar->m = is_big ? 0.0445 : 0.0032;
    tar->k = is_big ? params.big_k : params.small_k;
    
    // 弹速设置
    tar->v = 16;
    if (v_max == 30) tar->v = params.v30;
    else if (v_max == 18) tar->v = params.v18;
    else if (v_max == 16) tar->v = params.v16;
    else if (v_max == 15) tar->v = params.v15;
    else if (v_max == 9) tar->v = params.v9;
    
    // 表格参数
    tar->l = params.l;
    tar->theta_l = params.theta_l;
    tar->theta_r = params.theta_r;
    tar->theta_d = params.theta_d;
    tar->x_l = params.x_l;
    tar->x_r = params.x_r;
    tar->x_n = params.x_n;
    tar->y_l = params.y_l;
    tar->y_r = params.y_r;
    tar->y_n = params.y_n;
    
    // 2. 计算网格间距
    tar->x_d = (tar->x_r - tar->x_l) / tar->x_n;
    tar->y_d = (tar->y_r - tar->y_l) / tar->y_n;
    
    // 3. 初始化数组
    for (int i = 0; i < tar->x_n; i++) {
        std::fill_n(tar->sol_t[i], tar->y_n, -1.0);
    }
    
    RCLCPP_INFO(rclcpp::get_logger("ballistic"),
               "Generating ballistic table for v=%d m/s", (int)tar->v);
    
    // 4. 计算循环（计算有效点）
    int total_calculated = 0;
    
    for (double theta_deg = tar->theta_l; 
         theta_deg <= tar->theta_r; 
         theta_deg += tar->theta_d) {
        
        double theta_rad = deg2rad(theta_deg);
        double cos_theta = cos(theta_rad);
        double sin_theta = sin(theta_rad);
        
        for (int i = 0; i < tar->x_n; i++) {
            double x = tar->x_l + i * tar->x_d;
            double tra_x = x - tar->l * cos_theta;
            
            if (tra_x <= 0) continue;
            
            // 计算飞行时间
            double ratio = tar->k * tra_x / (tar->m * tar->v * cos_theta);
            if (ratio >= 0.95 || ratio <= 0.05) continue;  // 合理范围
            
            double t = -tar->m / tar->k * log(1.0 - ratio);
            if (t <= 0) continue;
            
            // 计算高度
            double tra_y = (tan(theta_rad) + tar->m * tar->g / 
                          (tar->k * tar->v * cos_theta)) * tra_x -
                          tar->m * tar->g / tar->k * t;
            double y = tra_y + tar->l * sin_theta;
            
            // 映射到网格
            int j = round((y - tar->y_l) / tar->y_d);
            if (j < 0 || j >= tar->y_n) continue;
            
            // 存储结果（覆盖较小角度）
            if (tar->sol_t[i][j] < 0 || theta_rad < tar->sol_theta[i][j]) {
                tar->sol_theta[i][j] = theta_rad;
                tar->sol_t[i][j] = t;
                total_calculated++;
            }
        }
    }
    
    // 5. 简单的线性插值填充（仅y方向）
    for (int i = 0; i < tar->x_n; i++) {
        int prev_valid = -1;
        for (int j = 0; j < tar->y_n; j++) {
            if (tar->sol_t[i][j] > 0) {
                if (prev_valid >= 0 && prev_valid < j-1) {
                    // 在两个有效点之间线性插值
                    double delta_theta = (tar->sol_theta[i][j] - tar->sol_theta[i][prev_valid]) 
                                        / (j - prev_valid);
                    double delta_t = (tar->sol_t[i][j] - tar->sol_t[i][prev_valid]) 
                                    / (j - prev_valid);
                    
                    for (int k = prev_valid + 1; k < j; k++) {
                        tar->sol_theta[i][k] = tar->sol_theta[i][prev_valid] + 
                                              delta_theta * (k - prev_valid);
                        tar->sol_t[i][k] = tar->sol_t[i][prev_valid] + 
                                          delta_t * (k - prev_valid);
                    }
                }
                prev_valid = j;
            }
        }
    }
    
    // 6. 统计信息
    int valid_points = 0;
    for (int i = 0; i < tar->x_n; i++) {
        for (int j = 0; j < tar->y_n; j++) {
            if (tar->sol_t[i][j] > 0) valid_points++;
        }
    }
    
    RCLCPP_INFO(rclcpp::get_logger("ballistic"),
               "Table generated: %d/%d points (%.1f%%)",
               valid_points, tar->x_n * tar->y_n,
               100.0 * valid_points / (tar->x_n * tar->y_n));
    
    // 7. 保存表格
    tar->valid = true;
    std::string save_buffer = params.table_dir + "/" + 
                             (is_big ? "big" : "small") + "_" +
                             std::to_string(tar->v_max) + ".dat";
    
    dumpTable(*tar, save_buffer);
}

