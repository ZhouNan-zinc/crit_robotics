#include <Eigen/Dense>
#include "enemy_predictor/enemy_predictor_node.h"
#include "enemy_predictor/enemy_ballistic.h"


Eigen::Isometry3d EnemyPredictorNode::getTrans(const std::string& source_frame, const std::string& target_frame, 
                                               rclcpp::Time timestamp_image) {
    
    geometry_msgs::msg::TransformStamped t;
    try {
        t = tf2_buffer->lookupTransform(
            target_frame,
            source_frame,
            timestamp_image,
            rclcpp::Duration::from_seconds(0.5)
        );
    } catch (const std::exception& ex) {
        printf(
            "Could not transform %s to %s: %s",
            source_frame.c_str(),
            target_frame.c_str(),
            ex.what()
        );
    }
    // to Eigen::Isometry3d
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() = Eigen::Vector3d(
        t.transform.translation.x,
        t.transform.translation.y,
        t.transform.translation.z
    );
    Eigen::Quaterniond quat(
        t.transform.rotation.w,
        t.transform.rotation.x,
        t.transform.rotation.y,
        t.transform.rotation.z
    );
    transform.rotate(quat);
    
    return transform;
    //return tf2::transformToEigen(t);
} 
void EnemyPredictorNode::updateArmorDetection(std::vector<cv::Point3f> object_points,
                                              Detection& det,
                                              rclcpp::Time timestamp_image) {
    std::vector<cv::Point2f> reprojected_points;

    cv::Mat tvec = (cv::Mat_<double>(3, 1) << 
                    det.position.x(), 
                    det.position.y(), 
                    det.position.z());
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);

    double roll = det.orientation.x();
    double pitch = det.orientation.y();
    double yaw = det.orientation.z();

    double cr = cos(roll), sr = sin(roll);
    double cp = cos(pitch), sp = sin(pitch);
    double cy = cos(yaw), sy = sin(yaw);

    cv::Mat R = (cv::Mat_<double>(3, 3) <<
        cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr,
        sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr,
        -sp,    cp*sr,             cp*cr);

    cv::Rodrigues(R, rvec);

    cv::projectPoints(object_points, rvec, tvec, 
                       visualize_.camera_matrix, visualize_.dist_coeffs, reprojected_points);
    //if (reprojected_points.size() >= 4) {
    //     cv::rectangle(visualize_.armor_img, reprojected_points[0], reprojected_points[2], 
    //                   cv::Scalar(255, 0, 0), 2);
    //}

    const cv::Point2f& p0 = reprojected_points[0];
    const cv::Point2f& p1 = reprojected_points[1];
    const cv::Point2f& p2 = reprojected_points[2];
    const cv::Point2f& p3 = reprojected_points[3];
    // 鞋带公式
    double area = 0.0;
    area += p0.x * p1.y - p1.x * p0.y;
    area += p1.x * p2.y - p2.x * p1.y;
    area += p2.x * p3.y - p3.x * p2.y;
    area += p3.x * p0.y - p0.x * p3.y;
    det.area_2d = std::abs(area) / 2.0;

    Eigen::Vector3d camera_tvec_eigen = Eigen::Map<Eigen::Vector3d>(visualize_.camera_tvec.ptr<double>());
    visualize_.camara_to_odom = getTrans("camera_optical_frame", "odom", timestamp_image);
    
    det.position = visualize_.camara_to_odom * det.position;  //camera to odom
    visualizeAimCenter(det.position, cv::Scalar(225, 0, 225));
}
//--------------------------------Tracking with Armor Filter--------------------------------------------------
EnemyPredictorNode::ArmorTracker::ArmorTracker(int tracker_idx, 
                                              int armor_class_id,
                                              const Eigen::Vector3d& init_pos, 
                                              double timestamp,
                                              double armor_yaw,
                                              double area_2d)
    : tracker_idx(tracker_idx), 
      armor_class_id(armor_class_id), 
      yaw(armor_yaw), 
      last_yaw(armor_yaw),
      area_2d(area_2d) {
    
    phase_id = -1;
    position = init_pos;
    last_position = init_pos;
    predicted_position = init_pos;
    position_history.push_back(init_pos);
    last_update_time = timestamp;
    Eigen::Vector3d to_ekf_xyyaw = Eigen::Vector3d(init_pos.x(), init_pos.y(), armor_yaw);
    ekf.init(to_ekf_xyyaw, timestamp);
}


void EnemyPredictorNode::ArmorTracker::update(const Eigen::Vector3d& new_position, 
                                            int armor_class_id_, double timestamp,
                                            double armor_yaw) {
    last_position = position;
    position = new_position;
    armor_class_id = armor_class_id_;
    // 处理yaw过零（模仿上一版本）
    if (armor_yaw - last_yaw < -M_PI * 1.5) {
        yaw_round++;
    } else if (armor_yaw - last_yaw > M_PI * 1.5) {
        yaw_round--;
    }
    last_yaw = armor_yaw;
    yaw = armor_yaw + yaw_round * 2 * M_PI;
    
    // 更新EKF
    Eigen::Vector3d new_xyyaw = Eigen::Vector3d(new_position.x(), new_position.y(), armor_yaw);
    ekf.update(new_xyyaw, timestamp);
    ZEKF::Vz z_obs;
    z_obs(0) = new_position(2);
    zekf.update(z_obs, timestamp);
    is_active = true;
    // 更新历史
    position_history.push_back(new_position);
    if (position_history.size() > 100) {
        position_history.erase(position_history.begin());
    }
    
    last_update_time = timestamp;
    missing_frames = 0;
}
void EnemyPredictorNode::ToupdateArmors(const std::vector<Detection, Eigen::aligned_allocator<Detection>>& detections, double timestamp,
                                        std::vector<int>& active_armor_idx) {
    
    if (detections.empty()) {
        RCLCPP_INFO(get_logger(), "No Armor This Frame");
        return;
    }
  
    for(size_t i = 0; i < detections.size(); i++){

        enemies_[detections[i].armor_class_id % 10 - 1].is_active = true;
        bool has_history_tracker = false;

        for (int j = 0; j < armor_trackers_.size(); j++) {

            armor_trackers_[j].is_active = false;
            
            if(armor_trackers_[j].tracker_idx == detections[i].armor_idx){
                active_armor_idx.push_back(j);
               
                armor_trackers_[j].update(detections[i].position, detections[i].armor_class_id, timestamp, detections[i].yaw);
        
                has_history_tracker = true;
                armor_trackers_[j].is_active = true;
        
                break;
            }
        }   
        if(has_history_tracker == false){
            create_new_tracker(detections[i], timestamp, active_armor_idx);
        }
    }
}

void EnemyPredictorNode::updateEnemy(Enemy& enemy, double timestamp, std::vector<int>& active_armor_idx) {
    
    std::vector<ArmorTracker*> active_armors_this_enemy;

    for (int idx : active_armor_idx) {
        if(armor_trackers_[idx].armor_class_id % 10 == enemy.class_id){
            active_armors_this_enemy.push_back(&armor_trackers_[idx]);
        }
    }
    for(ArmorTracker* tracker : active_armors_this_enemy){
        findBestPhaseForEnemy(enemy, *tracker, active_armors_this_enemy);
        RCLCPP_INFO(get_logger(), "Armor(%d), Phase_id = %d, area_2d = %lf", tracker->tracker_idx, tracker->phase_id, tracker->area_2d);
    }
    
    //calculateEnemyCenterAndRadius(enemy, timestamp, active_armors_this_enemy);
    //enemy.enemy_ckf.radius = enemy.radius;
    if(!enemy.enemy_ckf.is_initialized_){
        enemy.enemy_ckf.initializeCKF();
        enemy.enemy_ckf.reset(active_armors_this_enemy[0] -> position, active_armors_this_enemy[0] -> yaw, active_armors_this_enemy[0] -> phase_id, timestamp);
    }
    else if(enemy.missing_frame > 20 || enemy.is_valid == false){
        enemy.enemy_ckf.reset(active_armors_this_enemy[0] -> position, active_armors_this_enemy[0] -> yaw, active_armors_this_enemy[0] -> phase_id, timestamp);
    }
    else{
        for(ArmorTracker* tracker : active_armors_this_enemy){
           enemy.enemy_ckf.update(tracker->position, tracker->yaw, timestamp, tracker->phase_id);
        }
    }
    
}

void EnemyPredictorNode::EnemyManage(double timestamp, rclcpp::Time timestamp_image, 
                                     std::vector<int>& active_enemies_idx, std::vector<int>& active_armor_idx) {

    for(Enemy& enemy : enemies_){
        if(enemy.is_active){
            updateEnemy(enemy, timestamp, active_armor_idx);
            active_enemies_idx.push_back(enemy.class_id - 1); 
            enemy.missing_frame = 0;
        }else{
            enemy.missing_frame ++;
            if(enemy.missing_frame > 15){
                enemy.reset();
                enemy.is_valid = false;
            }
        }
    }
    int target_enemy_idx = -1;

    if(active_enemies_idx.size()== 0){
        return;
    }
    //是否需要考虑操作手按right键，但这一帧没有detect到操作手正在tracking的enemy？？？
    if(active_enemies_idx.size()== 1){
        target_enemy_idx = active_enemies_idx[0];
        //RCLCPP_INFO(get_logger(),"target_enemy_idx:%d",target_enemy_idx);
    }
    else if(active_enemies_idx.size() > 1){
        //基于到准星距离（操作手）的决策
        double enemy_to_heart_min = 10000.0;
        if(cmd.last_target_enemy_idx != -1 && params_.right_press == true){
            target_enemy_idx = cmd.last_target_enemy_idx;
        }
        else{
            for(int i = 0; i < active_enemies_idx.size(); i++){
                //如果某个enemy只存在过一个装甲板的detection,使用const_radius但只做ekf得到cmd结果
                //那么此时的enemy.center不准，但是choose enemy的时候需要使用enemy.center
                RCLCPP_INFO(get_logger(),"Start To Choose Target");
                Eigen::Vector3d enemy_center_cam = visualize_.camara_to_odom.inverse() *enemies_[active_enemies_idx[i]].center;
               
                // DEBUG!!!!!!!!!!!! odom to camera ,but distance???????? 
                std::vector<cv::Point2f> reprojected_points;
                std::vector<cv::Point3f> points_3d;
                cv::Point3f point_3d(enemy_center_cam.x(), enemy_center_cam.y(), enemy_center_cam.z());
                points_3d.push_back(point_3d);
                
                cv::projectPoints(points_3d, visualize_.camera_rvec, visualize_.camera_tvec, 
                          visualize_.camera_matrix, visualize_.dist_coeffs, reprojected_points);
                
                if(reprojected_points.size() > 0){
                    // Attention: camera和图传中心不一样，记得获取图传准星的标定结果！！！
                    float dx = static_cast<float>(reprojected_points[0].x - visualize_.camera_heart.x);
                    float dy = static_cast<float>(reprojected_points[0].y - visualize_.camera_heart.y);
                    double dis = std::sqrt(dx * dx + dy * dy);
                    if(dis < enemy_to_heart_min){
                        enemy_to_heart_min = dis;
                        target_enemy_idx = active_enemies_idx[i];

                    }
                }
                target_enemy_idx = active_enemies_idx[i];
            }
            RCLCPP_INFO(get_logger(),"Finish To Choose Target");
        }
    }
    getCommand(enemies_[target_enemy_idx], timestamp, timestamp_image, active_armor_idx);
}
//void EnemyPredictorNode::calculateEnemyCenterAndRadius(Enemy& enemy, double timestamp, std::vector<ArmorTracker*> active_armors_this_enemy) {
//
//    if (active_armors_this_enemy.empty()){
//        return;
//    }
//    if (active_armors_this_enemy.size() == 1) {
//        // 单个装甲板：基于相位推测中心
//        double phase_angle = active_armors_this_enemy[0]->phase_id * (M_PI / 2.0);
//        double center_yaw = active_armors_this_enemy[0]->yaw - phase_angle + M_PI;
//        enemy.center = active_armors_this_enemy[0]->position + enemy.radius[active_armors_this_enemy[0]->phase_id % 2] * 
//                      Eigen::Vector3d(std::cos(center_yaw), -std::sin(center_yaw), 0);  //认为armor_z == enemy_z ? ? ?
//        //RCLCPP_INFO(get_logger(), "Calculate center = %f, %f, %f", enemy.center(0), enemy.center(1), enemy.center(2));
//    }
//    else if (active_armors_this_enemy.size() >= 2) {
//    // 最小二乘法求同心圆心（处理>=2个装甲板）
//    // 使用装甲板的法向量（垂直于装甲板平面）和位置信息
//    std::vector<Eigen::Vector2d> armor_points;      // 装甲板位置（2D）
//    std::vector<Eigen::Vector2d> normal_vectors;    // 法向量（垂直于装甲板平面向外）
//    std::vector<int> phase_ids;                     // 相位ID
//    double z_sum = 0.0;                             // Z坐标总和
//
//    // 收集所有装甲板的信息
//    for (const auto& armor_ptr : active_armors_this_enemy) {
//        ArmorTracker& armor = *armor_ptr;
//
//        // 装甲板位置（XY平面）
//        armor_points.emplace_back(armor.position.x(), armor.position.y());
//
//        // 计算法向量：yaw是垂直装甲板平面的朝向角
//        // cos(yaw), -sin(yaw) (垂直armor平面向内)
//        Eigen::Vector2d normal(std::cos(armor.yaw), -std::sin(armor.yaw));
//        normal_vectors.push_back(normal.normalized());
//
//        // 相位ID
//        phase_ids.push_back(armor.phase_id);
//
//        z_sum += armor.position.z();
//    }
//
//    // 最小二乘法求解圆心
//    if (armor_points.size() >= 2) {
//        //  A^T * A 和 A^T * b
//        // 对于每个装甲板，约束方程为：法向量与圆心到装甲板向量的叉积为0
//        // 即：n_i × (center - p_i) = 0
//        // 展开：-n_i.y * center.x + n_i.x * center.y = -n_i.y * p_i.x + n_i.x * p_i.y
//        Eigen::Matrix2d ATA = Eigen::Matrix2d::Zero();
//        Eigen::Vector2d ATb = Eigen::Vector2d::Zero();
//
//        for (size_t i = 0; i < armor_points.size(); ++i) {
//            double nx = normal_vectors[i].x();
//            double ny = normal_vectors[i].y();
//            double px = armor_points[i].x();
//            double py = armor_points[i].y();
//
//            // 每个约束方程的权重（可以根据装甲板质量调整）
//            double weight = 1.0;
//
//            ATA(0, 0) += weight * ny * ny;          // (-ny)^2
//            ATA(0, 1) += weight * (-ny) * nx;       // (-ny) * nx
//            ATA(1, 0) += weight * nx * (-ny);       // nx * (-ny)
//            ATA(1, 1) += weight * nx * nx;          // nx^2
//
//            double bi = -ny * px + nx * py;
//            ATb(0) += weight * (-ny) * bi;
//            ATb(1) += weight * nx * bi;
//        }
//
//        // 求解方程组 (ATA * center_2d = ATb)
//        double det = ATA.determinant();
//
//        if (std::abs(det) > 1e-8) {
//            Eigen::Vector2d center_2d = ATA.inverse() * ATb;
//
//            // 对于两个装甲板的情况，进行特殊处理
//            if (armor_points.size() == 2) {
//                // 检查两个装甲板是否近似90度
//                double angle_between = std::acos(normal_vectors[0].dot(normal_vectors[1]));
//
//                // 如果两个法向量夹角接近180度（平行），说明估计不可靠
//                if (std::abs(angle_between) > M_PI * 0.9) {
//                    RCLCPP_WARN(get_logger(), "Two armor normals are nearly parallel, estimation may be inaccurate");
//                }
//
//                // 计算圆心到两个装甲板的距离
//                double dist1 = (center_2d - armor_points[0]).norm();
//                double dist2 = (center_2d - armor_points[1]).norm();
//
//                // 检查圆心是否在法线正方向上
//                Eigen::Vector2d v1 = center_2d - armor_points[0];
//                Eigen::Vector2d v2 = center_2d - armor_points[1];
//                double dot1 = v1.dot(normal_vectors[0]);
//                double dot2 = v2.dot(normal_vectors[1]);
//
//                // 如果点积为负，说明圆心在法线反方向，需要调整
//                if (dot1 < 0 || dot2 < 0) {
//                    RCLCPP_INFO(get_logger(), "Center appears to be behind armor planes, adjusting...");
//
//                    // 使用法线交点法
//                    Eigen::Matrix2d A;
//                    A << normal_vectors[0].x(), -normal_vectors[1].x(),
//                         normal_vectors[0].y(), -normal_vectors[1].y();
//
//                    Eigen::Vector2d b_vec = armor_points[1] - armor_points[0];
//                    double det_A = A.determinant();
//
//                    if (std::abs(det_A) > 1e-8) {
//                        Eigen::Vector2d t = A.inverse() * b_vec;
//                        Eigen::Vector2d intersection = armor_points[0] + t(0) * normal_vectors[0];
//
//                        // 检查交点是否在法线正方向
//                        Eigen::Vector2d test_v1 = intersection - armor_points[0];
//                        Eigen::Vector2d test_v2 = intersection - armor_points[1];
//
//                        if (test_v1.dot(normal_vectors[0]) > 0 && test_v2.dot(normal_vectors[1]) > 0) {
//                            center_2d = intersection;
//                            RCLCPP_INFO(get_logger(), "Using intersection method for center");
//                        }
//                    }
//                }
//            }
//
//            // 验证圆心合理性
//            bool center_valid = true;
//            std::vector<double> radii;
//            std::vector<double> dot_products;
//
//            for (size_t i = 0; i < armor_points.size(); ++i) {
//                Eigen::Vector2d v = center_2d - armor_points[i];
//                double dot = v.dot(normal_vectors[i]);
//                double radius = v.norm();
//
//                radii.push_back(radius);
//                dot_products.push_back(dot);
//
//                // 圆心应该在法线正方向上（点积为正）
//                if (dot < 0.01) {
//                    center_valid = false;
//                    RCLCPP_INFO(get_logger(), "Center validation failed: armor %zu dot=%.3f", i, dot);
//                }
//
//                // 半径应该在合理范围内
//                if (radius < 0.1 || radius > 0.5) {
//                    center_valid = false;
//                    RCLCPP_INFO(get_logger(), "Center validation failed: armor %zu radius=%.3f", i, radius);
//                }
//            }
//
//            if (center_valid) {
//                double z = z_sum / active_armors_this_enemy.size();
//                enemy.center = Eigen::Vector3d(center_2d.x(), center_2d.y(), z);
//
//                //visualizeAimCenter(enemy.center, cv::Scalar(255, 0, 0));
//
//                // 计算每个装甲板的半径，并根据相位ID更新对应的半径
//                std::vector<int> valid_radii_count = {0, 0};
//
//                for (size_t i = 0; i < active_armors_this_enemy.size(); ++i) {
//                    const auto& armor_ptr = active_armors_this_enemy[i];
//                    double r = (enemy.center - armor_ptr->position).norm();
//                    //RCLCPP_INFO(get_logger(), "armor %d position : %lf, %lf, %lf", i, armor_ptr->position(0),armor_ptr->position(1),armor_ptr->position(2));
//                    // 检查半径是否在合理范围内
//                    if (r >= 0.15 && r <= 0.4) {
//                        int phase_index = phase_ids[i] % 2;
//
//                        enemy.radius[phase_index] = 0.7 * enemy.radius[phase_index] + 0.3 * r;
//
//                        valid_radii_count[phase_index]++;
//
//                        //RCLCPP_INFO(get_logger(), 
//                        //    "Armor %zu (phase %d): r=%.3f, updated radius[%d]=%.3f",
//                        //    i, phase_ids[i], r, phase_index, enemy.radius[phase_index]);
//                    }
//                }
//                if(valid_radii_count[0] > 0 && valid_radii_count[1] > 0){
//                    enemy.radius_cal = true;
//                }
//                //RCLCPP_INFO(get_logger(), 
//                //    "Calculated enemy center: (%.3f, %.3f, %.3f)",
//                //    enemy.center.x(), enemy.center.y(), enemy.center.z());
//
//            } else {
//                RCLCPP_WARN(get_logger(), 
//                    "Calculated center failed validation checks");
//            }
//
//        } else {
//            RCLCPP_WARN(get_logger(), 
//                "Matrix determinant too small (det=%.2e), cannot solve for center", det);
//        }
//    }
//}
//    
//    // =================== 简化版的可视化代码 ===================
//    
//    // 清空之前的标记
//    enemy_markers_.markers.clear();
//    
//    // 1. 创建球体标记（表示敌人中心）
//    visualization_msgs::msg::Marker sphere_marker;
//    sphere_marker.header.frame_id = "odom";  // 使用正确的坐标系
//    sphere_marker.header.stamp = this->now();  // 使用当前时间
//    
//    sphere_marker.ns = "enemy_centers";
//    sphere_marker.id = enemy.class_id;  // 使用敌人ID
//    
//    sphere_marker.type = visualization_msgs::msg::Marker::SPHERE;
//    sphere_marker.action = visualization_msgs::msg::Marker::ADD;
//    
//    sphere_marker.pose.position.x = enemy.center.x();
//    sphere_marker.pose.position.y = enemy.center.y();
//    sphere_marker.pose.position.z = enemy.center.z();
//    sphere_marker.pose.orientation.w = 1.0;
//    
//    sphere_marker.scale.x = 0.1;  // 直径
//    sphere_marker.scale.y = 0.1;
//    sphere_marker.scale.z = 0.1;
//    
//    // 设置颜色（红色表示敌人中心）
//    sphere_marker.color.r = 1.0;
//    sphere_marker.color.g = 0.0;
//    sphere_marker.color.b = 0.0;
//    sphere_marker.color.a = 0.8;  // 半透明
//    
//    sphere_marker.lifetime = rclcpp::Duration::from_seconds(0.2);  // 200ms生命周期
//    
//    // 将球体标记添加到数组
//    enemy_markers_.markers.push_back(sphere_marker);

//}
    
void EnemyPredictorNode::findBestPhaseForEnemy(Enemy& enemy, ArmorTracker& tracker, std::vector<ArmorTracker*> active_armors_this_enemy) {

    if(!enemy.is_valid){
        tracker.phase_id = 0;
        enemy.yaw = tracker.yaw;
        enemy.is_valid = true; // 每个enemy有了第一个armor时，第一个armor phase_id = 0, enemy.is_valid = true
        RCLCPP_INFO(get_logger(), "This enemy has its first armor, Give Phase_id %d",tracker.phase_id);
        return;
    }
    //std::vector<double> scores = {0.0, 0.0, 0.0, 0.0};
    bool by_relative = false;
    int candidate_phase = -1;
    for(size_t i = 0; i < active_armors_this_enemy.size(); i++){

        double angle_diff = normalize_angle(tracker.yaw - active_armors_this_enemy[i] -> yaw);
        
        if(active_armors_this_enemy[i]->phase_id != -1 && abs(angle_diff) > 1.4 && abs(angle_diff) < 1.75){
           if(angle_diff > 0){
              candidate_phase  = active_armors_this_enemy[i]->phase_id + 1;
              if(candidate_phase > 3){
                 candidate_phase = 0;
               }
            }
            else if(angle_diff < 0){
                candidate_phase = active_armors_this_enemy[i]->phase_id - 1;

                if(candidate_phase < 0){
                  candidate_phase = 3;
                }
            }
            by_relative = true;
        }
    }
    if(!by_relative){
        
        RCLCPP_INFO(get_logger(), "enemy.enemy_ckf.Xe(4) = %lf", enemy.enemy_ckf.Xe(4));
        RCLCPP_INFO(get_logger(), "tracker.yaw = %lf", tracker.yaw);
        double angle_diff = normalize_angle(tracker.yaw - enemy.enemy_ckf.Xe(4));

        if(angle_diff >= -M_PI/4 && angle_diff < M_PI/4){
            candidate_phase = 0; // 前方
        }
        else if(angle_diff >= M_PI/4 && angle_diff < 3*M_PI/4){
            candidate_phase = 1; // 左侧
        }
        else if(angle_diff >= -3*M_PI/4 && angle_diff < -M_PI/4){
            candidate_phase = 3; // 右侧
        }
        else{
            candidate_phase = 2; // 后方
        }
        RCLCPP_INFO(get_logger(), "Don't Have Useful relative armor!!!!!! candidate_phase = %d", candidate_phase);

    }
    if(tracker.phase_id != -1 && tracker.phase_id != candidate_phase){
        tracker.phase_id_cnt ++;
        if(tracker.phase_id_cnt > 4){
            tracker.phase_id = candidate_phase;
            tracker.phase_id_cnt = 0;
        }
    }
    else{
        tracker.phase_id = candidate_phase;
    }
    RCLCPP_INFO(get_logger(), "Give phase_id : %d", tracker.phase_id);
   // 收集已使用的相位
    //std::set<int> used_phases;
    //for(int idx : active_armor_idx){
    //    // collect active_armor's already phase_id [in this frame]
    //    if(armor_trackers_[idx].armor_class_id % 10 == enemy.class_id && 
    //       armor_trackers_[idx].phase_id != -1){
    //        used_phases.insert(armor_trackers_[idx].phase_id);
    //    }
    //}
  
    //if(used_phases.empty()){
    //    tracker.phase_id = 0;
    //    RCLCPP_INFO(get_logger(), "This enemy has no armor, give phase_id 0");
    //    return;
    //}
    //double min_cost = 1000.0;
    //double second_min_cost = 1000.0;
    //int best_phase = -1;
    //int second_best_phase = -1;
   
    // 简易匈牙利算法
    //for (int phase = 0; phase < 4; ++phase) {
//
    //    double total_cost = 0.0;
//
    //    double phase_angle = phase * (M_PI / 2.0);
    //    double expected_yaw = enemy.yaw + phase_angle;
    //    
    //    Eigen::Vector3d expected_pos = enemy.center + enemy.radius[phase % 2] * 
    //                                  Eigen::Vector3d(-std::cos(expected_yaw), 
    //                                                 std::sin(expected_yaw), 
    //                                                 0.0);
    //    double distance = (tracker.position - expected_pos).norm();
    //
    //    // 使用sigmoid函数归一化到[0, 1]
    //    // 参数：0.3表示30cm时成本为0.5  adjust it later !!!
    //    // 1.position cost
    //    double normalized_cost = 1.0 / (1.0 + exp(-10.0 * (distance - 0.3)));
//
    //    total_cost += 0.5 *normalized_cost;
//
    //    double angle_diff = std::atan2(std::sin(expected_yaw - tracker.yaw),
    //                               std::cos(expected_yaw - tracker.yaw));
    //
    //    // 取绝对值并归一化到[0, 1]
    //    // 2.yaw cost
    //    double normalized_diff = std::abs(angle_diff) / M_PI;
//
    //    total_cost += 0.3 *normalized_diff* normalized_diff;
    //    // 3.相位冲突 cost
    //    if (used_phases.find(phase) != used_phases.end()) {
    //        total_cost += 0.2 * 1.0; // 相位已占用，增加成本
    //    }
    //    if (phase == tracker.phase_id){
    //        total_cost*= 0.8;
    //    }
    //    if(total_cost < min_cost){
    //        second_min_cost = min_cost;
    //        second_best_phase = best_phase;
//
    //        min_cost = total_cost;
    //        best_phase = phase;
    //    }
    //    else if(total_cost < second_min_cost){
    //        second_min_cost = total_cost;
    //        second_best_phase = phase;
    //    }
    //}
    //RCLCPP_INFO(get_logger(), "Finish Xiongyali");
    //tracker.phase_conf = 1 - min_cost;
    //// 一致性奖励
    //double expected_yaw = enemy.yaw + best_phase * (M_PI / 2.0);
    //Eigen::Vector3d expected_pos = enemy.center + enemy.radius[best_phase % 2] * 
    //                              Eigen::Vector3d(-std::cos(expected_yaw), 
    //                                             std::sin(expected_yaw), 
    //                                             0.0);
    //double position_error = (tracker.position - expected_pos).norm();
    //
    //if (position_error < 0.07) { // 10cm以内认为很匹配  adjust it later !!!
    //    tracker.phase_conf = std::min(tracker.phase_conf + 0.1, 1.0);
    //}
    //
    //if(used_phases.find(best_phase) == used_phases.end()){
    //    tracker.phase_id = best_phase;
    //    RCLCPP_INFO(get_logger(), "assigned to enemy , phase_id %d: ", best_phase);
    //}
    //else{
    //    RCLCPP_INFO(get_logger(), "Conflict Phase_id!!!! %d", best_phase);
    //    for(int idx : active_armor_idx){
    //        if(armor_trackers_[idx].armor_class_id % 10 == enemy.class_id && armor_trackers_[idx].phase_id == best_phase){
    //            if(armor_trackers_[idx].phase_conf < tracker.phase_conf){
    //               tracker.phase_id = best_phase;
    //               // 冲突的tracker, 若conf不如新armor, phase_id设置为-1 TO DO : 对原来conf更低的tracker处理phase_id
    //               armor_trackers_[idx].phase_id = -1;
    //            }else{
    //               tracker.phase_id = second_best_phase;
    //            }
    //        }
    //    }
    //}
    // 所有相位都被使用，基于距离选择
    //double min_distance = std::numeric_limits<double>::max();
    //int best_phase = 0;
    //
    //for (int phase = 0; phase < 4; ++phase) {
    //    double phase_angle = phase * (M_PI / 2.0);
    //    double expected_yaw = enemy.yaw + phase_angle - M_PI;
    //    
    //    Eigen::Vector3d expected_pos = enemy.center + enemy.radius * 
    //                                  Eigen::Vector3d(std::cos(expected_yaw), 
    //                                                 std::sin(expected_yaw), 
    //                                                 tracker.position.z());
    //    
    //    double distance = (tracker.position - expected_pos).norm();
    //    if (distance < min_distance) {
    //        min_distance = distance;
    //        best_phase = phase;
    //    }
    //}
}
int EnemyPredictorNode::ChooseMode(Enemy &enemy, double timestamp){
    if(abs(enemy.enemy_ckf.Xe(5)) > cmd.high_spd_rotate_thresh){
       return 1;
    }else{
       return 0;
    }
}
void EnemyPredictorNode::getCommand(Enemy& enemy, double timestamp, rclcpp::Time timestamp_image, std::vector<int>& active_armor_idx){
    
    cmd.cmd_mode = ChooseMode(enemy, timestamp);
    RCLCPP_INFO(get_logger(),"MODE :%d",cmd.cmd_mode);
    cmd.target_enemy_idx = enemy.class_id -1;
    cmd.last_target_enemy_idx = cmd.target_enemy_idx;
    //RCLCPP_INFO(get_logger(),"cmd.last_target_enemy_idx:%d",cmd.last_target_enemy_idx);
    auto predict_func_double = [this, &enemy](ArmorTracker& tracker, double time_offset, double timestamp) -> Eigen::Vector3d{
        return FilterManage(enemy, time_offset, tracker, timestamp);
    };
    std::vector<ArmorTracker*> active_trackers;

    for(int idx : active_armor_idx){
        if(armor_trackers_[idx].armor_class_id % 10 == enemy.class_id){
            active_trackers.push_back(&armor_trackers_[idx]);
        }
    }
    if(cmd.cmd_mode == 1){
        
        Eigen::Vector3d armor_center_pre = Eigen::Vector3d(0, 0, 0);
        std::vector<double> yaws(4);
        //use enemy.center or enemy.center_pre ???
        //一直瞄准yaw近似车体中心的位置，but pitch不等于整车中心，当预测某一装甲板即将旋转到该直线时给发弹指令
        
        for(int i = 0; i < active_trackers.size(); i++){
           auto [ball_res, p] = calc_ballistic_((params_.response_delay + params_.shoot_delay), timestamp_image, *active_trackers[i], timestamp, predict_func_double);
           
           Eigen::Vector3d enemy_center_pre = enemy.enemy_ckf.predictCenterPosition(enemy.center(2), ball_res.t, timestamp);
           double enemy_yaw_xy = std::atan2(enemy_center_pre[1], enemy_center_pre[0]); 

           armor_center_pre = FilterManage(enemy, ball_res.t, *active_trackers[i], timestamp);

           yaws[i] = std::atan2(armor_center_pre[1], armor_center_pre[0]);
           
           if (std::abs(yaws[i] - enemy_yaw_xy) < cmd.yaw_thresh){
               cmd.aim_center = armor_center_pre; 
               cmd.cmd_pitch = ball_res.pitch;
               cmd.cmd_yaw = ball_res.yaw;
               RCLCPP_INFO(this->get_logger(), 
                           "Firing at phase %d: pitch=%.3f°, yaw=%.3f°, aim=(%.3f,%.3f,%.3f)",
                           i, cmd.cmd_pitch, cmd.cmd_yaw,
                           cmd.aim_center.x(), cmd.aim_center.y(), cmd.aim_center.z());
               break; 
            }
        }
    }
    else if(cmd.cmd_mode == 0){
        if(active_trackers.size() == 0){
            RCLCPP_WARN(this->get_logger(), "No active trackers found");
            return;
        }
        else if(active_trackers.size() == 1){
            
            //如果没有用同时出现的两个armor计算过radius,那么不用整车ckf,直接使用ekf.update/predict
            RCLCPP_INFO(get_logger(),"To Calculate ballistic");
            auto [ball_res, p] = calc_ballistic_(
                (params_.response_delay + params_.shoot_delay), 
                timestamp_image, 
                *active_trackers[0],  
                timestamp,
                predict_func_double
            );
            cmd.cmd_yaw = ball_res.yaw;
            cmd.cmd_pitch = ball_res.pitch;
        }
        
        else{
            ArmorTracker* best_tracker = nullptr;
            double yaw_need_min = 1.05;
            double yaw = 0.0;
            double pitch_need = 0.0;
            
            for (ArmorTracker* tracker : active_trackers) {
                auto [ball_res, p] = calc_ballistic_(
                    (params_.response_delay + params_.shoot_delay), 
                    timestamp_image, 
                    *tracker, 
                    timestamp,
                    predict_func_double
                );
                if(abs(ball_res.yaw - yaw_now) < abs(yaw_need_min)){
                    yaw_need_min = ball_res.yaw - yaw_now;
                    pitch_need = ball_res.pitch;
                    yaw = ball_res.yaw;
                }
                
            }
            if(abs(yaw_need_min - yaw_now) < 0.79){
                cmd.cmd_yaw = yaw;
                cmd.cmd_pitch = pitch_need;
            }else{
                cmd.cmd_mode = -1;
            }
        }
    }
}
std::pair<Ballistic::BallisticResult, Eigen::Vector3d> EnemyPredictorNode::calc_ballistic_
            (double delay, rclcpp::Time timestamp_image, ArmorTracker& tracker, double timestamp,
                std::function<Eigen::Vector3d(ArmorTracker&, double, double)> _predict_func)
{

    Ballistic::BallisticResult ball_res;
    Eigen::Vector3d predict_pos_odom;
    Eigen::Isometry3d odom2gimbal_transform = getTrans("odom", "gimbal", timestamp_image);
    double t_fly = 0.0;  // 飞行时间（迭代求解）

    rclcpp::Time tick = this->now();

    for (int i = 0; i < 6; ++i) {
        rclcpp::Time tock = this->now();

        rclcpp::Duration elapsed = tock - tick;
        
        double latency = delay + elapsed.seconds();

        double code_dt = tock.seconds() - timestamp_image.seconds();
        //RCLCPP_INFO(this->get_logger(), "latency time: %.6f", latency);
        predict_pos_odom = _predict_func(tracker, t_fly + latency + code_dt, timestamp);
    } 
    ball_res = bac.final_ballistic(odom2gimbal_transform, predict_pos_odom);

    if (ball_res.fail) {
        RCLCPP_WARN(get_logger(), "[calc Ballistic] too far to hit it\n");
        return {ball_res, predict_pos_odom};
    }
    t_fly = ball_res.t;

    // 考虑自身z的变化
    //Eigen::Vector3d z_vec = Eigen::Vector3d::Zero();
    // address it later!!!!!!!!!!!!!!!!!!!
    //z_vec << 0, 0, cmd.robot.z_velocity * (params_.shoot_delay + t_fly);

    //ball_res = bac.final_ballistic(odom2gimbal_transform, predict_pos_odom - z_vec);
    //RCLCPP_DEBUG(get_logger(), "calc_ballistic: predict_pos_odom: %f %f %f", predict_pos_odom(0), predict_pos_odom(1), predict_pos_odom(2));
    return {ball_res, predict_pos_odom};
}
Eigen::Vector3d EnemyPredictorNode::FilterManage(Enemy &enemy, double dt, ArmorTracker& tracker, double timestamp){
    
    Eigen::Vector3d xyyaw_pre_ekf = tracker.ekf.predict_position(dt);
    ZEKF::Vz z_pre = tracker.zekf.predict_position(dt);              // z的处理后面再调，用z_ekf or 均值滤波
    Eigen::Vector3d xyz_ekf_pre = Eigen::Vector3d(xyyaw_pre_ekf[0], xyyaw_pre_ekf[1], z_pre[0]);
    //RCLCPP_INFO(get_logger(), "xyz_ekf_pre : %lf, %lf, %lf",xyz_ekf_pre(0),xyz_ekf_pre(1),xyz_ekf_pre(2));
    visualizeAimCenter(xyz_ekf_pre, cv::Scalar(0, 255, 0));
    //if(!enemy.radius_cal){
    //   return xyz_ekf_pre;
    //}
    Eigen::Vector3d xyz_pre_ckf = enemy.enemy_ckf.predictArmorPosition(enemy.center(2), tracker.phase_id, dt, timestamp);
    //RCLCPP_INFO(get_logger(), "xyz_pre_ckf : %lf, %lf, %lf",xyz_pre_ckf(0),xyz_pre_ckf(1),xyz_pre_ckf(2));
    Eigen::Vector3d enemy_xyz = Eigen::Vector3d(enemy.enemy_ckf.Xe(0),enemy.enemy_ckf.Xe(2), z_pre[0]);
    //RCLCPP_INFO(get_logger(), "enemy_center_ckf : %lf, %lf, %lf",enemy_xyz(0),enemy_xyz(1),enemy_xyz(2));
    for(int i = 0; i < 8; i++){
        RCLCPP_INFO(get_logger(), "Enemy_CKF.Xe(%d) = %lf", i, enemy.enemy_ckf.Xe(i));
    }
    visualizeAimCenter(xyz_pre_ckf, cv::Scalar(255, 0, 0));
    visualizeAimCenter(enemy_xyz, cv::Scalar(0, 255, 255));   // For DEBUG
    
    double k = 1.0;
    double r0 = 1.0;

    double v_x = tracker.ekf.Xe(3);
    double v_y = tracker.ekf.Xe(4);
    double v = std::sqrt(v_x * v_x + v_y * v_y);
    double V = std::max(v, 0.01);

    double omega = tracker.ekf.Xe(5);

    double r = std::abs(omega)/V;

    // exp(k * (r/r0 - 1))
    double exponent = k * (r / r0 - 1.0);
    double exp_value = std::exp(exponent);
    
    // 分母: 1 + exp(k * (r/r0 - 1))
    double denominator = 1.0 + exp_value;
    
    //EKF权重系数: 1 / (1 + exp(k * (r/r0 - 1)))
    double w_ekf = 1.0 / denominator;
    
    // 5. 计算CKF权重系数: exp(k * (r/r0 - 1)) / (1 + exp(k * (r/r0 - 1)))
    double w_ckf = exp_value / denominator;
    
    Eigen::Vector3d fusion_pre = w_ekf * xyz_ekf_pre + w_ckf * xyz_pre_ckf;
    //RCLCPP_INFO(get_logger(), "fusion_pre : %lf, %lf, %lf",fusion_pre(0),fusion_pre(1),fusion_pre(2));
    visualizeAimCenter(fusion_pre, cv::Scalar(0, 0, 255));

    // =================== 可视化tracker当前位置（紫色小球） ===================
    enemy_markers_.markers.clear();
    visualization_msgs::msg::Marker tracker_marker;
    tracker_marker.header.frame_id = "odom";
    tracker_marker.header.stamp = this->now();
    tracker_marker.ns = "tracker_current";
    tracker_marker.id = enemy.class_id * 1000 + tracker.tracker_idx * 10; // 唯一ID
    
    tracker_marker.type = visualization_msgs::msg::Marker::SPHERE;
    tracker_marker.action = visualization_msgs::msg::Marker::ADD;
    
    // 使用tracker当前位置
    tracker_marker.pose.position.x = tracker.position.x();
    tracker_marker.pose.position.y = tracker.position.y();
    tracker_marker.pose.position.z = tracker.position.z();
    tracker_marker.pose.orientation.w = 1.0;
    
    tracker_marker.scale.x = 0.05;  // 稍微小一点，与装甲板区分
    tracker_marker.scale.y = 0.05;
    tracker_marker.scale.z = 0.05;
    
    // 紫色表示tracker当前位置
    tracker_marker.color.r = 0.8;   // 紫色：红+蓝
    tracker_marker.color.g = 0.0;
    tracker_marker.color.b = 0.8;
    tracker_marker.color.a = 0.9;
    
    tracker_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    enemy_markers_.markers.push_back(tracker_marker);

    // =================== 可视化tracker朝向（箭头） ===================
    visualization_msgs::msg::Marker yaw_marker;
    yaw_marker.header.frame_id = "odom";
    yaw_marker.header.stamp = this->now();
    yaw_marker.ns = "tracker_yaw";
    yaw_marker.id = enemy.class_id * 1000 + tracker.tracker_idx * 10 + 1; // 唯一ID
    
    yaw_marker.type = visualization_msgs::msg::Marker::ARROW;
    yaw_marker.action = visualization_msgs::msg::Marker::ADD;
    
    // 箭头的起始点（tracker当前位置）
    geometry_msgs::msg::Point start_point;
    start_point.x = tracker.position.x();
    start_point.y = tracker.position.y();
    start_point.z = tracker.position.z();
    
    // 箭头的结束点（根据yaw计算方向）
    geometry_msgs::msg::Point end_point;
    double arrow_length = 0.3; // 箭头长度
    double yaw_rad = tracker.yaw; // tracker的yaw（弧度）
    end_point.x = start_point.x - arrow_length * cos(yaw_rad);
    end_point.y = start_point.y + arrow_length * sin(yaw_rad);
    end_point.z = start_point.z; // 保持在相同高度
    
    yaw_marker.points.push_back(start_point);
    yaw_marker.points.push_back(end_point);
    
    yaw_marker.scale.x = 0.02;  // 箭头杆直径
    yaw_marker.scale.y = 0.04;  // 箭头头直径
    yaw_marker.scale.z = 0.1;   // 箭头头长度

     // 橙色表示朝向箭头
    yaw_marker.color.r = 1.0;
    yaw_marker.color.g = 0.5;
    yaw_marker.color.b = 0.0;
    yaw_marker.color.a = 0.9;
    
    yaw_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    enemy_markers_.markers.push_back(yaw_marker);
    // =================== 添加EKF滤波器可视化到rviz ===================
    visualization_msgs::msg::Marker ekf_armor_marker;
    ekf_armor_marker.header.frame_id = "odom";
    ekf_armor_marker.header.stamp = this->now();
    ekf_armor_marker.ns = "filter_results";
    ekf_armor_marker.id = enemy.class_id * 100 + 3;  // EKF使用不同的ID
    
    ekf_armor_marker.type = visualization_msgs::msg::Marker::SPHERE;
    ekf_armor_marker.action = visualization_msgs::msg::Marker::ADD;
    
    ekf_armor_marker.pose.position.x = xyz_ekf_pre.x();
    ekf_armor_marker.pose.position.y = xyz_ekf_pre.y();
    ekf_armor_marker.pose.position.z = xyz_ekf_pre.z();
    ekf_armor_marker.pose.orientation.w = 1.0;
    
    ekf_armor_marker.scale.x = 0.06;  // 装甲板尺寸
    ekf_armor_marker.scale.y = 0.06;
    ekf_armor_marker.scale.z = 0.06;
    
    // 设置颜色：绿色表示EKF预测的装甲板位置
    ekf_armor_marker.color.r = 0.0;
    ekf_armor_marker.color.g = 1.0;
    ekf_armor_marker.color.b = 0.0;
    ekf_armor_marker.color.a = 0.9;
    
    ekf_armor_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    enemy_markers_.markers.push_back(ekf_armor_marker);
    
    // 1. 可视化CKF估计的敌人中心（enemy_xyz）
    visualization_msgs::msg::Marker ckf_center_marker;
    ckf_center_marker.header.frame_id = "odom";  // 与之前的坐标系一致
    ckf_center_marker.header.stamp = this->now();
    ckf_center_marker.ns = "filter_results";
    ckf_center_marker.id = enemy.class_id * 100 + 1;  // 使用不同的ID范围
    
    ckf_center_marker.type = visualization_msgs::msg::Marker::SPHERE;
    ckf_center_marker.action = visualization_msgs::msg::Marker::ADD;
    
    ckf_center_marker.pose.position.x = enemy_xyz.x();
    ckf_center_marker.pose.position.y = enemy_xyz.y();
    ckf_center_marker.pose.position.z = enemy_xyz.z();
    ckf_center_marker.pose.orientation.w = 1.0;
    
    ckf_center_marker.scale.x = 0.08;  // 比实际敌人中心小一点
    ckf_center_marker.scale.y = 0.08;
    ckf_center_marker.scale.z = 0.08;
    
    // 设置颜色：青色表示CKF估计的敌人中心
    ckf_center_marker.color.r = 1.0;
    ckf_center_marker.color.g = 1.0;
    ckf_center_marker.color.b = 0.0;
    ckf_center_marker.color.a = 0.9;
    
    ckf_center_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    enemy_markers_.markers.push_back(ckf_center_marker);
    
    visualization_msgs::msg::Marker ckf_armor_marker;
    ckf_armor_marker.header.frame_id = "odom";
    ckf_armor_marker.header.stamp = this->now();
    ckf_armor_marker.ns = "filter_results";
    ckf_armor_marker.id = enemy.class_id * 100 + 2;  // 不同ID
    
    ckf_armor_marker.type = visualization_msgs::msg::Marker::SPHERE;
    ckf_armor_marker.action = visualization_msgs::msg::Marker::ADD;
    
    ckf_armor_marker.pose.position.x = xyz_pre_ckf.x();
    ckf_armor_marker.pose.position.y = xyz_pre_ckf.y();
    ckf_armor_marker.pose.position.z = xyz_pre_ckf.z();
    ckf_armor_marker.pose.orientation.w = 1.0;
    
    ckf_armor_marker.scale.x = 0.06;  // 装甲板尺寸比中心小
    ckf_armor_marker.scale.y = 0.06;
    ckf_armor_marker.scale.z = 0.06;
    
    // 设置颜色：蓝色表示CKF预测的装甲板位置
    ckf_armor_marker.color.r = 0.0;
    ckf_armor_marker.color.g = 0.0;
    ckf_armor_marker.color.b = 1.0;  // 纯蓝色
    ckf_armor_marker.color.a = 0.9;
    
    ckf_armor_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    enemy_markers_.markers.push_back(ckf_armor_marker);

    visualization_msgs::msg::Marker fusion_marker;
    fusion_marker.header.frame_id = "odom";
    fusion_marker.header.stamp = this->now();
    fusion_marker.ns = "fusion_results";
    fusion_marker.id = enemy.class_id * 100 + 4;  // 新的ID，确保不冲突
    
    fusion_marker.type = visualization_msgs::msg::Marker::SPHERE;
    fusion_marker.action = visualization_msgs::msg::Marker::ADD;
    
    fusion_marker.pose.position.x = fusion_pre.x();
    fusion_marker.pose.position.y = fusion_pre.y();
    fusion_marker.pose.position.z = fusion_pre.z();
    fusion_marker.pose.orientation.w = 1.0;
    
    // 设置稍大的尺寸以突出显示融合结果
    fusion_marker.scale.x = 0.07;
    fusion_marker.scale.y = 0.07;
    fusion_marker.scale.z = 0.07;
    
    // 设置颜色：红色表示融合结果
    fusion_marker.color.r = 1.0;  // 红色
    fusion_marker.color.g = 0.0;
    fusion_marker.color.b = 0.0;
    fusion_marker.color.a = 1.0;  // 不透明度设为1.0，使其更显眼
    
    fusion_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    enemy_markers_.markers.push_back(fusion_marker);
   
    return fusion_pre;

}
//--------------------------------------------TOOL----------------------------------------------------------------

void EnemyPredictorNode::create_new_tracker(const Detection &detection,double timestamp, std::vector<int>& active_armor_idx) {

    ArmorTracker new_tracker;
    new_tracker.position = detection.position;
    new_tracker.last_position = detection.position;
    new_tracker.yaw = detection.yaw;
    new_tracker.area_2d = detection.area_2d;
    new_tracker.armor_class_id = detection.armor_class_id;
    new_tracker.tracker_idx = detection.armor_idx;
    new_tracker.is_active = true;
    new_tracker.ekf.init(detection.position, timestamp);
    new_tracker.zekf.init(detection.position(2), timestamp);
    
    enemies_[new_tracker.armor_class_id % 10 - 1].is_active = true;

    active_armor_idx.push_back(armor_trackers_.size()); // 先后push_back的顺序
    
    armor_trackers_.push_back(new_tracker);
    //findBestPhaseForEnemy(enemies_[new_tracker.armor_class_id - 1], new_tracker, active_armor_idx);
}
//int EnemyPredictorNode::estimatePhaseFromPosition(const Enemy& enemy, const ArmorTracker& tracker) {
//    
//    double relative_angle = normalize_angle(tracker.yaw - enemy.yaw);
//    
//    int phase = 0;
//    if (relative_angle >= -M_PI/4 && relative_angle < M_PI/4) {
//        phase = 0;  // 前
//    } else if (relative_angle >= M_PI/4 && relative_angle < 3*M_PI/4) {
//        phase = 1;  // 右
//    } else if (relative_angle >= -3*M_PI/4 && relative_angle < -M_PI/4) {
//        phase = 3;  // 左
//    } else {
//        phase = 2;  // 后
//    }
//    
//    //RCLCPP_INFO(this->get_logger(), 
//    //            "estimatePhaseFromPosition: tracker %d at (%.2f,%.2f), "
//    //            "enemy center (%.2f,%.2f), enemy_yaw=%.1f°, "
//    //            "rel_angle=%.1f° -> phase=%d",
//    //            tracker.tracker_idx,
//    //            tracker.position.x(), tracker.position.y(),
//    //            enemy.center.x(), enemy.center.y(),
//    //            enemy.yaw * 180.0 / M_PI,
//    //            relative_angle * 180.0 / M_PI,
//    //            phase);
//    
//    return phase;
//}
double EnemyPredictorNode::normalize_angle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle <= -M_PI) angle += 2 * M_PI;
    return angle;
}

double EnemyPredictorNode::angle_difference(double a, double b) {
    double diff = normalize_angle(a - b);
    if (diff > M_PI) diff -= 2 * M_PI;
    return diff;
}

// 可视化：aim center
void EnemyPredictorNode::visualizeAimCenter(const Eigen::Vector3d& armor_odom, 
                                           const cv::Scalar& point_color) {
    if (visualize_.armor_img.empty()) {
        RCLCPP_WARN(this->get_logger(), "armor_img is empty, skipping visualization");
        return;
    }
    
    // 检查图像是否有效
    if (visualize_.armor_img.cols <= 0 || visualize_.armor_img.rows <= 0) {
        RCLCPP_WARN(this->get_logger(), "armor_img has invalid size, skipping visualization");
        return;
    }
    try{
            // 1. 将odom系坐标转换到相机系
    Eigen::Vector3d aim_center_cam = visualize_.camara_to_odom.inverse() * armor_odom;
    
    // 2. 准备3D点（相机坐标系下）
    std::vector<cv::Point3d> object_points;
    object_points.push_back(cv::Point3d(
        aim_center_cam.x(), 
        aim_center_cam.y(), 
        aim_center_cam.z()
    ));

    // 3. 投影时使用零旋转和零平移（因为点已经在相机坐标系中）
    cv::Mat zero_rvec = cv::Mat::zeros(3, 1, CV_64F);  // 零旋转
    cv::Mat zero_tvec = cv::Mat::zeros(3, 1, CV_64F);  // 零平移

    // 4. 投影3D点到2D图像平面
    std::vector<cv::Point2d> reprojected_points;
    cv::projectPoints(object_points,
                      zero_rvec,                      // 零旋转
                      zero_tvec,                      // 零平移
                      visualize_.camera_matrix,       // 相机内参矩阵
                      visualize_.dist_coeffs,         // 畸变系数
                      reprojected_points);
    
    // 在访问vector前检查
    if (reprojected_points.empty()) {
        RCLCPP_WARN(this->get_logger(), "No reprojected points, skipping visualization");
        return;
    }
    
    cv::Point2d center = reprojected_points[0];
    
    // 检查坐标是否有效（不是NaN或INF）
    if (std::isnan(center.x) || std::isnan(center.y) || 
        std::isinf(center.x) || std::isinf(center.y)) {
        RCLCPP_WARN(this->get_logger(), "Invalid projected coordinates: (%.1f, %.1f)", 
                   center.x, center.y);
        return;
    }

    // 6. 在图像上绘制点
    if (!reprojected_points.empty() && !visualize_.armor_img.empty()) {
        cv::Point2d center = reprojected_points[0];
        
        // 检查点是否在图像范围内
        if (center.x >= 0 && center.x < visualize_.armor_img.cols &&
            center.y >= 0 && center.y < visualize_.armor_img.rows) {
            
            // 绘制指定颜色的圆点
            cv::circle(visualize_.armor_img, center, 5, point_color, -1);   // 实心圆
            
            // 显示3D坐标和颜色信息
            //std::string label = cv::format("(%.1f,%.1f,%.1f) [BGR:%d,%d,%d]", 
            //                              armor_odom.x(), armor_odom.y(), armor_odom.z(),
            //                              (int)point_color[0], (int)point_color[1], (int)point_color[2]);
            //cv::putText(visualize_.armor_img, label, 
            //           cv::Point(center.x + 10, center.y - 10),
            //           cv::FONT_HERSHEY_SIMPLEX, 0.5, 
            //           cv::Scalar(255, 255, 255), 1);  // 白色文字
        } else {
            RCLCPP_WARN(this->get_logger(), 
                       "Projected point (%.1f, %.1f) out of image bounds [%d x %d]", 
                       center.x, center.y, visualize_.armor_img.cols, visualize_.armor_img.rows);
        }
    } else {
        RCLCPP_WARN(this->get_logger(), "No projected points or empty image");
    }
    } catch (const std::exception& e) {
        RCLCPP_ERROR_STREAM(this->get_logger(), 
                    "Cv Project:"  << e.what());
    }
   
}


