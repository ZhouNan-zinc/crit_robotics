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
    RCLCPP_INFO(get_logger(), "armor_cam: %f, %f, %f", det.position[0], det.position[1], det.position[2]);
    
    det.position = visualize_.camara_to_odom * det.position;  //camera to odom
    visualizeAimCenter(det.position, cv::Scalar(225, 0, 225));
    RCLCPP_INFO(get_logger(), "armor_odom: %f, %f, %f", det.position[0], det.position[1], det.position[2]);
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
void EnemyPredictorNode::ToupdateArmors(const std::vector<Detection>& detections, double timestamp) {
    // 先更新missing_frames
    if (detections.empty()) {
        RCLCPP_INFO(get_logger(), "No Armor This Frame");
        return;
    }
    //Hack : For unkonwn Detection Results
    for(size_t i = 0; i < 1; i++){
        RCLCPP_INFO(get_logger(), "detections.size() =%d",detections.size());
        bool has_history_tracker = false;
        for (int i = 0; i < armor_trackers_.size(); i++) {
            
            armor_trackers_[i].is_active = false;
            
            if(armor_trackers_[i].tracker_idx == detections[i].armor_idx){
                active_armor_idx.push_back(i);
                RCLCPP_INFO(get_logger(),"active_armors + 1");
                armor_trackers_[i].update(detections[i].position, detections[i].armor_idx, timestamp, detections[i].yaw);
                active_enemies_idx.push_back(armor_trackers_[i].assigned_enemy_idx);
                //RCLCPP_INFO(this->get_logger(), "update tracker OK");
                //只在create new tracker时assign to enemy or check or every time ???
                //armor_trackers_[i].assigned_enemy_idx = assignToEnemy(armor_trackers_[i], timestamp);
                //RCLCPP_INFO(this->get_logger(), "tracker_idx: %d", tracker.tracker_idx);
                has_history_tracker = true;
                armor_trackers_[i].is_active = true;
            
                // To Visualize
                //RCLCPP_INFO(this->get_logger(), "To Visualize");
                Eigen::Vector3d pos_update = Eigen::Vector3d(armor_trackers_[i].ekf.Xe(0), armor_trackers_[i].ekf.Xe(1), armor_trackers_[i].zekf.Xe(0));
                RCLCPP_INFO(this->get_logger(), "ekf_update:xy: %f, %f, z: %f", pos_update(0), pos_update(1), pos_update(2));
                //visualizeAimCenter(pos_update);
                break;
            }
        }   
        if(!has_history_tracker){
            create_new_tracker(detections[i], timestamp);
        }
    }
}

//---------------------------------Add Armor To Enemy------------------------------------------------
int EnemyPredictorNode::assignToEnemy(ArmorTracker& tracker, double timestamp) {
    // use detector data
    // assign to enemy 后 active_enemies_idx.push_back(enemy.enemy_idx)
    // 1. 如果已经分配，检查是否仍然有效
    if (tracker.assigned_enemy_idx >= 0 && 
        tracker.assigned_enemy_idx < static_cast<int>(enemies_.size())) {
        
        Enemy& enemy = enemies_[tracker.assigned_enemy_idx];
        
        double distance = (tracker.position - enemy.center).norm();
        //double max_distance = robot_2armor_dis_thresh;
        double max_distance = 10.0;
        if (enemy.radius > 0.3) max_distance = enemy.radius * 1.5;
        
        if (distance < max_distance) {
            findBestPhaseForEnemy(enemy, tracker);
            RCLCPP_INFO(this->get_logger(), "assignToEnemy: %d", tracker.assigned_enemy_idx);
            enemies_[tracker.assigned_enemy_idx].is_alive = true;
            active_enemies_idx.push_back(tracker.assigned_enemy_idx);
            return tracker.assigned_enemy_idx;
        } 
        else {
            enemy.remove_armor(tracker.tracker_idx);
            tracker.assigned_enemy_idx = -1;
            tracker.phase_id = -1;
        }
    }
    
    // 2. 寻找最佳敌人
    int best_enemy_idx = -1;
    double best_score = -1.0;
    
    for (size_t enemy_idx = 0; enemy_idx < enemies_.size(); ++enemy_idx) {
        Enemy& enemy = enemies_[enemy_idx];
        
        if (enemy.armor_tracker_ids.size() >= 4) continue;
        if (enemy.type != tracker.tracker_idx) continue;
        
        double distance = (tracker.position - enemy.center).norm();
        double max_dist = robot_2armor_dis_thresh;
        if (enemy.radius > 0.3) max_dist = enemy.radius * 1.5;
        
        double dist_score = std::max(0.0, 1.0 - distance / max_dist);
        double count_score = enemy.armor_tracker_ids.size() / 4.0;
        
        double score = 0.7 * dist_score + 0.3 * count_score;
        
        if (score > best_score) {
            best_score = score;
            best_enemy_idx = enemy_idx;
        }
    }
    
    // 3. 分配/创建敌人
    if (best_enemy_idx >= 0 && best_score > 0.3) {
        Enemy& enemy = enemies_[best_enemy_idx];
        findBestPhaseForEnemy(enemy, tracker);
        active_enemies_idx.push_back(best_enemy_idx);
        return best_enemy_idx;
    }
    
    // 4. 创建新敌人
    else{
        Enemy new_enemy;
        
        tracker.assigned_enemy_idx = enemies_.size();
        // 如果只detect到一个armor,还要投到整车吗，直接瞄这一块armor效果会不会更好？？？
        new_enemy.center = tracker.position;  // calulate it later
        new_enemy.yaw = tracker.yaw;  // enemy.yaw = its first armor yaw
        new_enemy.alive_ts = timestamp;
        new_enemy.type = tracker.armor_class_id;
        new_enemy.enemy_idx = enemies_.size();
        new_enemy.enemy_ckf.initializeCKF();
        new_enemy.enemy_ckf.reset(new_enemy.center, new_enemy.yaw, 0, 
                              4, timestamp, 
                              std::vector<double>(4, 0.28),                     
                              std::vector<double>(4, new_enemy.center(2))); 
        new_enemy.is_alive = true;
        //new_enemy.armor_tracker_ids.push_back(tracker.tracker_idx);
        findBestPhaseForEnemy(new_enemy, tracker);
        enemies_.push_back(new_enemy);
   
        RCLCPP_INFO(this->get_logger(),
                   "Created new enemy %d for tracker %d----phase_id:%d",
                   new_enemy.enemy_idx,
                   tracker.tracker_idx,
                   tracker.phase_id);
        active_enemies_idx.push_back(new_enemy.enemy_idx);
        return new_enemy.enemy_idx;
    }
}
void EnemyPredictorNode::EnemyManage(double timestamp, rclcpp::Time timestamp_image) {
    
    int target_enemy_id = -1;
    for (auto& enemy : enemies_) {
        bool update_this_frame = false;
        for(int i = 0; i < enemies_.size(); i++){
           if(enemy.enemy_idx == active_enemies_idx[i]){
              calculateEnemyCenterAndRadius(enemies_[active_enemies_idx[i]], timestamp);
              enemy.is_alive = true;
              update_this_frame = true;
           }
        }
        if(!update_this_frame){
            enemy.is_alive = false;
            enemy.missing_frame++;
            if(enemy.missing_frame >30){
                enemy.is_valid = false;
            }
        }
    }
    RCLCPP_INFO(get_logger(),"active_enemies_idx:%d",active_enemies_idx.size());
    if(active_enemies_idx.size()== 0){
        return;
    }
    //是否需要考虑操作手按right键，但这一帧没有detect到操作手正在tracking的enemy？？？
    if(active_enemies_idx.size()== 1){
        target_enemy_id = active_enemies_idx[0];
        RCLCPP_INFO(get_logger(),"target_enemy_id:%d",target_enemy_id);
    }
    else if(active_enemies_idx.size() > 1){
        RCLCPP_INFO(get_logger(),"active_enemies_idx > 1");
        //基于优先级的决策
        //int target_enemy_id = 0;
        //int type_temp = 100;
        //if(last_target_enemy_id != -1 && right_press == true){
        //    target_enemy_id = last_target_enemy_id;
        //}
        //else{
        //    for(int i = 0; i < acive_enemy_ids.size(); i++){
        //        if(enemies_[acive_enemy_ids[i]].type < type_temp){
        //            target_enemy_id = acive_enemy_ids[i];
        //        }
        //    }
        //}

        //基于到准星距离（操作手）的决策
        double enemy_to_heart_min = 10000;
        if(cmd.last_target_enemy_id != -1 && params_.right_press == true){
            target_enemy_id = cmd.last_target_enemy_id;
        }
        else{
            for(int i = 0; i < active_enemies_idx.size(); i++){
                //如果某个enemy只存在过一个装甲板的detection,使用const_radius但只做ekf得到cmd结果
                //那么此时的enemy.center不准，但是choose enemy的时候需要使用enemy.center
                RCLCPP_INFO(get_logger(),"Start To Choose Target");
                Eigen::Vector3d enemy_center_cam = visualize_.camara_to_odom.inverse() *enemies_[active_enemies_idx[i]].center;
                RCLCPP_INFO(get_logger(), "Enemy center in camera frame: x=%.3f, y=%.3f, z=%.3f", 
                enemy_center_cam.x(), enemy_center_cam.y(), enemy_center_cam.z());

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
                        target_enemy_id = active_enemies_idx[i];

                    }
                }
                target_enemy_id = active_enemies_idx[i];
            }
            RCLCPP_INFO(get_logger(),"Finish To Choose Target");
        }
    }
    getCommand(enemies_[target_enemy_id], timestamp, timestamp_image);
    cmd.last_target_enemy_id = target_enemy_id;
    RCLCPP_INFO(get_logger(),"cmd.last_target_enemy_id:%d",cmd.last_target_enemy_id);
    //armor_trackers_ doesn't be cleared, enemies_索性也先不clear了

    // 清理长时间不更新的敌人
    //auto it = enemies_.begin();
    //while (it != enemies_.end()) {
    //    if (it->empty() || (timestamp - it->alive_ts > 2.0)) {
    //        it = enemies_.erase(it);
    //    } else {
    //        ++it;
    //    }
    //}
}
void EnemyPredictorNode::calculateEnemyCenterAndRadius(Enemy& enemy, double timestamp) {
    // 收集活跃装甲板
    std::vector<ArmorTracker*> active_armors_this_enemy;
    RCLCPP_INFO(this->get_logger(), "armor_trackers_.size = %d",armor_trackers_.size());
    RCLCPP_INFO(this->get_logger(), "active_armor_trackers_.size = %d",active_armor_idx.size());
    for (int idx : active_armor_idx) {
        if(armor_trackers_[idx].assigned_enemy_idx == enemy.enemy_idx){
            active_armors_this_enemy.push_back(&armor_trackers_[idx]);
        }
    }
    RCLCPP_INFO(get_logger(), "active_armors_this_enemy:%d", active_armors_this_enemy.size());
    //RCLCPP_INFO(this->get_logger(), "active_armors[0]->phase_id = %d",active_armors_this_enemy[0]->phase_id);
    if (active_armors_this_enemy.empty()) return;
    
    // 根据装甲板数量采用不同的计算方法
    if (active_armors_this_enemy.size() == 1) {
        // 单个装甲板：基于相位推测中心
        double phase_angle = active_armors_this_enemy[0]->phase_id * (M_PI / 2.0);
        double center_yaw = active_armors_this_enemy[0]->yaw - phase_angle + M_PI;
        enemy.center = active_armors_this_enemy[0]->position + enemy.radius * 
                      Eigen::Vector3d(std::cos(center_yaw), std::sin(center_yaw), 0);  //认为armor_z == enemy_z ? ? ?
    } 
    else if (active_armors_this_enemy.size() == 2) {
        // 两个装甲板：使用法向量交点法
        ArmorTracker& armor1 = *active_armors_this_enemy[0];
        ArmorTracker& armor2 = *active_armors_this_enemy[1];
        
        Eigen::Vector2d p1(armor1.position.x(), armor1.position.y());
        Eigen::Vector2d p2(armor2.position.x(), armor2.position.y());
        
        Eigen::Vector2d dir1(-std::cos(armor1.yaw), -std::sin(armor1.yaw));
        Eigen::Vector2d dir2(-std::cos(armor2.yaw), -std::sin(armor2.yaw));
        
        Eigen::Matrix2d A;
        A << dir1.x(), -dir2.x(),
             dir1.y(), -dir2.y();
        
        Eigen::Vector2d b = p2 - p1;
        
        double det = A.determinant();
        
        if (std::abs(det) > 1e-6) {
            Eigen::Vector2d t = A.inverse() * b;
            
            if (t(0) > 0 && t(1) > 0) {
                Eigen::Vector2d center_2d = p1 + t(0) * dir1;
                double z = (armor1.position.z() + armor2.position.z()) * 0.5;
                
                enemy.center = Eigen::Vector3d(center_2d.x(), center_2d.y(), z);
                
                double actual_radius1 = (enemy.center - armor1.position).norm();
                double actual_radius2 = (enemy.center - armor2.position).norm();
                double avg_radius = (actual_radius1 + actual_radius2) * 0.5;
                
                enemy.radius = 0.8 * enemy.radius + 0.2 * avg_radius;
                enemy.radius_cal = true;
                RCLCPP_INFO(get_logger(), "Calculate Radius Is TRUE!");
            } else {
                // 交点不在正向，使用几何中心
                useGeometricCenterSimple(enemy, active_armors_this_enemy);
            }
        } else {
            // 行列式为0，方向平行，使用几何中心
            useGeometricCenterSimple(enemy, active_armors_this_enemy);
        }
    }
    else {
        // 多个装甲板：几何中心(no possiblity)[Acutally, in our super new imagepipe, that is possible.]
        useGeometricCenterSimple(enemy, active_armors_this_enemy);
    }
    if(!active_armors_this_enemy.empty() && active_armors_this_enemy[0] != nullptr) {
    RCLCPP_INFO(this->get_logger(), "Enemy %d: center = (%.2f, %.2f, %.2f), yaw = %.2f, phase_id = %d",
                enemy.enemy_idx, enemy.center(0), enemy.center(1), enemy.center(2), 
                enemy.yaw, active_armors_this_enemy[0]->phase_id);
    } else {
        RCLCPP_INFO(this->get_logger(), "Enemy %d: active_armors_this_enemy is empty or null", enemy.enemy_idx);
    }
    for(int i = 0; i < active_armors_this_enemy.size(); i++){
        enemy.enemy_ckf.update(enemy.center, enemy.yaw, timestamp, active_armors_this_enemy[i]->phase_id);
    }
    //visualizeAimCenter(enemy.center);
    //RCLCPP_INFO(this->get_logger(), "Enemy %d: center = (%.2f, %.2f, %.2f), yaw = %.2f",enemy.center(0), enemy.center(1), enemy.center(2), enemy.yaw);
}

void EnemyPredictorNode::useGeometricCenterSimple(Enemy& enemy, 
                                                 const std::vector<ArmorTracker*>& active_armors) {
    Eigen::Vector3d sum_pos = Eigen::Vector3d::Zero();
    for (auto armor : active_armors) {
        sum_pos += armor->position;
    }
    enemy.center = sum_pos / active_armors.size();

    double sum_radius = 0.0;
    for (auto armor : active_armors) {
        sum_radius += (enemy.center - armor->position).norm();
    }
    double avg_radius = sum_radius / active_armors.size();
    enemy.radius = 0.5 * enemy.radius + 0.5 * avg_radius;
    enemy.radius_cal = true;
    RCLCPP_INFO(get_logger(), "Calculate Radius Is TRUE!");
}
//Phase_id 处理1
//void EnemyPredictorNode::updateArmorPhase(Enemy& enemy, ArmorTracker& tracker, double timestamp) {
//    // 1.没有其他tracker，基于R估计相位
//    if (enemy.armor_tracker_ids.size() <= 1) {
//        tracker.phase_id = estimatePhaseFromPosition(enemy, tracker);
//        RCLCPP_DEBUG(this->get_logger(), 
//                    "Single tracker %d: assigned phase %d from position",
//                    tracker.tracker_idx, tracker.phase_id);
//        return;
//    }
//    
//    // 找到最近的活跃装甲板（时间上）
//    ArmorTracker* nearest_armor = nullptr;
//    double nearest_ts = 0.01;  //  adjust it later !!!
//    
//    for (int tracker_id = 0; i < enemy.armor_tracker_ids.size(); ++tracker_id) {
//        if (tracker_id == tracker.tracker_idx) continue;
//        
//        if (armor_trackers_[tracker_id].is_alive){
//            nearest_armor = &armor_trackers_[tracker_id]; //如果有同样active状态的armor_tracker，那么必然是nearest
//        }
//        else if(armor_trackers_[tracker_id].is_valid){
//            //给一个很小的间隔，在该间隔下armor移动距离有限，可以用来估计其他新armor的phase_id
//            // 想起来有一定合理性，不知道测出来效果如何 
//            if(timestamp - armor_trackers_[tracker_id].last_update_time < nearest_ts){
//                nearest_armor = &armor_trackers_[tracker_id]; 
//            }
//        }
//    }
//    
//    if (!nearest_armor) {
//        tracker.phase_id = estimatePhaseFromPosition(enemy, tracker);
//        return;
//    }
//    else{
//        double angle_diff = normalize_angle(tracker.yaw - nearest_armor->yaw);
//        int phase_diff = static_cast<int>(std::round(angle_diff / (M_PI / 2)));
//        int new_phase = (nearest_armor->phase_id + phase_diff) % 4;
//        
//        if (new_phase < 0) new_phase += 4;
//        
//        if (new_phase < 0 || new_phase >= 4) {
//            RCLCPP_WARN(this->get_logger(), 
//                   "Invalid calculated phase %d for tracker %d, using position-based estimation",
//                   new_phase, tracker.tracker_idx);
//            tracker.phase_id = estimatePhaseFromPosition(enemy, tracker);
//        } 
//        else {
//            tracker.phase_id = new_phase;
//            RCLCPP_DEBUG(this->get_logger(), 
//                        "Tracker %d: phase %d (based on nearest tracker %d phase %d, angle_diff=%.1f°)",
//                        tracker.tracker_idx, tracker.phase_id,
//                        nearest_armor->tracker_idx, nearest_armor->phase_id,
//                        angle_diff * 180.0 / M_PI);
//        }
//    }
    // 基于左右关系确定相位
    //bool is_left = check_left(tracker.position, nearest_armor->position);
    //
    //if (is_left) {
    //    tracker.phase_id = (nearest_armor->phase_id - 1 + enemy.armor_cnt) % enemy.armor_cnt;
    //} else {
    //    tracker.phase_id = (nearest_armor->phase_id + 1) % enemy.armor_cnt;
    //}
//}
//Phase_id 处理2
    
void EnemyPredictorNode::findBestPhaseForEnemy(Enemy& enemy, ArmorTracker& tracker) {
    
    if (enemy.armor_tracker_ids.empty()) {
        // 第一个tracker，基于旋转矩阵估计相位
        tracker.phase_id = estimatePhaseFromPosition(enemy, tracker);  
        RCLCPP_INFO(this->get_logger(), 
                    "Tracker %d: phase %d (based on position)",
                    tracker.tracker_idx, tracker.phase_id);
    }
    else{
        // 收集已使用的相位
        std::set<int> used_phases;
        for(int i = 0; i < armor_trackers_.size(); i++){
            if(armor_trackers_[i].assigned_enemy_idx == enemy.enemy_idx && 
               armor_trackers_[i].is_active == true){
                used_phases.insert(armor_trackers_[i].phase_id);
            }
        }
        double min_cost = 1000.0;
        double second_min_cost = 1000.0;
        int best_phase = -1;
        int second_best_phase = -1;
        // 简易匈牙利算法
        for (int phase = 0; phase < 4; ++phase) {
    
            double total_cost = 0.0;
    
            double phase_angle = phase * (M_PI / 2.0);
            double expected_yaw = enemy.yaw + phase_angle - M_PI;
            
            Eigen::Vector3d expected_pos = enemy.center + enemy.radius * 
                                          Eigen::Vector3d(std::cos(expected_yaw), 
                                                         std::sin(expected_yaw), 
                                                         tracker.position.z());
            double distance = (tracker.position - expected_pos).norm();
        
            // 使用sigmoid函数归一化到[0, 1]
            // 参数：0.3表示30cm时成本为0.5  adjust it later !!!
            // 1.position cost
            double normalized_cost = 1.0 / (1.0 + exp(-10.0 * (distance - 0.3)));
    
            total_cost += 0.5 *normalized_cost;
    
            double angle_diff = std::atan2(std::sin(expected_yaw - tracker.yaw),
                                       std::cos(expected_yaw - tracker.yaw));
        
            // 取绝对值并归一化到[0, 1]
            // 2.yaw cost
            double normalized_diff = std::abs(angle_diff) / M_PI;
    
            total_cost += 0.3 *normalized_diff* normalized_diff;
            // 3.相位冲突 cost
            if (used_phases.find(phase) == used_phases.end()) {
                total_cost += 0.2 * 0.0;
            }else{
                total_cost += 0.2 * 1.0;
            }
            if (phase == tracker.phase_id){
                total_cost*= 0.5;
            }
            if(total_cost < min_cost){
                second_min_cost = min_cost;
                second_best_phase = best_phase;
    
                min_cost = total_cost;
                best_phase = phase;
            }else{
                second_min_cost = total_cost;
                second_best_phase = phase;
            }
        }
        tracker.phase_conf = 1 - min_cost;
        // 一致性奖励
        double expected_yaw = enemy.yaw + best_phase * (M_PI / 2.0) - M_PI;
        Eigen::Vector3d expected_pos = enemy.center + enemy.radius * 
                                      Eigen::Vector3d(std::cos(expected_yaw), 
                                                     std::sin(expected_yaw), 
                                                     tracker.position.z());
        double position_error = (tracker.position - expected_pos).norm();
        
        if (position_error < 0.1) { // 10cm以内认为很匹配  adjust it later !!!
            tracker.phase_conf = std::min(tracker.phase_conf + 0.1, 1.0);
        }
        
        if(used_phases.find(best_phase) == used_phases.end()){
            tracker.phase_id = best_phase;
        }else{
            for(int idx : active_armor_idx){
                if(armor_trackers_[idx].assigned_enemy_idx == enemy.enemy_idx && armor_trackers_[idx].phase_id == best_phase){
                    if(armor_trackers_[idx].phase_conf < tracker.phase_conf){
                       tracker.phase_id = best_phase;
                       // 冲突的tracker, 若conf不如新armor, phase_id设置为-1
                       armor_trackers_[idx].phase_id = -1;
                    }else{
                       tracker.phase_id = second_best_phase;
                    }
                }
            }
        }
    }
    if(tracker.phase_id != -1){
        enemy.armor_tracker_ids.push_back(tracker.tracker_idx);  // assigned to enemy 后再push_back，否则第一个板就会走第二个if
        RCLCPP_INFO(this->get_logger(),"enemy has push back new tracker");   // debug OK
    }
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
    if(enemy.enemy_ckf.Xe(5) > cmd.high_spd_rotate_thresh){
       return 1;
    }else{
       return 0;
    }
}
void EnemyPredictorNode::getCommand(Enemy& enemy, double timestamp, rclcpp::Time timestamp_image){
    
    cmd.cmd_mode = ChooseMode(enemy, timestamp);
    RCLCPP_INFO(get_logger(),"MODE :%d",cmd.cmd_mode);
    cmd.target_enemy_id = enemy.type;
    auto predict_func_double = [this, &enemy](ArmorTracker& tracker, double time_offset) -> Eigen::Vector3d{
        return FilterManage(enemy, time_offset, tracker);
    };
    std::vector<ArmorTracker*> active_trackers;
    for(int idx : active_armor_idx){
        if(armor_trackers_[idx].assigned_enemy_idx == enemy.enemy_idx){
            active_trackers.push_back(&armor_trackers_[idx]);
        }
    }
    if(cmd.cmd_mode == 1){
        
        Eigen::Vector3d armor_center_pre = Eigen::Vector3d(0, 0, 0);
        std::vector<double> yaws(4);
        //use enemy.center or enemy.center_pre ???
        //一直瞄准yaw近似车体中心的位置，but pitch不等于整车中心，当预测某一装甲板即将旋转到该直线时给发弹指令
        double enemy_yaw_xy = std::atan2(enemy.center_pre[1], enemy.center_pre[0]); 
        
        for(int i = 0; i < active_trackers.size(); i++){
           auto [ball_res, p] = calc_ballistic_((params_.response_delay + params_.shoot_delay), timestamp_image, *active_trackers[i], predict_func_double);
           
           armor_center_pre = FilterManage(enemy, ball_res.t, *active_trackers[i]);

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
                predict_func_double
            );
            cmd.aim_center = FilterManage(enemy, ball_res.t, *active_trackers[0]);
            cmd.cmd_yaw = ball_res.yaw;
            cmd.cmd_pitch = ball_res.pitch;
        }
        
        else{
            double S_max = 0.0;
            ArmorTracker* best_tracker = nullptr;
            
            for (ArmorTracker* tracker : active_trackers) {
                if(tracker->area_2d > S_max){
                    S_max = tracker->area_2d;
                    best_tracker = tracker;
                }
            }
            
            if (!best_tracker) {
                RCLCPP_WARN(this->get_logger(), "No best tracker found");
                return;
            }
            
            enemy.best_armor = best_tracker->phase_id;  // 存储phase_id而非索引
            enemy.best_armor_idx = best_tracker->tracker_idx;

            auto [ball_res, p] = calc_ballistic_(
                (params_.response_delay + params_.shoot_delay), 
                timestamp_image, 
                *best_tracker, 
                predict_func_double
            );
            cmd.aim_center = FilterManage(enemy, ball_res.t, *best_tracker);
            cmd.cmd_yaw = ball_res.yaw;
            cmd.cmd_pitch = ball_res.pitch;
        }
    }
    RCLCPP_INFO_STREAM(get_logger(), "cmd.cmd_yaw:" << cmd.cmd_yaw);
    RCLCPP_INFO_STREAM(get_logger(), "cmd.cmd_pitch:" << cmd.cmd_pitch);
}
std::pair<Ballistic::BallisticResult, Eigen::Vector3d> EnemyPredictorNode::calc_ballistic_
            (double delay, rclcpp::Time timestamp_image, ArmorTracker& tracker, std::function<Eigen::Vector3d(ArmorTracker&, double)> _predict_func)
{

    Ballistic::BallisticResult ball_res;
    Eigen::Vector3d predict_pos_odom;
    Eigen::Isometry3d odom2gimbal_transform = getTrans("odom", "gimbal", timestamp_image);
    double t_fly = 0.0;  // 飞行时间（迭代求解）

    auto tick = std::chrono::steady_clock::now();
    auto tick_duration = tick.time_since_epoch();
    double tick_seconds = std::chrono::duration<double>(tick_duration).count();

    for (int i = 0; i < 6; ++i) {
        auto tock = std::chrono::steady_clock::now();
        auto tock_duration = tock.time_since_epoch();
        double tock_seconds = std::chrono::duration<double>(tock_duration).count();
        // 计算时间差（秒）
        std::chrono::duration<double> elapsed = tock - tick;
        double latency = delay + elapsed.count();
        RCLCPP_INFO(this->get_logger(), "latency time: %.6f", latency);
        predict_pos_odom = _predict_func(tracker, t_fly + latency);
    } 
    
    RCLCPP_INFO_STREAM(get_logger(), "predict_pos_odom: " << predict_pos_odom(0) << ", " << predict_pos_odom(1) << ", " << predict_pos_odom(2));
    RCLCPP_INFO_STREAM(rclcpp::get_logger("Ballistic"), "odom2gimbal_transform:\n" << odom2gimbal_transform.matrix());
    visualizeAimCenter(predict_pos_odom, cv::Scalar(0, 0, 255));
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
Eigen::Vector3d EnemyPredictorNode::FilterManage(Enemy &enemy, double dt, ArmorTracker& tracker){
    
    Eigen::Vector3d xyyaw_pre_ekf = tracker.ekf.predict_position(dt);
    ZEKF::Vz z_pre = tracker.zekf.predict_position(dt);              // z的处理后面再调，用z_ekf or 均值滤波
    Eigen::Vector3d xyz_ekf_pre = Eigen::Vector3d(xyyaw_pre_ekf[0], xyyaw_pre_ekf[1], z_pre[0]);
    visualizeAimCenter(xyz_ekf_pre, cv::Scalar(0, 255, 0));
    if(!enemy.radius_cal){
       return xyz_ekf_pre;
    }
    Eigen::Vector3d xyyaw_pre_ckf = enemy.enemy_ckf.predictArmorPosition(enemy.best_armor, dt);
    Eigen::Vector3d enemy_xyz = Eigen::Vector3d(enemy.enemy_ckf.Xe(0),enemy.enemy_ckf.Xe(2), z_pre[0]);
    Eigen::Vector3d xyz_ckf_pre = Eigen::Vector3d(xyyaw_pre_ckf[0], xyyaw_pre_ckf[1], z_pre[0]);

    //visualizeAimCenter(xyz_ckf_pre, cv::Scalar(255, 0, 0));
    visualizeAimCenter(enemy_xyz, cv::Scalar(255, 0, 0));
    
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
    
    Eigen::Vector3d fusion_pre = w_ekf * xyz_ekf_pre + w_ckf * xyz_ckf_pre;
    //visualizeAimCenter(fusion_pre, cv::Scalar(0, 0, 255));
   
    return fusion_pre;

}
//--------------------------------------------TOOL----------------------------------------------------------------
//EnemyPredictorNode::ArmorTracker* EnemyPredictorNode::getActiveArmorTrackerById(int tracker_id) {
//    if(armor_trackers_[tracker_id].is_active){
//        return &armor_trackers_[tracker_id];
//    }
//}
//bool EnemyPredictorNode::check_left(const Eigen::Vector3d& pos1, const Eigen::Vector3d& pos2) {
//    return pos1.x() < pos2.x();
//}
void EnemyPredictorNode::create_new_tracker(const Detection &detection,double timestamp) {

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
    RCLCPP_INFO(this->get_logger(), "Creating new tracker for armor %d", detection.armor_idx);  //DEBUG : OK
    //new_tracker.update(detection.position, detection.armor_idx, timestamp, detection.yaw);
    new_tracker.assigned_enemy_idx = assignToEnemy(new_tracker, timestamp);

    active_armor_idx.push_back(armor_trackers_.size()); // 先后push_back的顺序
    RCLCPP_INFO(get_logger(),"active_armors + 1");
    armor_trackers_.push_back(new_tracker);
}
int EnemyPredictorNode::estimatePhaseFromPosition(const Enemy& enemy, const ArmorTracker& tracker) {
    
    double relative_angle = normalize_angle(tracker.yaw - enemy.yaw);
    
    int phase = 0;
    if (relative_angle >= -M_PI/4 && relative_angle < M_PI/4) {
        phase = 0;  // 前
    } else if (relative_angle >= M_PI/4 && relative_angle < 3*M_PI/4) {
        phase = 1;  // 右
    } else if (relative_angle >= -3*M_PI/4 && relative_angle < -M_PI/4) {
        phase = 3;  // 左
    } else {
        phase = 2;  // 后
    }
    
    //RCLCPP_INFO(this->get_logger(), 
    //            "estimatePhaseFromPosition: tracker %d at (%.2f,%.2f), "
    //            "enemy center (%.2f,%.2f), enemy_yaw=%.1f°, "
    //            "rel_angle=%.1f° -> phase=%d",
    //            tracker.tracker_idx,
    //            tracker.position.x(), tracker.position.y(),
    //            enemy.center.x(), enemy.center.y(),
    //            enemy.yaw * 180.0 / M_PI,
    //            relative_angle * 180.0 / M_PI,
    //            phase);
    
    return phase;
}
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
            cv::circle(visualize_.armor_img, center, 8, point_color, -1);   // 实心圆
            
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


