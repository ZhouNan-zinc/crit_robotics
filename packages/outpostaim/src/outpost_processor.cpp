#include "outpostaim/outpost_node.h"
#include "outpostaim/outpost_estimator.h"
#include <algorithm>
#include <cmath>


//--------------------------------UpdateArmor-----------------------------------------------

void OutpostManager::updateArmors(const std::vector<TrackedArmor>& tracked_armors, double timestamp) {
    static std::vector<OutpostArmor> new_armors;
    new_armors.clear();

    RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Updating armors, tracked count: %zu", tracked_armors.size());
    
    // 仅当处于前哨站自瞄模式时维护相位逻辑；离开模式则清空相位状态
    if (recv_detection.mode != VisionMode::OUTPOST_AIM && recv_detection.mode != VisionMode::AUTO_AIM) {
        phase_initialized_ = false;
        last_active_tracker_id_ = -1;
        last_active_phase_ = -1;
        last_active_yaw_ = 0.0;
        last_active_ts_ = -1.0;
    }

    bool outpost_exist = (outpost.alive_ts > 0) || (!outpost.armors.empty());

    if (outpost_exist) {
        for (int k = 0; k < (int)outpost.armors.size(); ++k) {
            outpost.armors[k].matched_ = false;                         // 清空匹配标志位
            outpost.armors[k].last_status_ = outpost.armors[k].status_;  // 储存上一帧状态
            outpost.armors[k].status_ = Absent;                         // 所有装甲板初始化为暂时离线
        }
    }

    if(self_id.robot_id == RobotId::ROBOT_HERO){
        outpost.is_hero = true;
    }

    // 存储当前帧的装甲板信息，用于下一帧
    std::map<int, TrackedArmor> current_frame_armors;

    for (const auto& tracked_armor : tracked_armors) {


        ArmorPoseResult pc_result;
        pc_result.pose.x = tracked_armor.position_odom[0];
        pc_result.pose.y = tracked_armor.position_odom[1];
        pc_result.pose.z = tracked_armor.position_odom[2];
        // 计算位置 yaw 和 pitch
        Eigen::Vector3d pyd_pos = xyz2pyd(tracked_armor.position_odom);
        pc_result.pose.pitch = pyd_pos[0];
        
        // 计算法向量
        pc_result.normal_vec = tracked_armor.orientation_odom * Eigen::Vector3d::UnitZ();

        // 恢复旧版逻辑：Yaw 采用装甲板法向量的朝向，与旧代码 rm_base/pose_calculator.cpp 保持一致
        Eigen::Vector3d pyd_normal = xyz2pyd(pc_result.normal_vec);
        pc_result.pose.yaw = pyd_normal[1];

        // 其他成员保持默认
        pc_result.reproject_error = 0.0;

        // Armor now_detect_armor = recv_detection.res[i];
        // //     /9:color    %9:type
        // ArmorId now_armor_id(now_detect_armor.type, now_detect_armor.color);

        // 非前哨站装甲板过滤 class_id / 10 为颜色， % 10为类型
        if ((tracked_armor.class_id % 10) != 6) {
            RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Filtered out non-outpost armor (class_id: %d)", tracked_armor.class_id);
            continue;}
        // 过滤：距离过远
        double distance = pyd_pos[2];
        if (distance > params.sight_limit){
            RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Filtered out distant armor (distance: %.2f m), sight_limit: %.2f m", distance, params.sight_limit);
            continue;}
        // 过滤：高度限制
        if (fabs(tracked_armor.position_odom[2]) > params.high_limit){
            RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Filtered out high armor (height: %.2f m)", tracked_armor.position_odom[2]);
            continue;}

        // 同色过滤 待修改 class_id/10 = 1 红色
        //if (now_armor_id.armor_color == self_id.color) continue;
        
        // 滤掉过小的装甲板
        // for (int j = 0; j < 4; ++j) pts[j] = now_detect_armor.pts[j];
        // double pts_S = get_area_armor(now_detect_armor.pts);
        // if (pts_S < params.size_limit) continue;

        // 装甲板宽高比，若过于倾斜，则滤去装甲板
        // 必须使用外接矩形！！
        // fix flip training bug
        // if (pts[3].x + pts[2].x < pts[0].x + pts[1].x) {
        //     std::swap(pts[0], pts[3]);
        //     std::swap(pts[1], pts[2]);
        // }

        // double aspect = (pts[3].x + pts[2].x - pts[0].x - pts[1].x) / (pts[1].y + pts[2].y - pts[0].y - pts[3].y);
        // if (aspect <= params.aspect_limit_small) continue;

        // if (std::any_of(pts.begin(), pts.end(), [=](auto p) {
        //         return p.x < params.bound_limit || p.x > recv_detection.img.cols - params.bound_limit || p.y < params.bound_limit ||
        //                p.y > recv_detection.img.rows - params.bound_limit;
        //     })) {
        //     continue;

        // 过滤：顶装甲板（根据pitch角度）
        // double now_pitch = rad2deg(pc_result.pose.pitch);
        // double top_pitch_thresh = params.top_pitch_thresh;
        // if (-now_pitch > top_pitch_thresh) {
        //     // NGXY_INFO("TOP armor filtered out (pitch: %.1f deg)", now_pitch);
        //     continue;
        // }

        RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Tracked Armor ID: %d, Class ID: %d, Odom xyz: [%.2f, %.2f, %.2f], Odom pyd: [%.2f, %.2f, %.2f]", tracked_armor.id, tracked_armor.class_id,tracked_armor.position_odom[0], tracked_armor.position_odom[1], tracked_armor.position_odom[2], pyd_pos[0], pyd_pos[1], pyd_pos[2]);

        // 这个id是否有问题？
        current_frame_armors[tracked_armor.id] = tracked_armor;

        // 检查是否已有相同ID的装甲板（使用tracker分配的ID进行匹配）
        bool matched_existing = false;
        for (auto& armor : outpost.armors) {
            // 简单的Tracker ID匹配，仅用于帧间跟踪 当前不完全信任track id
            // phase_in_outpost_ 初始值为-1，只有在分配了相位后才有效
            // 先用Tracker ID匹配，如果匹配到，就使用该相位

            // int phase = tracked_armor.id % 3;
            
            if (armor.tracker_id == tracked_armor.id && armor.tracker_id != -1) {
                matched_existing = true;
                
                int phase = armor.phase_in_outpost_;

                // === 过中检测逻辑 ===
                // 检查是否有上一帧信息
                if (last_frame_timestamp_ > 0 && !last_frame_armors_.empty()) {
                    // 查找上一帧相同ID的装甲板
                    auto last_armor_it = last_frame_armors_.find(tracked_armor.id);
                    if (last_armor_it != last_frame_armors_.end()) {
                        const auto& last_armor = last_armor_it->second;
                        
                        // 获取上一帧和当前帧的yaw 需要转换吗，是否可以直接获取
                        double yaw_las = atan2(last_armor.position_odom[1], last_armor.position_odom[0]);
                        double yaw_now = atan2(tracked_armor.position_odom[1], tracked_armor.position_odom[0]);
                        
                        // 获取yaw边界和中线（如果历史记录存在）
                        if (!outpost.yaw_increase_history.empty() && !outpost.yaw_decrease_history.empty()) {
                            double yaw_l = outpost.yaw_increase_history.front().second;
                            double yaw_r = outpost.yaw_decrease_history.front().second;
                            double yaw_middle = angle_middle(yaw_l, yaw_r);
                            
                            // 获取旋转速度
                            double omega = outpost.common_yaw_spd.get();
                            
                            // 过中检测
                            // 装甲板前后yaw角跨越yaw中线
                            if ((omega > 0.1 && angle_between(yaw_l, yaw_middle, yaw_las) && angle_between(yaw_middle, yaw_r, yaw_now)) || 
                                (omega < -0.1 && angle_between(yaw_middle, yaw_r, yaw_las) && angle_between(yaw_l, yaw_middle, yaw_now))) {

                                if (armor.phase_in_outpost_ >= 0) {
                                    // 更新LS拟合器
                                    outpost.T_solver.update(timestamp);
                                    outpost.T_solver.solve();
                                    // 更新中间距离和pitch滤波器
                                    Eigen::Vector3d pass_middle_pyd = xyz2pyd(tracked_armor.position_odom);
                                    outpost.common_middle_dis.update(pass_middle_pyd[2]);
                                    outpost.common_middle_pitch.update(pass_middle_pyd[0]);
                                    // RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Pass Middle Timestamp: %lf", timestamp);
                                    // RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "yaw_now %.3lf, yaw_las %.3lf", yaw_now, yaw_las);
                                } else {
                                    // RCLCPP_WARN(rclcpp::get_logger("outpostaim"), "Matched but no id ???");
                                }
                            }
                        }
                    }
                }

                // 更新应该用这一帧还是上一帧的信息？？

                // 更新 yaw 速度滤波 
                outpost.common_yaw_spd.update(armor.getYawSpd());

                armor.update(pc_result, timestamp);
                armor.status_ = Alive;
                armor.alive_ts_ = timestamp;
                armor.matched_ = true;
                
                // 更新2D信息
                armor.dis_2d_ = sqrt(pow(tracked_armor.bbox.x + tracked_armor.bbox.width/2 - params.collimation.x, 2) +
                                     pow(tracked_armor.bbox.y + tracked_armor.bbox.height/2 - params.collimation.y, 2));
                armor.area_2d_ = tracked_armor.bbox.area();
                
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Matched existing armor with phase %d", phase);

                // 匹配成功：相位不变，并更新“最新活跃装甲板”基准
                if (armor.phase_in_outpost_ >= 0) {
                    phase_initialized_ = true;
                    last_active_tracker_id_ = armor.tracker_id;
                    last_active_phase_ = armor.phase_in_outpost_;
                    last_active_yaw_ = armor.getPositionPyd()[1];
                    last_active_ts_ = timestamp;
                }
                break;
            }
        }
        
        if (!matched_existing) {
            // ========= 快速恢复：trackid变更但物理装甲板未变（即将消失被当成新目标） =========
            // 通过yaw连续性把“新trackid”快速绑回原相位，避免相位漂移
            // 注意：本帧开始时 status_ 已被置为 Absent，不能用 status_ / last_status_ 来筛选“可重关联候选”
            // 这里用 alive_ts_ 的新鲜度作为候选条件（允许短暂丢失几帧后拉回）
            // 固定阈值（弧度）：用于快速把“新trackid”拉回原相位
            constexpr double kReassocYawThresh = 0.02;
            const double kReassocMaxAge = std::min(0.30, params.reset_time);
            int best_reassoc_idx = -1;
            double best_reassoc_dyaw = 1e9;
            for (int i = 0; i < (int)outpost.armors.size(); ++i) {
                const auto& a = outpost.armors[i];
                if (a.phase_in_outpost_ < 0) continue;
                if (a.alive_ts_ < 0) continue;
                const double age = timestamp - a.alive_ts_;
                if (age > kReassocMaxAge) continue;

                const double a_yaw = a.getPositionPyd()[1];
                const double dyaw = fabs(get_disAngle(pyd_pos[1], a_yaw));
                if (dyaw < best_reassoc_dyaw) {
                    best_reassoc_dyaw = dyaw;
                    best_reassoc_idx = i;
                }
            }
            if (best_reassoc_idx == -1) {
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"),
                            "Best reassoc idx: -1 (no candidates), armors=%zu, max_age=%.3f", outpost.armors.size(), kReassocMaxAge);
            } else {
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Best reassoc idx: %d, dyaw: %.3f", best_reassoc_idx, best_reassoc_dyaw);
            }
            if (best_reassoc_idx != -1 && best_reassoc_dyaw < kReassocYawThresh) {
                auto& a = outpost.armors[best_reassoc_idx];
                const int phase = a.phase_in_outpost_;

                // 绑定新的trackid并更新该装甲板
                a.tracker_id = tracked_armor.id;
                a.update(pc_result, timestamp);
                a.status_ = Alive;
                a.alive_ts_ = timestamp;
                a.matched_ = true;
                a.dis_2d_ = sqrt(pow(tracked_armor.bbox.x + tracked_armor.bbox.width / 2 - params.collimation.x, 2) +
                                 pow(tracked_armor.bbox.y + tracked_armor.bbox.height / 2 - params.collimation.y, 2));
                a.area_2d_ = tracked_armor.bbox.area();

                phase_initialized_ = true;
                last_active_tracker_id_ = a.tracker_id;
                last_active_phase_ = phase;
                last_active_yaw_ = a.getPositionPyd()[1];
                last_active_ts_ = timestamp;

                RCLCPP_WARN(rclcpp::get_logger("outpostaim"),
                            "Re-associated armor by yaw continuity: new_tracker_id=%d -> phase=%d (|dyaw|=%.3f < thr=%.3f)",
                            tracked_armor.id, phase, best_reassoc_dyaw, kReassocYawThresh);
                continue;
            }

            // 创建新的OutpostArmor
            OutpostArmor new_armor;
            
            // 首先设置相位（tracker ID对3取余）  不再使用
            // new_armor.phase_in_outpost_ = tracked_armor.id % 3;

            // 设置tracker ID用于后续帧间匹配
            new_armor.tracker_id = tracked_armor.id;

            // 设置装甲板ID（颜色和类型）  严查 这个id是？？
            // 如果无法从 detection 得到 ArmorId，使用默认构造
            new_armor.armor_id_ = ArmorId();
            new_armor.score = tracked_armor.score;
            
            // ========== 关键修改：基于“最新活跃装甲板”分配相位 ==========
            int assigned_phase = -1;

            // 计算新装甲板yaw（yaw向左为增大，atan2符合该定义）
            const Eigen::Vector3d new_pyd = xyz2pyd(tracked_armor.position_odom);

            // 只有前哨站自瞄启动后看到的第一块装甲板才走“第一块=0”的初始化逻辑
            if (!phase_initialized_) {
                assigned_phase = 0;
                phase_initialized_ = true;
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Phase init: first armor -> phase 0 (tracker_id=%d)", tracked_armor.id);
            } else {
                // 以“最后最新活跃装甲板”为基准：优先使用上次记录；若无则退化到历史里alive_ts_最新且有相位的装甲板
                int ref_phase = last_active_phase_;
                double ref_yaw = last_active_yaw_;

                if (ref_phase < 0) {
                    double best_ts = -1.0;
                    for (const auto& a : outpost.armors) {
                        if (a.phase_in_outpost_ < 0) continue;
                        if (a.alive_ts_ > best_ts) {
                            best_ts = a.alive_ts_;
                            ref_phase = a.phase_in_outpost_;
                            ref_yaw = a.getPositionPyd()[1];
                        }
                    }
                }

                if (ref_phase >= 0) {
                    const Eigen::Vector3d ref_pyd(0.0, ref_yaw, 0.0);

                    // 新装甲板在左：phase = (ref - 1) % 3；在右：phase = (ref + 1) % 3
                    // 注意：check_left(A,B) 为真表示 A 在 B 左侧（yaw更大一侧，考虑wrap）
                    if (check_left(new_pyd, ref_pyd)) {
                        assigned_phase = (ref_phase + 2) % 3;
                    } else {
                        assigned_phase = (ref_phase + 1) % 3;
                    }

                    RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "New armor (tracker_id=%d) yaw=%.3f ref_phase=%d ref_yaw=%.3f -> assigned_phase=%d",
                                tracked_armor.id, new_pyd[1], ref_phase, ref_yaw, assigned_phase);
                } else {
                    // 兜底：没有任何有效基准（理论上只会在刚启动/被重置时发生）
                    assigned_phase = 0;
                    RCLCPP_WARN(rclcpp::get_logger("outpostaim"), "No valid reference phase; fallback assigned_phase=0 (tracker_id=%d)", tracked_armor.id);
                }
            }

            new_armor.phase_in_outpost_ = assigned_phase;
            // ========== 关键修改结束 ==========

        ArmorPoseResult pc_result;
        pc_result.pose.x = tracked_armor.position_odom[0];
        pc_result.pose.y = tracked_armor.position_odom[1];
        pc_result.pose.z = tracked_armor.position_odom[2];
        // 计算位置 yaw 和 pitch
        Eigen::Vector3d pyd_pos = xyz2pyd(tracked_armor.position_odom);
        pc_result.pose.pitch = pyd_pos[0];
        
        // 计算法向量
        pc_result.normal_vec = tracked_armor.orientation_odom * Eigen::Vector3d::UnitZ();

        // 恢复旧版逻辑：Yaw 采用装甲板法向量的朝向，与旧代码 rm_base/pose_calculator.cpp 保持一致
        Eigen::Vector3d pyd_normal = xyz2pyd(pc_result.normal_vec);
        pc_result.pose.yaw = pyd_normal[1];

        // 其他成员保持默认
        pc_result.reproject_error = 0.0;
            // 初始化滤波器
            new_armor.init(pc_result, timestamp);
            new_armor.status_ = Alive;
            new_armor.alive_ts_ = timestamp;
            
            // 设置2D信息
            new_armor.dis_2d_ = sqrt(pow(tracked_armor.bbox.x + tracked_armor.bbox.width/2 - params.collimation.x, 2) +
                                     pow(tracked_armor.bbox.y + tracked_armor.bbox.height/2 - params.collimation.y, 2));
            new_armor.area_2d_ = tracked_armor.bbox.area();
            
            new_armors.push_back(new_armor);
            RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Created new armor with phase %d", new_armor.phase_in_outpost_);

            // 新建成功也更新“最新活跃装甲板”基准
            if (new_armor.phase_in_outpost_ >= 0) {
                last_active_tracker_id_ = new_armor.tracker_id;
                last_active_phase_ = new_armor.phase_in_outpost_;
                last_active_yaw_ = new_pyd[1];
                last_active_ts_ = timestamp;
            }
        }
    }
    
    // 更新上一帧信息
    last_frame_armors_ = std::move(current_frame_armors);
    last_frame_timestamp_ = timestamp;

    // 添加新的装甲板到前哨站
    for (auto& armor : new_armors) {
        if (!outpost_exist) {
            outpost = Outpost(armor.armor_id_, false, false, 3);
            outpost_exist = true;
        }

        // 保持“相位0/1/2对应固定装甲板”的映射：同相位只保留一份（新数据覆盖旧数据）
        bool replaced = false;
        if (armor.phase_in_outpost_ >= 0) {
            for (auto& existing : outpost.armors) {
                if (existing.phase_in_outpost_ == armor.phase_in_outpost_) {
                    existing = armor;
                    replaced = true;
                    break;
                }
            }
        }
        if (!replaced) {
            outpost.armors.push_back(armor);
        }

        // 更新瞄准点z值滤波器
        outpost.aiming_z_filter.reset();
        outpost.aiming_z_filter.update(armor.getPositionXyz()[2]);

    }
    
    // 处理过时的装甲板（超过reset_time未更新）
    if (outpost_exist) {
        for (auto it = outpost.armors.begin(); it != outpost.armors.end();) {
            if (it->alive_ts_ + params.reset_time < timestamp) {// ???alivets是-1.0？？？
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Armor with phase %d timed out (last alive: %.2f, current: %.2f, reset_time: %.2f)", it->phase_in_outpost_, it->alive_ts_, timestamp, params.reset_time);
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Removing outdated armor (phase %d)", it->phase_in_outpost_);
                it = outpost.armors.erase(it);
            } else {
                ++it;
            }
        }

        if (outpost.armors.empty()) {
            RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Outpost reset due to no active armors");
            outpost = Outpost();
            phase_initialized_ = false;
            last_active_tracker_id_ = -1;
            last_active_phase_ = -1;
            last_active_yaw_ = 0.0;
            last_active_ts_ = -1.0;
        }
    }
}

//-----------------------------------------UpdateArmor---------------------------------

//-----------------------------------------UpdateOutpost--------------------------------

void OutpostManager::updateOutpost(){
    outpost.status = Status::Absent;
    static std::vector<int> alive_indexs(outpost.armor_cnt);
    alive_indexs.clear();

     // 收集活跃装甲板
    for (int i = 0; i < (int)outpost.armors.size(); ++i) {
        if (outpost.armors[i].status_ == Status::Alive) {
            outpost.status = Alive;
            outpost.alive_ts = outpost.armors[i].alive_ts_;
            alive_indexs.push_back(i);
        }
    }
    
    if (outpost.status == Status::Absent) {
        return;
    }
    
    // 更新最小2D距离
    outpost.min_dis_2d = INFINITY;
    for (auto& armor : outpost.armors) {
        outpost.min_dis_2d = std::min(outpost.min_dis_2d, armor.dis_2d_);
    }
    
    // 如果有活跃装甲板，选择面积最大的进行更新
    if (!alive_indexs.empty()) {
        int big_idx = 0;
        if (alive_indexs.size() > 1) {
            big_idx = (outpost.armors[alive_indexs[0]].area_2d_ > outpost.armors[alive_indexs[1]].area_2d_) ? 0 : 1;
        }
        
        OutpostArmor& armor = outpost.armors[alive_indexs[big_idx]];
        
        // 更新历史记录
        while (!outpost.yaw_increase_history.empty() && 
               outpost.yaw_increase_history.back().second >= armor.getYaw()) {
            outpost.yaw_increase_history.pop_back();
        }
        outpost.yaw_increase_history.push_back(std::make_pair(armor.alive_ts_, armor.getYaw()));
        
        while (!outpost.yaw_decrease_history.empty() && 
               outpost.yaw_decrease_history.back().second <= armor.getYaw()) {
            outpost.yaw_decrease_history.pop_back();
        }
        outpost.yaw_decrease_history.push_back(std::make_pair(armor.alive_ts_, armor.getYaw()));

        // 更新CKF（如果有有效的相位）   暂时不过滤
        if (armor.phase_in_outpost_ >= 0) {
            double angle_dis = M_PI * 2 / outpost.armor_cnt;
            OutpostCkf::Observe now_observe;
            now_observe.x = armor.getPositionXyz()[0];
            now_observe.y = armor.getPositionXyz()[1];
            now_observe.z = armor.getPositionXyz()[2];

            now_observe.yaw = armor.pc_result_.pose.yaw - armor.phase_in_outpost_ * angle_dis;

            // 更新高度映射，检查是否已收集到三块装甲板的高度数据
            outpost.update_armor_height(armor.phase_in_outpost_, now_observe.z);
            int collected_count = 0;
            for (int i = 0; i < 3; ++i) {
                if (fabs(outpost.armor_height_filter[i].get() - 0.0) > 0.01) {
                    collected_count++;
                }
            }
            if (collected_count >= 3 && !outpost.height_mapping_initialized_) {
                outpost.height_mapping_initialized_ = true;
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Height mapping initialized with 3 armor heights");
            }
            // 第一次初始化 待修改
            if (!outpost.outpost_kf_init) {
                if(armor.phase_in_outpost_ != 0) return; // 还没找到问题，暂时限制一下
                outpost.last_yaw = now_observe.yaw;
                outpost.yaw_round = 0;
                outpost.op_ckf.reset(now_observe, armor.phase_in_outpost_, outpost.armor_cnt, 
                                     outpost.alive_ts, now_observe.z);
                outpost.outpost_kf_init = true;
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Outpost CKF initialized with armor phase=%d", armor.phase_in_outpost_);
            }
            // 处理过0
            if (now_observe.yaw - outpost.last_yaw < -M_PI * 1.5) {
                outpost.yaw_round++;
            } else if (now_observe.yaw - outpost.last_yaw > M_PI * 1.5) {
                outpost.yaw_round--;
            }
            outpost.last_yaw = now_observe.yaw;
            now_observe.yaw = now_observe.yaw + outpost.yaw_round * 2 * M_PI;

            outpost.update(now_observe, outpost.alive_ts, armor.phase_in_outpost_);
            RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "CKF updated with phase %d", armor.phase_in_outpost_);
            

            // 更新高度滤波器
            outpost.const_z_filter.update(now_observe.z);
            outpost.op_ckf.const_z_ = outpost.const_z_filter.get();

            // 计算方向差异
            Eigen::Vector3d center_xyz = outpost.op_ckf.get_center();
            Eigen::Vector3d center_pyd = xyz2pyd(center_xyz);
            outpost.ori_diff = Eigen::Vector2d(center_pyd[0] - robot.imu.pitch, 
                                               center_pyd[1] - robot.imu.yaw).norm();
        }
    }
    
    // 可视化：边界、预测装甲点、中心点与高度映射状态（使用已注册的 projector ）
    if (params.enable_imshow && !params.debug && projector && !result_img_.empty()) {
        // 需要足够的历史数据
        if (!outpost.yaw_increase_history.empty() && !outpost.yaw_decrease_history.empty()) {
            double yaw_min = outpost.yaw_increase_history.front().second;
            double yaw_max = outpost.yaw_decrease_history.front().second;

            // 左右边界 有点歪
            cv::Point2d p1 = projector(pyd2xyz(Eigen::Vector3d{0.7, yaw_min, 10.}));
            cv::Point2d p2 = projector(pyd2xyz(Eigen::Vector3d{-0.7, yaw_min, 10.}));
            if (p1.x >= 0 && p2.x >= 0) cv::line(result_img_, p1, p2, cv::Scalar(0,0,255), 2);
            cv::Point2d p3 = projector(pyd2xyz(Eigen::Vector3d{0.7, yaw_max, 10.}));
            cv::Point2d p4 = projector(pyd2xyz(Eigen::Vector3d{-0.7, yaw_max, 10.}));
            if (p3.x >= 0 && p4.x >= 0) cv::line(result_img_, p3, p4, cv::Scalar(0,0,255), 2);

            // 若为英雄，绘制中线并显示边界文本
            if (self_id.robot_id == RobotId::ROBOT_HERO) {
                double yaw_middle = angle_middle(yaw_min, yaw_max);
                cv::Point2d pm1 = projector(pyd2xyz(Eigen::Vector3d{0.7, yaw_middle, 10.}));
                cv::Point2d pm2 = projector(pyd2xyz(Eigen::Vector3d{-0.7, yaw_middle, 10.}));
                if (pm1.x >= 0 && pm2.x >= 0) cv::line(result_img_, pm1, pm2, cv::Scalar(0,255,0), 1);

                std::string boundary_text = "Yaw: min=" + std::to_string(rad2deg(yaw_min)).substr(0,4) +
                                            "°, max=" + std::to_string(rad2deg(yaw_max)).substr(0,4) + "°";
                cv::putText(result_img_, boundary_text, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
            }

            // 显示高度映射状态
            std::string height_status = outpost.height_mapping_initialized_ ? "Height Mapping: INITIALIZED" : "Height Mapping: COLLECTING";
            cv::putText(result_img_, height_status, cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255), 2);

            // 当前装甲板ID（选择第一个Alive的装甲板）
            int current_id = -1;
            for (const auto &a : outpost.armors) {
                if (a.status_ == Status::Alive) { current_id = a.phase_in_outpost_; break; }
            }
            std::string armor_id_text = "Current Armor ID: " + std::to_string(current_id);
            cv::putText(result_img_, armor_id_text, cv::Point(10,90), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);

            int left_text_y = 120;
            // 显示各装甲板高度
            for (int i = 0; i < 3; ++i) {
                // 获取装甲板高度
                double armor_height = outpost.get_armor_height_by_id(i);
                // 构建显示文本
                char height_text[100];
                snprintf(height_text, sizeof(height_text), "ID %d: %.3f m)", i, armor_height);
                cv::putText(result_img_, height_text, 
                        cv::Point(10, left_text_y), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
                left_text_y += 25;
                }
            // 预测装甲点与中心点
            Outpost::OutpostPosition predicted_positions = outpost.predict_positions(recv_detection.time_stamp);
            for (int i = 0; i < outpost.armor_cnt; ++i) {
                if (i < (int)predicted_positions.armors_xyz_.size()) {
                    Eigen::Vector3d armor_pos = predicted_positions.armors_xyz_[i];
                    cv::Point2d ip = projector(armor_pos);
                    if (ip.x >= 0) {
                        cv::circle(result_img_, ip, 5, cv::Scalar(0,255,255), -1);
                        std::string id_text = std::to_string(i);
                        cv::putText(result_img_, id_text, cv::Point(ip.x - 5, ip.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,255,255), 1);
                    }
                }
            }
            cv::Point2d center_p = projector(predicted_positions.center_);
            if (center_p.x >= 0) cv::circle(result_img_, center_p, 5, cv::Scalar(0,255,255), -1);

            std::string fitter_text = "omega " + std::to_string(outpost.get_rotate_spd()).substr(0, 5) +
                                      " rad/s, yaw_spd " + std::to_string(outpost.common_yaw_spd.get()).substr(0, 5);
            cv::putText(result_img_, fitter_text, cv::Point(10, 200), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(255, 255, 255), 1);
        }
    }
    // rviz可视化

    // 清理过时的历史记录
    double current_time = recv_detection.time_stamp;
    while (!outpost.yaw_increase_history.empty() && 
           current_time - outpost.yaw_increase_history.front().first > params.census_period_max) {
        outpost.yaw_increase_history.pop_front();
    }
    while (!outpost.yaw_decrease_history.empty() && 
           current_time - outpost.yaw_decrease_history.front().first > params.census_period_max) {
        outpost.yaw_decrease_history.pop_front();
    }

    // clear TSP, a balance between outliers & good_fitting_samples
    // for (; !outpost.yaw_increase_history.empty() && recv_detection.time_stamp - outpost.yaw_increase_history.front().first > params.census_period_max;
    //         outpost.yaw_increase_history.pop_front());
    // for (; !outpost.yaw_decrease_history.empty() && recv_detection.time_stamp - outpost.yaw_decrease_history.front().first > params.census_period_max;
    //         outpost.yaw_decrease_history.pop_front());
    // // 清理过时的过中数据
    // for (; outpost.T_solver.N && recv_detection.time_stamp - outpost.T_solver.datas.front() > params.anti_outpost_census_period;
    //     outpost.T_solver.pop_front());
}

//----------------------------------UpdateOutpost-------------------------------

//----------------------------------SelectTarget--------------------------------

/**
 * @brief 英雄动态选择装甲板击打
 * @param selected_time_diff 选中的装甲板时间差（输出）
 * @param selected_total_delay 选中的装甲板总延迟（输出）
 * @param selected_system_delay 选中的装甲板系统延迟（输出）
 * @param time_diffs 三块装甲板的时间差（输出）
 * @param total_delays 三块装甲板的总延迟（输出）
 * @param adjust_flags 三块装甲板的调整云台标志（输出）
 * @return 选中的装甲板ID，-1表示没有合适的装甲板
 */
int OutpostNode::select_armor_dynamically(double &selected_time_diff, double &selected_total_delay, 
                                                   double &selected_system_delay, std::vector<double> &time_diffs,
                                                   std::vector<double> &total_delays, std::vector<bool> &adjust_flags) {
    auto& outpost = manager_.outpost;
    double now = manager_.recv_detection.time_stamp;

    // 清空输出向量
    time_diffs.clear();
    total_delays.clear();
    adjust_flags.clear();

    // 检查是否有足够的过中数据（至少需要3次过中）
    if (outpost.T_solver.N < 3) {
        RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Not enough middle events data: N=%zu", outpost.T_solver.N);
        return -1;
    }
    // 获取当前正在跟踪的装甲板（如果有的话） 如果同时有两块，可以根据旋转方向判断取左还是右。如果没有呢？这里待修改
    int current_armor_id = -1;
    double current_armor_yaw = 0;
    bool found_current = false;
    for (const auto& armor : outpost.armors) {
        if (armor.in_follow && armor.phase_in_outpost_ >= 0) {
            current_armor_id = armor.phase_in_outpost_;
            current_armor_yaw = armor.getYaw();
            found_current = true;
            break;
        }
    }
    if (current_armor_id < 0 || current_armor_id >= 3) {
        RCLCPP_WARN(rclcpp::get_logger("outpostaim"), "Invalid current armor id: %d", current_armor_id);
        return -1;
    }
    // 预测后两次过中时间
    outpost.T_solver.solve();
    int current_n = outpost.T_solver.N; // 当前已过中次数
    // 预测第N+1次和第N+2次过中时间
    double middle_time_1 = outpost.T_solver.predict_nth_middle_time(current_n + 1);
    double middle_time_2 = outpost.T_solver.predict_nth_middle_time(current_n + 2);
    if (middle_time_1 < 0 || middle_time_2 < 0) {
        RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Failed to predict next middle times");
        return -1;
    }
    // 预测后两个过中的装甲板ID
    // 目标右移，yaw减小，yaw_spd < 0
    // 获取中线yaw（用于判断是否过中）
    double yaw_middle = 0;
    if (!outpost.yaw_increase_history.empty() && !outpost.yaw_decrease_history.empty()) {
        double yaw_min = outpost.yaw_increase_history.front().second;
        double yaw_max = outpost.yaw_decrease_history.front().second;
        yaw_middle = angle_middle(yaw_min, yaw_max);
    } else {
        // 如果没有历史数据，使用云台yaw作为中线
        yaw_middle = imu.yaw;
        RCLCPP_ERROR(rclcpp::get_logger("outpostaim"), "No yaw history for hero select armor");
    }
    // 判断当前装甲板是否已过中
    bool has_passed_middle = false;
    if (found_current) {
        if (outpost.common_yaw_spd.get() > 0.1) {  // 目标左移
            // yaw从min增加到max，过中线后 middle < yaw < max
            if (angle_between(yaw_middle, yaw_max, current_armor_yaw)) {
                has_passed_middle = true;
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Current armor %d has passed middle in CCW rotation, yaw_middle %.3lf, yaw_max %.3lf, yaw_armor %.3lf", current_armor_id, yaw_middle, yaw_max, current_armor_yaw);
            } else {
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Current armor %d has NOT passed middle in CCW rotation, yaw_middle %.3lf, yaw_max %.3lf, yaw_armor %.3lf", current_armor_id, yaw_middle, yaw_max, current_armor_yaw);
            }
        } else if (outpost.common_yaw_spd.get() < -0.1){  // 目标右移
            // yaw从max减少到min，过中线后 min < yaw < middle
            if (angle_between(yaw_min, yaw_middle, current_armor_yaw)) {
                has_passed_middle = true;
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Current armor has passed middle in CW rotation %d", current_armor_id);
            } else {
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Current armor has NOT passed middle in CW rotation %d", current_armor_id);
            }
        } else {
            RCLCPP_WARN(rclcpp::get_logger("outpostaim"), "invaild yaw speed");
            return -1;
        }
    }
    // 根据是否已过中，确定下一次过中的装甲板ID
    int next_ids[2];
    if (!has_passed_middle) {
        // 当前装甲板还未过中，下一次过中的是当前装甲板
        next_ids[0] = current_armor_id;
        next_ids[1] = (current_armor_id + 1) % 3;  // 下一块装甲板
    } else {
        // 当前装甲板已过中，下一次过中的是下一块装甲板 -0.1<spd<0.1的情况在上面已处理
        next_ids[0] = (current_armor_id + 1) % 3;
        next_ids[1] = (current_armor_id + 2) % 3;
    }
    
    // 计算时间差
    double time_diff_1 = middle_time_1 - now;
    double time_diff_2 = middle_time_2 - now;
    // 存储两个装甲板的信息
    struct ArmorInfo {
        int id;
        double middle_time;
        double time_diff;
        double system_delay;
        double total_delay;
        bool is_selectable;
    };
    std::vector<ArmorInfo> armor_infos;
    armor_infos.resize(2);
    // 计算两个装甲板的延迟
    for (int i = 0; i < 2; ++i) {
        int armor_id = (i == 0) ? next_ids[0] : next_ids[1];
        double middle_time = (i == 0) ? middle_time_1 : middle_time_2;
        double time_diff = (i == 0) ? time_diff_1 : time_diff_2;
        // 获取装甲板高度
        double armor_height = outpost.get_armor_height_by_id(armor_id);
        // 计算飞行时间（用于系统延迟）
        Eigen::Vector3d dummy_center;
        Ballistic::BallisticResult ball_res = center_ballistic(dummy_center, armor_height);
        if (ball_res.fail) {
            RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Ballistic calculation for armor %d failed", armor_id);
            armor_infos[i] = {armor_id, middle_time, time_diff, INFINITY, INFINITY, false};
            continue;
        }
        // 计算系统延迟（响应+飞行+发弹）
        double system_delay = params_.response_delay + ball_res.t + params_.shoot_delay;
        // 计算总延迟（云台调整+系统延迟）
        double total_delay = params_.gimbal_adjust_delay + system_delay;
        // 判断是否可选择：总延迟 < 时间差 且 时间差 > 0
        bool is_selectable = (total_delay < time_diff) && (time_diff > 0);
        armor_infos[i] = {armor_id, middle_time, time_diff, system_delay, total_delay, is_selectable};
        RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Armor %d: middle_time=%.3f, time_diff=%.3f, system_delay=%.3f, total_delay=%.3f, selectable=%d", 
                armor_id, middle_time, time_diff, system_delay, total_delay, is_selectable);
    }
    // 选择符合条件的装甲板（优先选择总延迟最小的）
    int selected_id = -1;
    double min_total_delay = INFINITY;
    for (const auto& info : armor_infos) {
        if (info.is_selectable && info.total_delay < min_total_delay) {
            min_total_delay = info.total_delay;
            selected_id = info.id;
        }
    }
    // 如果没有符合条件的装甲板，返回-1
    if (selected_id == -1) {
        RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "No selectable armor found");
        return -1;
    }
    // 找到选中的装甲板信息
    ArmorInfo selected_info;
    for (const auto& info : armor_infos) {
        if (info.id == selected_id) {
            selected_info = info;
            break;
        }
    }
    // 填充输出向量（大小为3，后两个装甲板的信息放在索引0和1）
    time_diffs.resize(3, INFINITY);
    total_delays.resize(3, INFINITY);
    adjust_flags.resize(3, false);
    for (int i = 0; i < 2; ++i) {
        time_diffs[i] = armor_infos[i].time_diff;
        total_delays[i] = armor_infos[i].total_delay;
        // 调整标志：总延迟 < 时间差 且 选中该装甲板
        adjust_flags[i] = armor_infos[i].is_selectable && (armor_infos[i].id == selected_id);
    }
    // 设置选中的装甲板信息
    selected_time_diff = selected_info.time_diff;
    selected_total_delay = selected_info.total_delay;
    selected_system_delay = selected_info.system_delay;
    RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Selected armor %d: time_diff=%.3f, total_delay=%.3f, system_delay=%.3f", 
                 selected_id, selected_time_diff, selected_total_delay, selected_system_delay);

    return selected_id;
}

OpNewSelectArmor OutpostNode::select_armor_directly(){
    static int last_selected_id = -1;
    static double change_target_armor_ts = 0;
    
    auto& outpost = manager_.outpost;
    // if (outpost.armors.empty()) {
    //     OpNewSelectArmor res;
    //     res.armors_index_in_outpost = -1;
    //     return res;
    // }
    // 直接选择最优的
    // 计算大致飞行时间
    Outpost::OutpostPosition pos_now = outpost.predict_positions(manager_.recv_detection.time_stamp);
    Ballistic::BallisticResult ball_estimate 
                    = bac->final_ballistic(getTrans("odom", "gimbal"), pos_now.center_);
    // 预测
    Outpost::OutpostPosition pos_predict = outpost.predict_positions(
        params_.response_delay + ball_estimate.t + manager_.recv_detection.time_stamp);
    // 这里的yaw是id0装甲板的yaw 是否有问题？
    double center_yaw = atan2(pos_now.center_[1], pos_now.center_[0]);
    
    // 选取最正对的装甲板
    double min_dis_yaw = INFINITY;
    int selected_id = -1;

    for (int i = 0; i < outpost.armor_cnt; ++i) {
        if (static_cast<size_t>(i) >= pos_predict.armor_yaws_.size()) continue;
        
        double pre_dis = get_disAngle(pos_predict.armor_yaws_[i], center_yaw + M_PI);  // 加PI，换方向
        
        if (abs(pre_dis) < abs(min_dis_yaw)) {
            min_dis_yaw = pre_dis;
            selected_id = i;
        }
    }
    int anti_clock_wise = outpost.op_ckf.state_.omega > 0 ? 1 : -1; //?并未用上
    
    if (params_.rmcv_id.robot_id != RobotId::ROBOT_HERO) {
        if (manager_.recv_detection.time_stamp - change_target_armor_ts < params_.change_armor_time_thresh) {
            selected_id = last_selected_id;
            min_dis_yaw = get_disAngle(pos_predict.armor_yaws_[selected_id], center_yaw + M_PI);
        }
        if (selected_id != last_selected_id) {
            change_target_armor_ts = manager_.recv_detection.time_stamp;
        }
        last_selected_id = selected_id;
    }
    OpNewSelectArmor res;
    res.armors_index_in_outpost = selected_id;
    res.yaw_distance_predict = min_dis_yaw;
    res.xyz = pos_now.armors_xyz_[selected_id];

    if (selected_id >= 0) {
        // 遍历armors寻找当前选中的装甲板
        for (const auto& armor : outpost.armors) {
            if (armor.phase_in_outpost_ == selected_id && armor.status_ == Status::Alive) {
                // 如果有则使用当前观测到的位置
                res.xyz[2] = armor.getPositionXyz()[2];
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Selected armor %d observed z: %.3f m", selected_id, res.xyz[2]);
                break;
            }
        }
    }
    return res;
}

//-----------------------------------SelectTarget--------------------------

//-----------------------------------GenerateCommand-----------------------

/**
 * @brief 英雄专用弹道计算函数 针对前哨站特殊优化
 * @param predict_center 预测中心点（输出）
 * @param armor_z 目标装甲板的实际z坐标（高度）
 * @return 弹道计算结果
 * 
 * 功能说明：
 * 1. xy坐标使用前哨站中心（通过滤波器平滑）
 * 2. z坐标使用传入的装甲板实际高度
 * 3. 英雄不需要跟随装甲板yaw，只需瞄准中心，根据装甲板高度调整pitch
 */
Ballistic::BallisticResult OutpostNode::center_ballistic(Eigen::Vector3d &predict_center, double armor_z){
    Ballistic::BallisticResult ball_res;
    double t_fly = 0;  // 飞行时间（迭代求解）
    Outpost& outpost = manager_.outpost;
    // 获取前哨站中心的球坐标（pitch, yaw, distance）
    for (int i = 0; i < 3; ++i) {
        predict_center[0] = outpost.common_middle_pitch.get();
        predict_center[1] = angle_middle(yaw_min, yaw_max);
        predict_center[2] = outpost.common_middle_dis.get();
        predict_center = pyd2xyz(predict_center);
        // 对xy坐标进行滤波（保持平滑）
        for (int j = 0; j < 2; ++j) {  // 只滤波x,y
            outpost.center_pos_filter[j].update(predict_center[j]);
            predict_center[j] = outpost.center_pos_filter[j].get();
        }
        // 使用传入的装甲板实际高度，而不是前哨站中心高度
        // 这样云台yaw指向中心，pitch根据装甲板高度调整
        predict_center[2] = armor_z;
        // 不考虑z_velocity的误差
        ball_res = bac->final_ballistic(getTrans("odom", "gimbal"), predict_center);
        if (ball_res.fail) {
            RCLCPP_WARN(rclcpp::get_logger("outpostaim"), "too far to hit it");
            return ball_res;
        }
        t_fly = ball_res.t;
    }
    return ball_res;
}
Ballistic::BallisticResult OutpostNode::calc_ballistic(int armor_phase, double delay, Eigen::Vector3d &predict_pos, double armor_height){
    Ballistic::BallisticResult ball_res;
    double t_fly = 0;  // 飞行时间（迭代求解）
    for (int i = 0; i < 3; ++i) {
        tock = CLK.now();
        double latency = delay + (tock - tick).seconds();
        predict_pos = manager_.outpost.predict_positions(manager_.recv_detection.time_stamp + t_fly + latency).armors_xyz_[armor_phase];
        // 使用观测的z坐标
        predict_pos[2] = armor_height;
        ball_res = bac->final_ballistic(getTrans("odom", "gimbal"), predict_pos);
        if (ball_res.fail) {
            RCLCPP_WARN(rclcpp::get_logger("outpostaim"), "too far to hit it");
            return ball_res;
        }
        t_fly = ball_res.t;
    }
    return ball_res;
}

ControlMsg OutpostNode::get_command(){
    
    auto& outpost = manager_.outpost;
    bool outpost_exist = outpost.alive_ts > 0;

    // 安全检查：确保outpost存在且有效
    if (!outpost_exist) {
        RCLCPP_WARN(rclcpp::get_logger("outpostaim"), "NO OUTPOST!!!");
        // pub_float_data(target_dis_pub, std::nan("1"));    // 无目标时认为目标距离无限远
        return off_cmd;
    }
    if(!outpost.outpost_kf_init){
        RCLCPP_WARN(rclcpp::get_logger("outpostaim"), "aim wating for outpost init");
        return off_cmd;
    }

    // 获取前哨站中心距离（用于目标距离发布） 新识别貌似不需要接受距离了？
    Eigen::Vector3d center_pos = outpost.predict_positions(manager_.recv_detection.time_stamp).center_;
    double outpost_target_dis = center_pos.norm();
    // pub_float_data(target_dis_pub, outpost_target_dis);
    RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "outpost_dis: %lf", outpost_target_dis);

    // if (outpost.armors.empty()) {
    //     // NGXY_WARN("Outpost has no armors!");
    //     return off_cmd;
    // }
    outpost.in_follow = true;
    
    if ((params_.rmcv_id.robot_id == RobotId::ROBOT_HERO) &&
         !outpost.height_mapping_initialized_ && !params_.debug) {
        RCLCPP_WARN(rclcpp::get_logger("outpostaim"), "Height mapping not initialized yet for hero");
        return off_cmd;
    }

    //  英雄动态选择装甲板
    int follow_armor_id = -1;
    int dynamic_selected_id = -1;
    double selected_time_diff = INFINITY;
    double selected_total_delay = INFINITY;
    double selected_system_delay = INFINITY;
    std::vector<double> time_diffs;
    std::vector<double> total_delays;
    std::vector<bool> adjust_flags;
    std::vector<bool> fire_flags(3, false);  // 是否开火的判断

    auto target = select_armor_directly();  // 整车建模策略下选择的装甲板

    if ((params_.rmcv_id.robot_id == RobotId::ROBOT_HERO) && !params_.debug) {
            // 英雄使用动态选择
            dynamic_selected_id = select_armor_dynamically(selected_time_diff, selected_total_delay, 
                                                          selected_system_delay, time_diffs, total_delays, adjust_flags);
            if (dynamic_selected_id >= 0) {
                follow_armor_id = dynamic_selected_id;
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Hero using dynamic selection, armor id: %d", follow_armor_id);
            } else {
                // 动态选择失败，使用固定ID
                follow_armor_id = params_.hero_fixed_armor_id;
                RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Hero dynamic selection failed, using fixed armor id: %d", follow_armor_id);
            }
        // 检查装甲板ID有效性
        if (follow_armor_id < 0 || follow_armor_id >= outpost.armor_cnt) {
            follow_armor_id = 0;
            RCLCPP_WARN(rclcpp::get_logger("outpostaim"), "Invalid armor id: %d, using 0", follow_armor_id);
        }
    } else {
        // 哨兵或英雄debug模式：使用动态选择的装甲板
        follow_armor_id = target.armors_index_in_outpost;
    }

    // 根据历史信息记录yaw的最大最小值（老自瞄反陀螺需要的信息）
    // center_ball需要yaw边界值
    yaw_min = imu.yaw;
    yaw_max = imu.yaw;
    bool mono_exist = 1;
    if (outpost.yaw_decrease_history.empty() || outpost.yaw_increase_history.empty())  // 冗余处理
    {
        mono_exist = 0;
    } else {
        yaw_min = outpost.yaw_increase_history.front().second;
        yaw_max = outpost.yaw_decrease_history.front().second;
    }

    Eigen::Vector3d aiming_pos_xyz, aiming_pos_pyd, center_pos_xyz, center_pos_pyd, top_armor;

    // 获取装甲板实际高度
    double armor_height = 0.0;
    if (params_.rmcv_id.robot_id == RobotId::ROBOT_HERO) {
        // 英雄：使用高度映射中的实际高度
        armor_height = outpost.get_armor_height_by_id(follow_armor_id);
    } else {
        // 其他兵种当高度映射建立后使用映射值，否则直接使用观测值
        if(outpost.height_mapping_initialized_){
            armor_height = outpost.get_armor_height_by_id(follow_armor_id);
        } else {
            outpost.aiming_z_filter.update(target.xyz[2]);
            armor_height = outpost.aiming_z_filter.get();
        }
    }
    
    // 解算瞄准点
    Ballistic::BallisticResult follow_ball = calc_ballistic(follow_armor_id, params_.response_delay + prev_latency, aiming_pos_xyz, armor_height);
    Ballistic::BallisticResult center_ball = center_ballistic(center_pos_xyz, armor_height);
    aiming_pos_pyd = xyz2pyd(aiming_pos_xyz);
    center_pos_pyd = xyz2pyd(center_pos_xyz);
    ControlMsg cmd;

    if (follow_ball.fail || center_ball.fail) {
        RCLCPP_WARN(rclcpp::get_logger("outpostaim"), "Ballistic calculation failed");
        return off_cmd;
    }

    // if(fabs(target.yaw_distance_predict) < deg2rad(30.)){
    //     RCLCPP_INFO(this->get_logger(), "min_dis_yaw: %lf, fire", rad2deg(fabs(target.yaw_distance_predict)));
    // }else{
    //     RCLCPP_INFO(this->get_logger(), "min_dis_yaw: %lf, no fire", rad2deg(fabs(target.yaw_distance_predict)));
    // }
    //cv::putText(manager_.recv_detection.img, std::to_string(rad2deg(fabs(target.yaw_distance_predict))), 
    //        cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 3.0, get_cv_color_scalar(CV_COLOR_LABEL::RED));

    // 自动开火条件判断
    double target_dis = get_dis3d(target.xyz);
    double gimbal_error_dis;
    bool should_fire = false;

    // ControlMsg cmd;

    // 主要由哨兵输出伤害
    if ((params_.rmcv_id.robot_id != RobotId::ROBOT_HERO) || params_.debug) {
        // 过滤掉边界外的瞄准点
        if(follow_ball.yaw < yaw_min || follow_ball.yaw > yaw_max){
            RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "aiming out of boundary,aiming yaw %.3lf, yaw_min %.3lf, yaw_max %.3lf", follow_ball.yaw, yaw_min, yaw_max);
            return off_cmd;
        }
        cmd = createControlMsg((float)follow_ball.pitch, (float)follow_ball.yaw, 
                                1, 1, 10, 
                                6 // vision_follow_id暂时直接用6？
                                /*send_cam_mode(params_.cam_mode)*/
            );
        // 为何不用follow_ball的pitch？？
        // gimbal_error_dis = calc_surface_dis_xyz(pyd2xyz(Eigen::Vector3d{imu.pitch, follow_ball.yaw, target_dis}),
        //                                         pyd2xyz(Eigen::Vector3d{imu.pitch, imu.yaw, target_dis}));
        gimbal_error_dis = calc_surface_dis_xyz(pyd2xyz(Eigen::Vector3d{follow_ball.pitch, follow_ball.yaw, target_dis}),
                                                pyd2xyz(Eigen::Vector3d{imu.pitch, imu.yaw, target_dis}));

        // 高度映射建立前使用小阈值以保证准确，因为此时pitch调整到正确值需要一定时间
        bool can_fire = false;
        double yaw_thresh = outpost.op_ckf.const_dis_ * deg2rad(70.) / target_dis; // rad
        if(outpost.height_mapping_initialized_){
            can_fire = fabs(target.yaw_distance_predict) < deg2rad(30.) &&
                                fabs(follow_ball.yaw - center_ball.yaw) < yaw_thresh && mono_exist;
        } else {
            can_fire = fabs(target.yaw_distance_predict) < deg2rad(20.) &&
                                fabs(follow_ball.yaw - center_ball.yaw) < yaw_thresh && mono_exist;
        }

        RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "aiming target dis %.3lf, yaw_dis %.3lf, yaw_f_c %.3lf", target_dis, fabs(target.yaw_distance_predict), fabs(follow_ball.yaw - center_ball.yaw));
        if(can_fire){
            // cv::putText(manager_.recv_detection.img, "Time right", 
            // cv::Point(100, 200), cv::FONT_HERSHEY_SIMPLEX, 3.0, get_cv_color_scalar(CV_COLOR_LABEL::RED));
        }
        if (can_fire && gimbal_error_dis < params_.gimbal_error_dis_thresh) {
            cmd.rate = 15;
            cmd.one_shot_num = 1;
            // cv::putText(manager_.recv_detection.img, "Fire!", 
            // cv::Point(100, 300), cv::FONT_HERSHEY_SIMPLEX, 3.0, get_cv_color_scalar(CV_COLOR_LABEL::RED));
        } else {
            cmd.rate = 0;
            cmd.one_shot_num = 0;
        }
    } else if (params_.rmcv_id.robot_id == RobotId::ROBOT_HERO) {
        // 英雄打前哨站（特化处理）
        double now = manager_.recv_detection.time_stamp;
        if (params_.mode == OUTPOST_AIM || params_.mode == AUTO_AIM) {
            // 英雄AIM模式：瞄准前哨站中心，但使用装甲板实际高度调整pitch
            cmd = createControlMsg(
            (float)center_ball.pitch,  // 使用装甲板高度的pitch
            (float)center_ball.yaw,   // 使用中心的yaw（保持瞄准中心）
            1, 0, 0, 
            6 // vision_follow_id暂时直接用6？
            /*send_cam_mode(params_.cam_mode*/
            );

            // 重新计算开火判断（基于动态选择的时间差）
            if (params_.enable_hero_dynamic_selection && dynamic_selected_id >= 0) {
                // 使用动态选择计算的时间差进行开火判断
                if (selected_time_diff < INFINITY) {

                    double shot_arrive_delay = center_ball.t + params_.shoot_delay;
                    // 时间差接近系统延迟（允许一定误差）
                    double time_error = fabs(selected_time_diff - shot_arrive_delay);
                    // 云台误差检查
                    gimbal_error_dis = calc_surface_dis_xyz(
                        pyd2xyz(Eigen::Vector3d{center_ball.pitch, center_ball.yaw, center_pos_pyd[2]}),
                        pyd2xyz(Eigen::Vector3d{imu.pitch, imu.yaw, center_pos_pyd[2]})
                    );
                    // 判断是否开火
                    bool time_condition = (time_error < params_.timestamp_thresh) && (selected_time_diff > 0);
                    bool gimbal_condition = (gimbal_error_dis < params_.gimbal_error_dis_thresh);
                    bool rate_condition = (now - last_shoot_t > params_.midshot_period);
                    bool data_condition = (outpost.T_solver.N >= 3);  // RLS数据点足够
                    
                    should_fire = time_condition && gimbal_condition && rate_condition && data_condition;
                    
                    if (should_fire) {
                        cmd.one_shot_num = 1;
                        cmd.rate = 10;
                        last_shoot_t = now;
                        RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Hero firing at armor %d (dynamic): time_diff=%.3f, system_delay=%.3f, error=%.3f", 
                                    follow_armor_id, selected_time_diff, shot_arrive_delay, time_error);
                    } else {
                        // 记录不开火的原因
                        RCLCPP_INFO(rclcpp::get_logger("outpostaim"), "Not firing: time_condition=%d, gimbal_condition=%d, rate_condition=%d, data_condition=%d (N=%d)",
                                    time_condition, gimbal_condition, rate_condition, data_condition, outpost.T_solver.N);
                }
                    // 设置开火标志
                    if (follow_armor_id >= 0 && follow_armor_id < 3) {
                        fire_flags[follow_armor_id] = should_fire;
                    }
                }
            }
        }
    }

    if (manager_.robot.autoshoot_rate != 0) {
        cmd.rate = manager_.robot.autoshoot_rate;
    }

    if (cmd.yaw < -M_PI){
        cmd.yaw += 2 * M_PI;
    }else if (cmd.yaw > M_PI){
        cmd.yaw -= 2 * M_PI;
    }

    //下面均为可视化内容（使用新的重投影接口和 manager_.result_img_）
    if (params_.enable_imshow && !params_.debug) {
        cv::Mat &img = manager_.result_img_;
        if (!img.empty()) {
            // 左右边界
            // cv::line(img, projectPointToImage(pyd2xyz(Eigen::Vector3d{0.7, yaw_min, 10.})),
            //         projectPointToImage(pyd2xyz(Eigen::Vector3d{-0.7, yaw_min, 10.})),
            //         cv::Scalar(0, 0, 255), 2);
            // cv::line(img, projectPointToImage(pyd2xyz(Eigen::Vector3d{0.7, yaw_max, 10.})),
            //         projectPointToImage(pyd2xyz(Eigen::Vector3d{-0.7, yaw_max, 10.})),
            //         cv::Scalar(0, 0, 255), 2);

            // // 获取预测的装甲板位置并绘制
            // Outpost::OutpostPosition predicted_positions = outpost.predict_positions(manager_.recv_detection.time_stamp);
            // for (int i = 0; i < outpost.armor_cnt; ++i) {
            //     if (i < (int)predicted_positions.armors_xyz_.size()) {
            //         Eigen::Vector3d armor_pos = predicted_positions.armors_xyz_[i];
            //         cv::Point2f img_point = projectPointToImage(armor_pos);
            //         cv::circle(img, img_point, 5, cv::Scalar(0, 255, 255), -1);
            //         std::string id_text = std::to_string(i);
            //         cv::putText(img, id_text, cv::Point(img_point.x - 5, img_point.y + 20),
            //                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 255), 1);
            //     }
            // }

            // // 中心点
            // cv::Point2f center_img = projectPointToImage(predicted_positions.center_);
            // cv::circle(img, center_img, 5, cv::Scalar(0, 255, 255), -1);

            // if (outpost.height_mapping_initialized_) {
            //     int left_text_x = 10;
            //     int left_text_y = 120;
            //     cv::putText(img, "Armor Height Mapping:", cv::Point(left_text_x, left_text_y),
            //                 cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            //     for (int i = 0; i < 3; ++i) {
            //         left_text_y += 25;
            //         double armor_height = outpost.get_armor_height_by_id(i);
            //         char height_text[100];
            //         snprintf(height_text, sizeof(height_text), "ID %d: %.3f m", i, armor_height);
            //         int thickness = (i == follow_armor_id) ? 2 : 1;
            //         cv::putText(img, height_text, cv::Point(left_text_x, left_text_y),
            //                     cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), thickness);
            //     }
            // }

            if (params_.rmcv_id.robot_id != RobotId::ROBOT_HERO) {
                // 瞄准点
                cv::Point2f aim_point = projectPointToImage(aiming_pos_xyz);
                cv::circle(img, aim_point, 3, cv::Scalar(255, 0, 255), 5);
                std::string height_text = "Z:" + std::to_string(aiming_pos_xyz[2]).substr(0, 5) + "m";
                cv::putText(img, height_text, cv::Point(aim_point.x + 10, aim_point.y - 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1);
            } else {
                // 英雄界面
                cv::circle(img, projectPointToImage(center_pos_xyz), 3, cv::Scalar(255, 0, 255), 5);
                int img_width = img.cols;
                int right_text_x = img_width - 600;
                int right_text_y = 30;
                cv::putText(img, "Next 2 Armors Info:", cv::Point(right_text_x, right_text_y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(255, 255, 255), 2);

                for (int i = 0; i < 2 && i < (int)time_diffs.size(); ++i) {
                    right_text_y += 25;
                    std::string info_text = "next" + std::to_string(i) + ": ";
                    if (i < (int)time_diffs.size() && time_diffs[i] < INFINITY) {
                        char buffer[40];
                        snprintf(buffer, sizeof(buffer), "Td=%.3fs", time_diffs[i]);
                        info_text += buffer;
                    } else {
                        info_text += "Td=INF";
                    }
                    if (i < (int)total_delays.size() && total_delays[i] < INFINITY) {
                        char buffer[40];
                        snprintf(buffer, sizeof(buffer), " Dt=%.3fs", total_delays[i]);
                        info_text += buffer;
                    } else {
                        info_text += " Dt=INF";
                    }
                    bool is_selected = (i < (int)adjust_flags.size()) ? adjust_flags[i] : false;
                    info_text += is_selected ? " [SELECTED]" : "";
                    cv::Scalar text_color = is_selected ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 255, 255);
                    cv::putText(img, info_text, cv::Point(right_text_x, right_text_y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                text_color, 1);
                }

                right_text_y += 40;
                if (dynamic_selected_id >= 0) {
                    std::string selected_text = "Selected Armor: ID" + std::to_string(dynamic_selected_id);
                    if (selected_time_diff < INFINITY) {
                        char buffer[100];
                        snprintf(buffer, sizeof(buffer), " (Td=%.3fs, Dt=%.3fs, Ds=%.3fs)",
                                 selected_time_diff, selected_total_delay, selected_system_delay);
                        selected_text += buffer;
                    }
                    cv::putText(img, selected_text, cv::Point(right_text_x, right_text_y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                cv::Scalar(0, 255, 0), 1);
                    right_text_y += 25;
                    std::string fire_text = "Fire Decision: ";
                    if (selected_time_diff < INFINITY) {
                        Eigen::Vector3d dummy_center;
                        Ballistic::BallisticResult real_ball_res = center_ballistic(dummy_center, outpost.get_armor_height_by_id(dynamic_selected_id));
                        double shot_arrive_delay = params_.response_delay + real_ball_res.t + params_.shoot_delay;
                        char buffer[100];
                        snprintf(buffer, sizeof(buffer), "Ds_fire=%.3fs, Fire=%s", shot_arrive_delay,
                                 fire_flags[dynamic_selected_id] ? "YES" : "NO");
                        fire_text += buffer;
                    } else {
                        fire_text += "N/A";
                    }
                    cv::putText(img, fire_text, cv::Point(right_text_x, right_text_y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                fire_flags[dynamic_selected_id] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1);
                }

                right_text_y += 25;
                std::string fitter_text = "Fitter: N=" + std::to_string(outpost.T_solver.N) + ", yaw_spd " + std::to_string(outpost.common_yaw_spd.get()).substr(0,5);
                cv::putText(img, fitter_text, cv::Point(right_text_x, right_text_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);
            }
        }
    }
    //manager_.outpost_visual.addPoint(aiming_pos_xyz, ColorType::WHITE);

    // manager_.data_visual.pub_single_data(2, cmd.pitch);
    // manager_.data_visual.pub_single_data(3, cmd.yaw);

    return cmd;
}

//---------------------------------GenerateCommand---------------------------