#include "rm_msgs/msg/control.hpp"

#pragma once
#pragma pack(1)

struct autoaim_send_to_port_data_t{
    float pitch;
    float yaw;
    uint8_t flag;
    uint8_t one_shot_num;
    uint8_t rate;
    uint8_t vision_follow_id;
    uint8_t cam_mode;

    void fromControlMsg(const rm_msgs::msg::Control& msg){
        pitch = msg.imu.pitch;
        yaw = msg.imu.yaw;
        flag = msg.booster_enable;
        vision_follow_id = msg.vision_follow_id;
        one_shot_num = 3;
        rate = 10;
        cam_mode = 0;
    }
};

struct autoaim_recv_from_port_data_t{
    // 上下小陀螺
    float height;
    float z_velocity;  // 相对于gimbal的平行面的速度
    
    // imu发送绝对的rpy姿态
    float roll;
    float pitch;       // 以枪管(、pitch轴)而言
    float yaw;
    /* mode定义
        0: 关闭自瞄（步兵、无人机松开右键）（英雄按F切换自瞄开关）
        1: 普通自瞄
        2: 前哨站模式 (英雄带有击打前哨站旋转装甲板能力)
        3: 小风车模式（步兵专用）
        4: 大风车模式（步兵专用）
        5: 吊射模式（击打前哨站和基地的顶装甲板）（英雄专用）
        */

    uint8_t mode;
    // id定义：与官方定义相同
    uint8_t robot_id;
    // v定义:实际弹速
    float bullet_speed;

    // 外部指定射频
    uint8_t autoshoot_rate;

    // 目标选择指令
    //  目标类型：强制（若设定了强制目标则只能击打强制目标）；普通（未设置强制目标时可作为目标）；屏蔽（不得作为目标）
    uint8_t priority_type_arr[8];
    // 优先等级：
    // 若同时出现多个可击打目标（如画面中出现多个强制目标，或者未指定强制目标时出现多个普通目标），则选取其中优先级高的
    uint8_t priority_level_arr[8];
    // 切换相机
    uint8_t switch_cam;
};

// 保证声明顺序和发送数据顺序一致
struct protocol_header_t{
    uint8_t start;
    uint8_t protocol_id;
    //id: 0x01 自瞄发给电控
    //id: 0x03 电控发给自瞄
    //id: 0x04 电控发给英雄吊射
};

struct protocol_tail_t{
    uint16_t crc16;
    uint8_t end;
};

#pragma pack()