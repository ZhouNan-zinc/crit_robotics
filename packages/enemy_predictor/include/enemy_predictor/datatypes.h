#pragma once

#include <cstdint>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <string>
#include <unordered_map>

#define UNKNOWN_ID (255)
    
#define Vecd(x) Eigen::Vector<double, x>
#define Matd(x,y) Eigen::Matrix<double, x, y>

// ==== vision_mode ====
enum VisionMode {
    NO_AIM = 0,    // 无瞄准
    AUTO_AIM,  // 普通自瞄，打车，打基地
    OUTPOST_AIM,   // 前哨站
    OUTPOST_LOB,   // 吊射前哨站
    S_WM,      // 小风车
    B_WM,      // 大风车
    LOB,       // 吊射
    HALT,      // 停机
    AUTOLOB,   // 自动吊射
    Unknown,   // 未知
};

VisionMode get_vision_mode(uint8_t mode);

VisionMode string2vision_mode(const std::string& str);

std::string vision_mode2string(VisionMode mode);

// ==== armor / rmcv id ====

enum RmColor{
#define RM_COLOR(x,y) x = y,
#ifndef RM_COLOR
#define RM_COLOR(...)
#endif 

RM_COLOR(BLUE, 0)
RM_COLOR(RED, 1)
RM_COLOR(WHITE, 2)
RM_COLOR(PURPLE, 3)
RM_COLOR(COLOR_ERROR,4)

#undef RM_COLOR
};

inline static std::string RmColorToString(RmColor c){
    switch(c){
#define RM_COLOR(x,y) case x: return #x;
#ifndef RM_COLOR
#define RM_COLOR(...)
#endif 

RM_COLOR(BLUE, 0)
RM_COLOR(RED, 1)
RM_COLOR(WHITE, 2)
RM_COLOR(PURPLE, 3)
RM_COLOR(COLOR_ERROR,4)

#undef RM_COLOR
    }
}

inline static std::string RmColorToSimpleString(RmColor c){
// #define GET_FIRST_CHAR(x) #x[0]
    std::string s;
    switch(c){
#define RM_COLOR(x,y) case x: s = #x; break;
#ifndef RM_COLOR
#define RM_COLOR(...)
#endif 

RM_COLOR(BLUE, 0)
RM_COLOR(RED, 1)
RM_COLOR(WHITE, 2)
RM_COLOR(PURPLE, 3)
RM_COLOR(COLOR_ERROR,4)

#undef RM_COLOR
    }
    return s.substr(0, 1);
}

static const std::unordered_map<int, RmColor> id2color = {
    {0, RmColor::BLUE},
    {1, RmColor::RED},
    {2, RmColor::WHITE},
    {3, RmColor::PURPLE},
};

// 官方定义/电控定义
enum RobotIdDji {
    RED_HERO = 1,
    RED_ENGINEER,
    RED_STANDARD_1,
    RED_STANDARD_2,
    RED_STANDARD_3 ,
    RED_AERIAL,
    RED_SENTRY,
    BLUE_HERO = 101,
    BLUE_ENGINEER,
    BLUE_STANDARD_1,
    BLUE_STANDARD_2,
    BLUE_STANDARD_3,
    BLUE_AERIAL,
    BLUE_SENTRY,
};

// 无颜色机器人id
enum RobotId{
    ROBOT_SENTRY = 0,
    ROBOT_HERO,
    ROBOT_ENGINEER,
    ROBOT_STANDARD_1,
    ROBOT_STANDARD_2,
    ROBOT_STANDARD_3,
    ROBOT_AERIAL,    
    ROBOT_ERROR,
};

// 机器人自身id
struct RmcvId{
    // int id;
    RobotId robot_id;
    RmColor color;

    RmcvId(){}

    RmcvId(RobotIdDji _robot_id_dji){
        color = RmColor::BLUE;
        if(_robot_id_dji <= RED_SENTRY) color = RmColor::RED;
        switch(_robot_id_dji){
            case RED_HERO:
            case BLUE_HERO:
                robot_id = ROBOT_HERO;
                break;
            case RED_ENGINEER:
            case BLUE_ENGINEER:
                robot_id = ROBOT_ENGINEER;
                break;
            case RED_STANDARD_1:
            case BLUE_STANDARD_1:
                robot_id = ROBOT_STANDARD_1;
                break;
            case RED_STANDARD_2:
            case BLUE_STANDARD_2:
                robot_id = ROBOT_STANDARD_2;
                break;
            case RED_STANDARD_3:
            case BLUE_STANDARD_3:
                robot_id = ROBOT_STANDARD_3;
                break;
            case RED_AERIAL:
            case BLUE_AERIAL:
                robot_id = ROBOT_AERIAL;
                break;
            case RED_SENTRY:
            case BLUE_SENTRY:
                robot_id = ROBOT_SENTRY;
                break;
            default:
                RCLCPP_ERROR(rclcpp::get_logger("Ballestic"), "Error RobotIdDji"); 
        }
    }

    bool operator==(const RmcvId& other){
        return robot_id == other.robot_id && color == other.color;
    }

    bool operator!=(const RmcvId& other){
        return !(*this==other);
    }
};

// RMCV定义
enum ArmorType {
#define ARMOR_TYPE(x,y) x = y,
#ifndef ARMOR_TYPE
#define ARMOR_TYPE(...)
#endif

ARMOR_TYPE(SENTRY, 0)
ARMOR_TYPE(HERO, 1)
ARMOR_TYPE(ENGINEER, 2)
ARMOR_TYPE(STANDARD_1, 3)
ARMOR_TYPE(STANDARD_2, 4)
ARMOR_TYPE(STANDARD_3, 5)
ARMOR_TYPE(OUTPOST, 6)
ARMOR_TYPE(BASE, 7)
ARMOR_TYPE(TOP, 8)

#undef ARMOR_TYPE



};

inline static std::string ArmorTypeToString(ArmorType t){
    switch(t){
#define ARMOR_TYPE(x,y) case x: return #x;
#ifndef ARMOR_TYPE
#define ARMOR_TYPE(...)
#endif

ARMOR_TYPE(SENTRY, 0)
ARMOR_TYPE(HERO, 1)
ARMOR_TYPE(ENGINEER, 2)
ARMOR_TYPE(STANDARD_1, 3)
ARMOR_TYPE(STANDARD_2, 4)
ARMOR_TYPE(STANDARD_3, 5)
ARMOR_TYPE(OUTPOST, 6)
ARMOR_TYPE(BASE, 7)
ARMOR_TYPE(TOP, 8)

#undef ARMOR_TYPE
    }
}

inline static std::string ArmorTypeToSimpleString(ArmorType t){
    std::string s1;
    std::string s2;
    switch(t){
#define ARMOR_TYPE(x,y) case x: {s1 = #x; s2 = #y; break;}
#ifndef ARMOR_TYPE
#define ARMOR_TYPE(...)
#endif

ARMOR_TYPE(SENTRY, 0)
ARMOR_TYPE(HERO, 1)
ARMOR_TYPE(ENGINEER, 2)
ARMOR_TYPE(STANDARD_1, 3)
ARMOR_TYPE(STANDARD_2, 4)
ARMOR_TYPE(STANDARD_3, 5)
ARMOR_TYPE(OUTPOST, 6)
ARMOR_TYPE(BASE, 7)
ARMOR_TYPE(TOP, 8)

#undef ARMOR_TYPE



    }
    return s1.substr(0, 1) + s2;
}

uint8_t armor_type2vision_follow_id(ArmorType type);

// 击打装甲板id
struct ArmorId{
    int id;
    ArmorType armor_type;
    RmColor armor_color;

    ArmorId(){}

    ArmorId(ArmorType _type, RmColor _color): armor_type(_type), armor_color(_color){
        id = static_cast<int>(_color) * 9 + static_cast<int>(_type);
    }

    ArmorId& operator=(int _id){
        id = _id;
        return *this;
    }

    bool operator==(const ArmorId& other) const{
        return armor_type == other.armor_type && armor_color == other.armor_color;
    }

    bool operator!=(const ArmorId& other) const{
        return !(*this == other);
    }

    std::string get_string_id() const{
        auto color_str = RmColorToString(armor_color);
        auto type_str = ArmorTypeToString(armor_type);
        return color_str + "_" + type_str;
    }

    std::string get_simple_string_id() const{
        auto color_str = RmColorToSimpleString(armor_color);
        auto type_str = ArmorTypeToSimpleString(armor_type);
        return color_str + "_" + type_str;
    }
};

// ==== 相机模式 ====

enum CameraMode : uint8_t{
    SINGLE = 0,  // 单相机，未区分长短焦；
    SHORTFOCAL,
    TELEPHOTO,
    UNSET,  // 未知当前启动的相机或无相机启动
};

uint8_t send_cam_mode(CameraMode mode);

// ==== 随每一帧图像发送的信息 ====
class FrameInfo {
public:
    VisionMode mode;
    bool right_press;
    CameraMode cam_mode;
    RobotIdDji robot_id;
    double bullet_velocity;
    std::vector<double> k;
    std::vector<double> d;
    double delay;
    std::string serialize();
    void deserialize(const std::string&);
};


struct NgxyPose{
    // 位置
    double x;
    double y;
    double z;
    // 姿态 rad
    double roll;
    double pitch;
    double yaw;
    // 参考坐标系
    std::string frame_id;

    inline Eigen::Vector3d get_xyz_vec() const{
        return Eigen::Vector3d(x, y, z);
    }
    inline Eigen::Vector3d get_pyd_vec() const{
        Eigen::Vector3d pyd;
        pyd[0] = -atan2(z, sqrt(x * x + y * y));
        pyd[1] = atan2(y, x);
        pyd[2] = sqrt(x * x + y * y + z * z);
        return pyd;
    }
    inline Eigen::Vector3d get_rpy_vec() const{
        return Eigen::Vector3d(roll, pitch, yaw);
    }
    inline Eigen::Matrix<double, 6, 1> get_pose_vec() const{
        return Eigen::Matrix<double, 6, 1>(x, y, z, roll, pitch, yaw);
    }
};

// 或许可用cv_bridge替代
int encoding2mat_type(const std::string &encoding);

// void rotate_img(double angle, cv::Mat &src, cv::Mat &dst);
void pub_float_data(rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr _publisher, double data);
