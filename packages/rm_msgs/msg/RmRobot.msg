std_msgs/Header header
string vision_mode
float64 height
float64 z_velocity
RmImu imu
float64 bullet_velocity
uint8 robot_id
# 发射频率推荐值
uint8 autoshoot_rate
# 攻击优先级，写成数组，每个目标指定一个目标类型和一个优先等级，
# 目标共有8个：1,2,3,4,5,7号以及 outpost base（前哨站常规情况下作为屏蔽目标，需要攻击时作为强制目标）
uint8[] priority_type_arr
# 目标类型：强制（若设定了强制目标则只能击打强制目标）；普通（未设置强制目标时可作为目标）；屏蔽（不得作为目标）
uint8[] priority_level_arr
#优先等级： 若同时出现多个可击打目标（如画面中出现多个强制目标，或者未指定强制目标时出现多个普通目标），则选取其中优先级高的
bool right_press  #按下右键
bool switch_cam   #切换相机