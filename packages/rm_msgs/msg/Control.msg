std_msgs/Header header
float32 pitch
float32 yaw
# 发射n颗弹，记为一次发射行为
uint8 flag
uint8 one_shot_num
# 一秒中发射行为的频率
uint8 rate
# 相机模式
uint8 cam_mode
# 瞄准的机器人id
uint8 vision_follow_id

#        vision_follow_id 解释
#        普通自瞄模式:返回当前锁定的敌军类型
#        0 哨兵
#        1 英雄
#        2 工程
#        3 4 5 步兵3 4 5号
#        6 前哨站
#        7 基地
#        8 英雄：前哨站或者基地的顶装甲
#        15 无目标或目标过远
#        风车自瞄模式：vision_follow_id无意义

#        旧flag 解释
#        0 无目标或目标过远
#        1 跟随目标，不开火，等同于rate,one_shot_num = 0
#        2 自动开火：普通自瞄模式下按一定频率开火，风车模式下收到一个2开火一次（自瞄代码保证不会有许多2）
#        3 高频率开火：普通自瞄模式下按较高的频率开火，默认斩杀线为5.2米，斩杀线范围可以调节；风车模式下保证不会出现3信号

#        新flag 解释
#        0 不控云台
#        1 跟得上目标，控制云台
