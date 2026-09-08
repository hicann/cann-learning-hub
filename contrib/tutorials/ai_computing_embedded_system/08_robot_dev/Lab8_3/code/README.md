# code 目录说明

本目录包含实验8.3 昇腾香橙派嵌入式智能机器人实验的 ROS2 工程代码，基于香橙派 AIPro 开发板实现机器人运动控制、SLAM 建图、自主定位与路径导航。

## 子目录说明

### config/ — ROS2 参数配置文件

| 文件 | 说明 |
| --- | --- |
| `motor.yaml` | 电机参数配置 |
| `robot_base_2wd.yaml` | 两轮差速机器人底盘参数 |
| `amcl_params.yaml` | AMCL 自主定位参数 |
| `amcl_nav2_params.yaml` | Nav2 AMCL 定位参数 |
| `costmap_common_params.yaml` | 代价地图通用参数 |
| `global_costmap_params.yaml` | 全局代价地图参数 |
| `local_costmap_params.yaml` | 局部代价地图参数 |
| `nav2_params.yaml` | Nav2 导航参数 |
| `teb_local_planner_params.yaml` | TEB 局部规划器参数 |
| `mapper_params_online_async.yaml` | SLAM 在线异步建图参数 |

### launch/ — ROS2 启动文件

| 文件 | 说明 |
| --- | --- |
| `crobot.launch.py` | 机器人系统总启动文件 |
| `crobot_control.launch.py` | 机器人控制节点启动 |
| `crobot_description.launch.py` | 机器人模型描述启动 |
| `slam_toolbox.launch.py` | SLAM 建图启动 |
| `localization.launch.py` | 自主定位启动 |
| `map_server.launch.py` | 地图服务启动 |
| `navigation.launch.py` | 导航启动 |
| `save_map.launch.py` | 保存地图启动 |
| `lslidar_serial.launch.py` | 激光雷达驱动启动 |

### src_control/ — 机器人控制源码

| 文件 | 说明 |
| --- | --- |
| `crobot_control.cpp` / `.hpp` | 机器人控制节点主程序 |
| `controller_core.cpp` / `.h` | 控制器核心逻辑 |
| `crobot_controller_callbacks.cpp` / `.hpp` | 控制器回调函数 |
| `crobot_control_node.cpp` | 控制节点入口 |

### src_lslidar/ — 激光雷达驱动源码

| 文件 | 说明 |
| --- | --- |
| `lslidar_driver.cc` | 激光雷达驱动核心 |
| `lslidar_driver_node.cc` | 激光雷达驱动节点入口 |

### urdf/ — 机器人模型描述

| 文件 | 说明 |
| --- | --- |
| `edu_robot.urdf` | 机器人 URDF 模型描述文件 |
| `robot_description_publisher.py` | 机器人模型描述发布器 |
