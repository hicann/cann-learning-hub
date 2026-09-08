#!/usr/bin/env python3
# ROS2 保存地图：调用 nav2_map_server 的 map_saver_cli
# 建图时 /map 由 slam_toolbox 发布；保存前请确保建图节点在运行。

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    map_name = LaunchConfiguration('map_name', default='map')
    # 默认保存到用户目录，可覆盖为绝对路径（不含扩展名）
    map_path = LaunchConfiguration('map_path', default='/tmp/crobot_map')

    # nav2_map_server 的 map_saver_cli：从 /map 话题保存，需在 SLAM 运行时执行
    # -f 输出路径（不含扩展名），-t 地图话题（slam_toolbox 发布 /map）
    save_cmd = ExecuteProcess(
        cmd=['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', map_path, '-t', 'map'],
        shell=False,
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('map_name', default_value='map', description='地图名称（旧参数，保留兼容）'),
        DeclareLaunchArgument('map_path', default_value='/tmp/crobot_map',
                             description='保存路径（不含 .pgm/.yaml），如 /tmp/crobot_map 或 /home/xxx/maps/my_map'),
        save_cmd,
    ])
