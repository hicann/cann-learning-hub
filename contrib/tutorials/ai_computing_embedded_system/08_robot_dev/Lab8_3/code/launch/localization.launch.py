#!/usr/bin/env python3
# 对应 ROS1：navigation.launch 中的 Map server + AMCL 部分
#   map_server + include/amcl.launch（即 amcl 节点 + 参数）
# 用于“仅定位”：加载地图并运行 AMCL，不启动规划。

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_localization(context):
    pkg_dir = get_package_share_directory('crobot_navigation')
    map_name = context.perform_substitution(LaunchConfiguration('map_name', default='map'))
    map_path = os.path.join(pkg_dir, 'maps', map_name + '.yaml')
    amcl_params = os.path.join(pkg_dir, 'params', 'amcl', 'amcl_nav2_params.yaml')

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{'yaml_filename': map_path}],
        output='screen',
    )

    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[amcl_params],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'node_names': ['map_server', 'amcl'], 'autostart': True}],
    )

    return [map_server_node, amcl_node, lifecycle_manager]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_name', default_value='map',
                             description='地图名（不含 .yaml），对应 maps/<map_name>.yaml'),
        OpaqueFunction(function=_launch_localization),
    ])

