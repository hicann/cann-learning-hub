#!/usr/bin/env python3
# 对应 ROS1：crobot_navigation/launch/navigation.launch 中的地图部分
#   <arg name="map_name" default="map" />
#   <arg name="map_file" default="$(find crobot_navigation)/maps/$(arg map_name).yaml"/>
#   <node pkg="map_server" name="map_server" type="map_server" args="$(arg map_file)" />
# 说明：/virtual_wall/map_name 在 ROS1 中于同一 launch 内设置；ROS2 将在完整 navigation.launch 中
#       向 virtual_wall 节点传入 map_name 参数，此处仅启动 map_server。

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_map_server(context):
    pkg_dir = get_package_share_directory('crobot_navigation')
    map_name = context.perform_substitution(LaunchConfiguration('map_name', default='map'))
    map_path = os.path.join(pkg_dir, 'maps', map_name + '.yaml')

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{'yaml_filename': map_path}],
        output='screen',
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{'node_names': ['map_server'], 'autostart': True}],
    )

    return [map_server_node, lifecycle_manager]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_name', default_value='map',
                             description='地图名（不含 .yaml），对应 maps/<map_name>.yaml'),
        OpaqueFunction(function=_launch_map_server),
    ])
