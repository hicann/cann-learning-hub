#!/usr/bin/env python3
# 对应 ROS1：crobot_navigation/launch/navigation.launch
# 整机 + 地图 + AMCL + 规划（bringup + map_server + amcl + move_base 等价）

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_bringup = get_package_share_directory('crobot_bringup')
    pkg_nav = get_package_share_directory('crobot_navigation')
    nav2_params = os.path.join(pkg_nav, 'params', 'nav2_params.yaml')

    map_name = LaunchConfiguration('map_name', default='map')
    robot_model = LaunchConfiguration('robot_model', default='edu_robot')
    base_port = LaunchConfiguration('base_port', default='/dev/smart_car')
    robot_base = LaunchConfiguration('robot_base', default='2wd')
    laser_port = LaunchConfiguration('laser_port', default='/dev/wheeltec_lidar')
    lidar_name = LaunchConfiguration('lidar_name', default='N10')

    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, 'launch', 'crobot.launch.py')
        ),
        launch_arguments=[
            ('robot_model', robot_model),
            ('base_port', base_port),
            ('robot_base', robot_base),
            ('laser_port', laser_port),
            ('lidar_name', lidar_name),
        ],
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav, 'launch', 'localization.launch.py')
        ),
        launch_arguments=[('map_name', map_name)],
    )

    controller_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_params],
    )

    planner_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[nav2_params],
    )

    behavior_node = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[nav2_params],
    )

    bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[nav2_params],
    )

    lifecycle_manager_nav = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'node_names': [
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
            ],
            'autostart': True,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('map_name', default_value='map',
                             description='地图名（不含 .yaml）'),
        DeclareLaunchArgument('robot_model', default_value='edu_robot'),
        DeclareLaunchArgument('base_port', default_value='/dev/smart_car'),
        DeclareLaunchArgument('robot_base', default_value='2wd'),
        DeclareLaunchArgument('laser_port', default_value='/dev/wheeltec_lidar'),
        DeclareLaunchArgument('lidar_name', default_value='N10'),
        bringup_launch,
        localization_launch,
        controller_node,
        planner_node,
        behavior_node,
        bt_navigator_node,
        lifecycle_manager_nav,
    ])
