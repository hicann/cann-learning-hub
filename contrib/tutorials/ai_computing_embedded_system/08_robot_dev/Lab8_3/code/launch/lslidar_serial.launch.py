#!/usr/bin/env python3
# ROS2 launch: 单串口雷达

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    port_name = LaunchConfiguration('port_name', default='/dev/lidar')
    lidar_name = LaunchConfiguration('lidar_name', default='N10')

    params = {
        'serial_port': port_name,
        'lidar_name': lidar_name,
        'interface_selection': 'serial',
        'frame_id': 'laser_link',
        'scan_topic': 'scan',
        'angle_disable_min': 0.0,
        'angle_disable_max': 0.0,
        'min_range': 0.15,
        'max_range': 100.0,
        'use_gps_ts': False,
        'compensation': False,
        'pubScan': True,
        'pubPointCloud2': False,
        'high_reflection': False,
    }

    driver_node = Node(
        package='lslidar_driver',
        executable='lslidar_driver_node',
        name='lslidar_driver_node',
        output='screen',
        parameters=[params],
    )

    return LaunchDescription([
        DeclareLaunchArgument('port_name', default_value='/dev/lidar', description='雷达串口设备'),
        DeclareLaunchArgument('lidar_name', default_value='N10', description='雷达型号: M10, N10, N10_P, L10 等'),
        driver_node,
    ])
