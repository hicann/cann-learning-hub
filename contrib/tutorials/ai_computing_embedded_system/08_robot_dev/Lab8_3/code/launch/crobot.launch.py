#!/usr/bin/env python3
# ROS2 整机启动：机器人模型 + 底盘控制 + 雷达

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    pkg_description = get_package_share_directory('crobot_description')
    pkg_control = get_package_share_directory('crobot_control')
    pkg_lidar = get_package_share_directory('lslidar_driver')

    robot_model = LaunchConfiguration('robot_model', default='edu_robot')
    base_port = LaunchConfiguration('base_port', default='/dev/smart_car')
    robot_base = LaunchConfiguration('robot_base', default='2wd')
    laser_port = LaunchConfiguration('laser_port', default='/dev/wheeltec_lidar')
    lidar_name = LaunchConfiguration('lidar_name', default='N10')

    description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_description, 'launch', 'crobot_description.launch.py')
        ),
        launch_arguments=[('robot_model', robot_model)],
    )

    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_control, 'launch', 'crobot_control.launch.py')
        ),
        launch_arguments=[
            ('port_name', base_port),
            ('robot_base', robot_base),
        ],
    )

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_lidar, 'launch', 'lslidar_serial.launch.py')
        ),
        launch_arguments=[
            ('port_name', laser_port),
            ('lidar_name', lidar_name),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_model', default_value='edu_robot',
                             description='URDF 模型名（不含 .urdf）'),
        DeclareLaunchArgument('base_port', default_value='/dev/smart_car',
                             description='底盘串口'),
        DeclareLaunchArgument('robot_base', default_value='2wd',
                             description='底盘类型: 2wd, 3wo, 4mec'),
        DeclareLaunchArgument('laser_port', default_value='/dev/lidar',
                             description='雷达串口'),
        DeclareLaunchArgument('lidar_name', default_value='N10',
                             description='雷达型号: N10, M10 等'),
        description_launch,
        control_launch,
        lidar_launch,
    ])
