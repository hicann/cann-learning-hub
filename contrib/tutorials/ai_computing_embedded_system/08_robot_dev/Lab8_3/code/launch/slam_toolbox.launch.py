#!/usr/bin/env python3
# ROS2 建图：机器人模型 + 雷达 + slam_toolbox；可选是否启动底盘（不接底盘时仅雷达建图）

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_description = get_package_share_directory('crobot_description')
    pkg_control = get_package_share_directory('crobot_control')
    pkg_lidar = get_package_share_directory('lslidar_driver')
    pkg_slam = get_package_share_directory('crobot_slam')
    pkg_slam_toolbox = get_package_share_directory('slam_toolbox')

    robot_model = LaunchConfiguration('robot_model', default='edu_robot')
    base_port = LaunchConfiguration('base_port', default='/dev/smart_car')
    robot_base = LaunchConfiguration('robot_base', default='2wd')
    laser_port = LaunchConfiguration('laser_port', default='/dev/wheeltec_lidar')
    lidar_name = LaunchConfiguration('lidar_name', default='N10')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    use_base = LaunchConfiguration('use_base', default='true')

    slam_params_file = os.path.join(pkg_slam, 'params', 'slam_toolbox', 'mapper_params_online_async.yaml')

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
        condition=IfCondition(use_base),
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

    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_slam_toolbox, 'launch', 'online_async_launch.py')
        ),
        launch_arguments=[
            ('slam_params_file', slam_params_file),
            ('use_sim_time', use_sim_time),
        ],
    )

    # 不接底盘时没有 odom->base_footprint，补一条静态变换，让 slam_toolbox 能正常发布 map
    static_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='odom_to_base_footprint',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_footprint'],
        condition=UnlessCondition(use_base),
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_model', default_value='edu_robot', description='URDF 模型名'),
        DeclareLaunchArgument('base_port', default_value='/dev/smart_car', description='底盘串口'),
        DeclareLaunchArgument('robot_base', default_value='2wd', description='底盘类型'),
        DeclareLaunchArgument('laser_port', default_value='/dev/wheeltec_lidar', description='雷达串口'),
        DeclareLaunchArgument('lidar_name', default_value='N10', description='雷达型号'),
        DeclareLaunchArgument('use_sim_time', default_value='false', description='是否使用仿真时间'),
        DeclareLaunchArgument('use_base', default_value='true',
                             description='是否启动底盘控制；不接底盘时设为 false，仅用雷达建图'),
        description_launch,
        control_launch,
        lidar_launch,
        static_odom_tf,
        slam_toolbox_launch,
    ])
