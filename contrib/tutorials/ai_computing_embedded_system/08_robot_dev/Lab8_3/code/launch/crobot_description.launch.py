#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_nodes(context):
    pkg_dir = get_package_share_directory('crobot_description')
    robot_model = context.perform_substitution(LaunchConfiguration('robot_model', default='edu_robot'))
    urdf_path = os.path.join(pkg_dir, 'urdf', robot_model + '.urdf')
    with open(urdf_path) as f:
        robot_description = f.read()

    return [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
        ),
        # 将 robot_description 发布到 /robot_description 话题，RViz2 RobotModel 可从该话题或参数获取
        Node(
            package='crobot_description',
            executable='robot_description_publisher.py',
            name='robot_description_publisher',
            parameters=[{'robot_description': robot_description}],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_model', default_value='edu_robot', description='URDF name without .urdf'),
        OpaqueFunction(function=_launch_nodes),
    ])
