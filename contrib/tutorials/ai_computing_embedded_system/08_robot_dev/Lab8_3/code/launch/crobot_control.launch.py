#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('crobot_control')
    port_name = LaunchConfiguration('port_name', default='/dev/smart_car')
    robot_base = LaunchConfiguration('robot_base', default='2wd')

    motor_yaml = os.path.join(pkg_dir, 'config', 'motor.yaml')

    def robot_base_path(context):
        base = context.perform_substitution(robot_base)
        return os.path.join(pkg_dir, 'config', 'robot_base', base + '.yaml')

    from launch.actions import OpaqueFunction

    def launch_node(context):
        robot_base_yaml = robot_base_path(context)
        return [
            Node(
                package='crobot_control',
                executable='crobot_control_node',
                name='crobot_control',
                output='screen',
                parameters=[
                    {'port_name': context.perform_substitution(port_name)},
                    motor_yaml,
                    robot_base_yaml,
                ],
                remappings=[('cmd_vel', '/cmd_vel')],
            ),
        ]

    return LaunchDescription([
        DeclareLaunchArgument('port_name', default_value='/dev/smart_car', description='Serial port'),
        DeclareLaunchArgument('robot_base', default_value='2wd', description='2wd, 3wo, 4mec'),
        OpaqueFunction(function=launch_node),
    ])
