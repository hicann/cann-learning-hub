#!/usr/bin/env python3
"""将 robot_description 参数发布到 /robot_description 话题，供 RViz2 等使用。"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String


def main(args=None):
    rclpy.init(args=args)
    node = Node('robot_description_publisher')

    node.declare_parameter('robot_description', '')
    robot_description = node.get_parameter('robot_description').get_parameter_value().string_value

    # TRANSIENT_LOCAL + RELIABLE：类似 ROS1 的 latched，后连上的 RViz2 也能收到
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    pub = node.create_publisher(String, '/robot_description', qos)

    # 等有订阅者再发一次，并周期发布以便后连上的客户端能收到
    def timer_cb():
        pub.publish(String(data=robot_description))

    timer = node.create_timer(1.0, timer_cb)
    timer_cb()  # 立即发一次

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
