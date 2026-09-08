#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executors.hpp"
#include "crobot_control/crobot_control.hpp"
#include "crobot_control/crobot_controller_callbacks.hpp"

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("crobot_control");
  crobot_ros::Crobot_Control_Callbacks callbacks(node);
  crobot_ros::Crobot_Control crobot_control(node, callbacks);

  if (!crobot_control.start()) {
    RCLCPP_ERROR(node->get_logger(), "Failed to start crobot control");
    rclcpp::shutdown();
    return -1;
  }

  // 多线程 executor：定时器与 cmd_vel 订阅并行执行，避免定时器占满导致 cmd_vel 无机会执行
  // 显式指定线程数（0=CPU 核数，可能仍不足），保证有足够线程处理 cmd_vel
  const size_t kExecutorThreads = 8;
  rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), kExecutorThreads);
  executor.add_node(node);
  RCLCPP_INFO(node->get_logger(), "MultiThreadedExecutor started with %zu threads",
              executor.get_number_of_threads());
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
