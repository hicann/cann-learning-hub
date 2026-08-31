#ifndef CROBOT_CONTROL__CROBOT_CONTROL_HPP
#define CROBOT_CONTROL__CROBOT_CONTROL_HPP

#include <chrono>
#include <memory>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "crobot/controller.h"
#include "crobot/controller_callbacks.h"
#include "crobot_control/srv/correction_factor.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace crobot_ros {

class Crobot_Control {
public:
  Crobot_Control(
    rclcpp::Node::SharedPtr node,
    crobot::Controller_Callbacks& cbs);
  ~Crobot_Control();

  void init();
  bool start();

private:
  bool set_motor_param();
  bool set_robot_base();
  void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg);
  void set_correction_factor_func(
    const std::shared_ptr<crobot_control::srv::CorrectionFactor::Request> req,
    std::shared_ptr<crobot_control::srv::CorrectionFactor::Response> resp);
  void reset_odometry_func(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
    std::shared_ptr<std_srvs::srv::Trigger::Response> resp);

  void odom_timer_cb();
  void imu_data_timer_cb();
  void ultrasonic_timer_cb();
  void imu_temp_timer_cb();
  void battery_timer_cb();
  void infrared_timer_cb();

  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Service<crobot_control::srv::CorrectionFactor>::SharedPtr set_correction_factor_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_odometry_srv_;

  rclcpp::TimerBase::SharedPtr odom_timer_;
  rclcpp::TimerBase::SharedPtr imu_data_timer_;
  rclcpp::TimerBase::SharedPtr ultrasonic_timer_;
  rclcpp::TimerBase::SharedPtr imu_temp_timer_;
  rclcpp::TimerBase::SharedPtr battery_timer_;
  rclcpp::TimerBase::SharedPtr infrared_timer_;

  static constexpr auto kOdomPeriod = std::chrono::milliseconds(500);
  static constexpr auto kImuDataPeriod = std::chrono::milliseconds(500);
  static constexpr auto kUltrasonicPeriod = std::chrono::milliseconds(500);
  static constexpr auto kImuTempPeriod = std::chrono::seconds(1);
  static constexpr auto kBatteryPeriod = std::chrono::seconds(1);
  static constexpr auto kInfraredPeriod = std::chrono::milliseconds(500);

  crobot::Controller controller_;
};

}  // namespace crobot_ros

#endif  // CROBOT_CONTROL__CROBOT_CONTROL_HPP
