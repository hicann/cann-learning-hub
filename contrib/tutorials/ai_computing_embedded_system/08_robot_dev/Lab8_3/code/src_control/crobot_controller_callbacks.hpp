#ifndef CROBOT_CONTROL__CROBOT_CONTROLLER_CALLBACKS_HPP
#define CROBOT_CONTROL__CROBOT_CONTROLLER_CALLBACKS_HPP

#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/u_int16.hpp"
#include "crobot/controller_callbacks.h"
#include "tf2_ros/transform_broadcaster.h"

namespace crobot_ros {

class Crobot_Control_Callbacks : public crobot::Controller_Callbacks {
public:
  explicit Crobot_Control_Callbacks(rclcpp::Node::SharedPtr node);

  void set_pid_interval_callback() override;
  void set_motor_param_callback() override;
  void set_robot_base_callback() override;
  void set_correction_factor_callback() override;
  void set_velocity_callback() override;
  void reset_odometry_callback() override;

  void get_odometry_callback(const crobot::Get_Odometry_Resp& resp) override;
  void get_imu_temperature_callback(const crobot::Get_IMU_Temperature_Resp& resp) override;
  void get_imu_data_callback(const crobot::Get_IMU_Data_Resp& resp) override;
  void get_ultrasonic_range_callback(const crobot::Get_Ultrasonic_Range_Resp& resp) override;
  void get_battery_voltage_callback(const crobot::Get_Battery_Voltage_Resp& resp) override;
  void get_infrared_distance_callback(const crobot::Get_Infrared_Distance_Resp& resp) override;

private:
  void init();

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr imu_temperature_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_raw_data_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt16>::SharedPtr ultrasonic_range_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr battery_voltage_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr infrared_distance_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> odom_base_tb_;
};

}  // namespace crobot_ros

#endif  // CROBOT_CONTROL__CROBOT_CONTROLLER_CALLBACKS_HPP
