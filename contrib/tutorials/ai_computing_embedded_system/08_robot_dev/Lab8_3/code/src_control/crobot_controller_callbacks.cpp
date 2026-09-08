#include "crobot_control/crobot_controller_callbacks.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

using namespace crobot;

namespace crobot_ros {

Crobot_Control_Callbacks::Crobot_Control_Callbacks(rclcpp::Node::SharedPtr node)
    : node_(node) {
  init();
}

void Crobot_Control_Callbacks::set_pid_interval_callback() {}
void Crobot_Control_Callbacks::set_velocity_callback() {}
void Crobot_Control_Callbacks::set_robot_base_callback() {}
void Crobot_Control_Callbacks::set_motor_param_callback() {}
void Crobot_Control_Callbacks::set_correction_factor_callback() {}
void Crobot_Control_Callbacks::reset_odometry_callback() {}

void Crobot_Control_Callbacks::get_odometry_callback(const Get_Odometry_Resp& resp) {
  rclcpp::Time current_time = node_->now();

  nav_msgs::msg::Odometry odom;
  odom.header.stamp = current_time;
  odom.header.frame_id = "odom";
  odom.child_frame_id = "base_footprint";

  odom.pose.pose.position.x = resp.position_x;
  odom.pose.pose.position.y = resp.position_y;
  odom.pose.pose.position.z = 0.0;

  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, resp.direction);
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();

  odom.twist.twist.linear.x = resp.linear_x;
  odom.twist.twist.linear.y = resp.linear_y;
  odom.twist.twist.linear.z = 0.0;
  odom.twist.twist.angular.x = 0.0;
  odom.twist.twist.angular.y = 0.0;
  odom.twist.twist.angular.z = resp.angular_z;

  odom_pub_->publish(odom);

  geometry_msgs::msg::TransformStamped tfs;
  tfs.header.stamp = current_time;
  tfs.header.frame_id = "odom";
  tfs.child_frame_id = "base_footprint";
  tfs.transform.translation.x = resp.position_x;
  tfs.transform.translation.y = resp.position_y;
  tfs.transform.translation.z = 0.0;
  tfs.transform.rotation = odom.pose.pose.orientation;
  odom_base_tb_->sendTransform(tfs);
}

void Crobot_Control_Callbacks::get_imu_temperature_callback(const Get_IMU_Temperature_Resp& resp) {
  std_msgs::msg::Float32 temperature;
  temperature.data = resp.temperature;
  imu_temperature_pub_->publish(temperature);
}

void Crobot_Control_Callbacks::get_imu_data_callback(const Get_IMU_Data_Resp& resp) {
  sensor_msgs::msg::Imu imu_raw_data;
  imu_raw_data.header.stamp = node_->now();
  imu_raw_data.header.frame_id = "imu_link";

  imu_raw_data.linear_acceleration.x = resp.accel_x * 9.7833f;
  imu_raw_data.linear_acceleration.y = resp.accel_y * 9.7833f;
  imu_raw_data.linear_acceleration.z = resp.accel_z * 9.7833f;
  imu_raw_data.angular_velocity.x = resp.angular_x * static_cast<float>(M_PI) / 180.0f;
  imu_raw_data.angular_velocity.y = resp.angular_y * static_cast<float>(M_PI) / 180.0f;
  imu_raw_data.angular_velocity.z = resp.angular_z * static_cast<float>(M_PI) / 180.0f;

  imu_raw_data_pub_->publish(imu_raw_data);
}

void Crobot_Control_Callbacks::get_ultrasonic_range_callback(const Get_Ultrasonic_Range_Resp& resp) {
  std_msgs::msg::UInt16 range;
  range.data = resp.range;
  ultrasonic_range_pub_->publish(range);
}

void Crobot_Control_Callbacks::get_battery_voltage_callback(const Get_Battery_Voltage_Resp& resp) {
  std_msgs::msg::Float32 voltage;
  voltage.data = resp.voltage;
  battery_voltage_pub_->publish(voltage);
}

void Crobot_Control_Callbacks::get_infrared_distance_callback(const Get_Infrared_Distance_Resp& resp) {
  std_msgs::msg::Float32 distance;
  distance.data = resp.distance;
  infrared_distance_pub_->publish(distance);
}

void Crobot_Control_Callbacks::init() {
  odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
  imu_temperature_pub_ = node_->create_publisher<std_msgs::msg::Float32>("imu_temperature", 10);
  imu_raw_data_pub_ = node_->create_publisher<sensor_msgs::msg::Imu>("imu_raw_data", 10);
  ultrasonic_range_pub_ = node_->create_publisher<std_msgs::msg::UInt16>("ultrasonic_range", 10);
  battery_voltage_pub_ = node_->create_publisher<std_msgs::msg::Float32>("battery_voltage", 10);
  infrared_distance_pub_ = node_->create_publisher<std_msgs::msg::Float32>("infrared_distance", 10);
  odom_base_tb_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
}

}  // namespace crobot_ros
