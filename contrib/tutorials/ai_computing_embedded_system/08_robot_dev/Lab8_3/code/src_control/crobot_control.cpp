#include "crobot_control/crobot_control.hpp"
#include "crobot/robot_base/robot_base.h"
#include <chrono>
#include <iostream>

using namespace crobot;
using namespace crobot_control::srv;

namespace crobot_ros {

Crobot_Control::Crobot_Control(
    rclcpp::Node::SharedPtr node,
    crobot::Controller_Callbacks& cbs)
    : node_(node),
      controller_(cbs) {}

Crobot_Control::~Crobot_Control() {
  odom_timer_.reset();
  imu_data_timer_.reset();
  ultrasonic_timer_.reset();
  imu_temp_timer_.reset();
  battery_timer_.reset();
  infrared_timer_.reset();
}

void Crobot_Control::init() {
  std::string port_name = "/dev/smart_car";
  node_->declare_parameter<std::string>("port_name", port_name);
  node_->get_parameter("port_name", port_name);

  // Nav2 发布 /cmd_vel 为 RELIABLE + VOLATILE；订阅端显式相同 QoS（durability 勿用 TRANSIENT_LOCAL）
  rclcpp::QoS cmd_vel_qos(rclcpp::KeepLast(10));
  cmd_vel_qos.reliable();
  cmd_vel_qos.durability(rclcpp::DurabilityPolicy::Volatile);
  cmd_vel_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", cmd_vel_qos, std::bind(&Crobot_Control::cmd_vel_callback, this, std::placeholders::_1));
  RCLCPP_INFO(node_->get_logger(), "Subscribed to cmd_vel -> /cmd_vel (publish to /cmd_vel to control)");
  set_correction_factor_srv_ = node_->create_service<CorrectionFactor>(
      "set_correction_factor",
      std::bind(&Crobot_Control::set_correction_factor_func, this,
                std::placeholders::_1, std::placeholders::_2));
  reset_odometry_srv_ = node_->create_service<std_srvs::srv::Trigger>(
      "reset_odometry",
      std::bind(&Crobot_Control::reset_odometry_func, this,
                std::placeholders::_1, std::placeholders::_2));

  controller_.init(port_name.c_str(),
                   itas109::BaudRate115200,
                   itas109::ParityNone,
                   itas109::DataBits8,
                   itas109::StopOne,
                   itas109::FlowNone);
}

bool Crobot_Control::start() {
  init();

  if (!controller_.open()) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to start crobot control");
    return false;
  }

  if (!set_motor_param() || !set_robot_base()) {
    return false;
  }

  controller_.send_request(Reset_Odometry_Req{});

  odom_timer_ = node_->create_wall_timer(kOdomPeriod, std::bind(&Crobot_Control::odom_timer_cb, this));
  imu_data_timer_ = node_->create_wall_timer(kImuDataPeriod, std::bind(&Crobot_Control::imu_data_timer_cb, this));
  ultrasonic_timer_ = node_->create_wall_timer(kUltrasonicPeriod, std::bind(&Crobot_Control::ultrasonic_timer_cb, this));
  imu_temp_timer_ = node_->create_wall_timer(kImuTempPeriod, std::bind(&Crobot_Control::imu_temp_timer_cb, this));
  battery_timer_ = node_->create_wall_timer(kBatteryPeriod, std::bind(&Crobot_Control::battery_timer_cb, this));
  infrared_timer_ = node_->create_wall_timer(kInfraredPeriod, std::bind(&Crobot_Control::infrared_timer_cb, this));
  return true;
}

bool Crobot_Control::set_motor_param() {
  int pid_interval = 50;
  int count_per_rev = 3900;
  bool reverse = false;
  node_->declare_parameter<int>("motor.pid_interval", pid_interval);
  node_->declare_parameter<int>("motor.count_per_rev", count_per_rev);
  node_->declare_parameter<bool>("motor.reverse", reverse);
  node_->get_parameter("motor.pid_interval", pid_interval);
  node_->get_parameter("motor.count_per_rev", count_per_rev);
  node_->get_parameter("motor.reverse", reverse);

  controller_.send_request(Set_PID_Interval_Req{static_cast<uint16_t>(pid_interval)});
  controller_.send_request(Set_Motor_Param_Req{static_cast<uint32_t>(count_per_rev), reverse});
  return true;
}

bool Crobot_Control::set_robot_base() {
  int type = 0;
  node_->declare_parameter<int>("robot_base.type", type);
  node_->get_parameter("robot_base.type", type);

  switch (static_cast<Robot_Base_Type>(type)) {
    case Robot_Base_Type::ROBOT_BASE_2WD: {
      Robot_Base_2WD_Param param{};
      node_->declare_parameter<double>("robot_base.radius", 0.0325);
      node_->declare_parameter<double>("robot_base.separation", 0.172);
      node_->get_parameter("robot_base.radius", param.radius);
      node_->get_parameter("robot_base.separation", param.separation);
      controller_.send_request(Set_Robot_Base_2WD_Req{param});
      break;
    }
    case Robot_Base_Type::ROBOT_BASE_3WO: {
      Robot_Base_3WO_Param param{};
      node_->declare_parameter<double>("robot_base.radius", 0.029);
      node_->declare_parameter<double>("robot_base.distance", 0.105);
      node_->get_parameter("robot_base.radius", param.radius);
      node_->get_parameter("robot_base.distance", param.distance);
      controller_.send_request(Set_Robot_Base_3WO_Req{param});
      break;
    }
    case Robot_Base_Type::ROBOT_BASE_4WD: {
      Robot_Base_4WD_Param param{};
      node_->declare_parameter<double>("robot_base.radius", 0.0325);
      node_->declare_parameter<double>("robot_base.separation", 0.172);
      node_->get_parameter("robot_base.radius", param.radius);
      node_->get_parameter("robot_base.separation", param.separation);
      controller_.send_request(Set_Robot_Base_4WD_Req{param});
      break;
    }
    case Robot_Base_Type::ROBOT_BASE_4MEC: {
      Robot_Base_4MEC_Param param{};
      node_->declare_parameter<double>("robot_base.radius", 0.049);
      node_->declare_parameter<double>("robot_base.distance_x", 0.095);
      node_->declare_parameter<double>("robot_base.distance_y", 0.1);
      node_->get_parameter("robot_base.radius", param.radius);
      node_->get_parameter("robot_base.distance_x", param.distance_x);
      node_->get_parameter("robot_base.distance_y", param.distance_y);
      controller_.send_request(Set_Robot_Base_4MEC_Req{param});
      break;
    }
  }
  return true;
}

void Crobot_Control::cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
  RCLCPP_INFO(node_->get_logger(),
              "cmd_vel recv: linear.x=%.3f linear.y=%.3f angular.z=%.3f",
              msg->linear.x, msg->linear.y, msg->angular.z);
  controller_.send_request(
      Set_Velocity_Req{static_cast<float>(msg->linear.x),
                      static_cast<float>(msg->linear.y),
                      static_cast<float>(msg->angular.z)});
}

void Crobot_Control::set_correction_factor_func(
    const std::shared_ptr<CorrectionFactor::Request> req,
    std::shared_ptr<CorrectionFactor::Response> resp) {
  controller_.send_request(Set_Correction_Factor_Req{req->linear_x, req->linear_y, req->angular});
  resp->success = true;
}

void Crobot_Control::reset_odometry_func(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> resp) {
  controller_.send_request(Reset_Odometry_Req{});
  resp->success = true;
}

void Crobot_Control::odom_timer_cb() {
  controller_.send_request(Get_Odometry_Req{});
}

void Crobot_Control::imu_data_timer_cb() {
  controller_.send_request(Get_IMU_Data_Req{});
}

void Crobot_Control::ultrasonic_timer_cb() {
  controller_.send_request(Get_Ultrasonic_Range_Req{});
}

void Crobot_Control::imu_temp_timer_cb() {
  controller_.send_request(Get_IMU_Temperature_Req{});
}

void Crobot_Control::battery_timer_cb() {
  controller_.send_request(Get_Battery_Voltage_Req{});
}

void Crobot_Control::infrared_timer_cb() {
  controller_.send_request(Get_Infrared_Distance_Req{});
}

}  // namespace crobot_ros
