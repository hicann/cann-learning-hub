/*
 * This file is part of lslidar driver.
 *
 * The driver is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The driver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with the driver.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <rclcpp/rclcpp.hpp>
#include <lslidar_driver/lslidar_driver.h>
#include <csignal>

volatile sig_atomic_t flag = 1;

static void my_handler(int sig)
{
  flag = 0;
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("lslidar_driver_node");

  lslidar_driver::LslidarDriver driver(node.get());
  if (!driver.initialize()) {
    RCLCPP_ERROR(node->get_logger(), "Cannot initialize lslidar driver...");
    return 0;
  }

  std::signal(SIGINT, my_handler);
  while (rclcpp::ok() && flag && driver.polling()) {
    rclcpp::spin_some(node);
  }

  rclcpp::shutdown();
  return 0;
}
