#include <franka_example_controllers/default_robot_behavior_utils.hpp>
#include <franka_example_controllers/fr3/joint_velocity_example_controller.hpp>
#include <franka_example_controllers/robot_utils.hpp>

#include <algorithm>
#include <cmath>
#include <string>

#include <Eigen/Eigen>

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
JointVelocityExampleController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) +
                           "/velocity");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointVelocityExampleController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) +
                           "/position");
    config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) +
                           "/velocity");
  }
  return config;
}

// ===================== UPDATE (COSINE INTERPOLATION) =====================
controller_interface::return_type JointVelocityExampleController::update(
    const rclcpp::Time&,
    const rclcpp::Duration&) {

  constexpr double PI = 3.141592653589793;

  std::lock_guard<std::mutex> lock(command_mutex_);

  double s = (kInterpSteps <= 1)
               ? 1.0
               : static_cast<double>(interp_step_) /
                 static_cast<double>(kInterpSteps);

  double w = 0.5 * (1.0 - std::cos(PI * s));  // cosine easing

  for (int i = 0; i < num_joints; i++) {

    interpolated_velocity_[i] =
        (1.0 - w) * start_velocity_[i] + w * target_velocity_[i];

    filtered_velocity_commands_[i] = interpolated_velocity_[i];

    command_interfaces_[i].set_value(filtered_velocity_commands_[i]);
  }

  if (interp_step_ < kInterpSteps) {
    interp_step_++;
  }

  return controller_interface::return_type::OK;
}

// ===================== INIT =====================
CallbackReturn JointVelocityExampleController::on_init() {
  try {
    auto_declare<std::string>("arm_prefix", "");
    auto_declare<bool>("gazebo", false);
    auto_declare<std::string>("robot_description", "");
  } catch (const std::exception& e) {
    fprintf(stderr, "Init error: %s\n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

// ===================== CONFIGURE =====================
CallbackReturn JointVelocityExampleController::on_configure(
    const rclcpp_lifecycle::State&) {

  is_gazebo = get_node()->get_parameter("gazebo").as_bool();

  auto client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  client->wait_for_service();

  auto future = client->get_parameters({"robot_description"});
  auto result = future.get();

  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  }

  robot_type_ =
      robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());

  arm_prefix_ = get_node()->get_parameter("arm_prefix").as_string();
  arm_prefix_ = arm_prefix_.empty() ? "" : arm_prefix_ + "_";

  RCLCPP_INFO(get_node()->get_logger(),
              "JointVelocityExampleController ready with cosine interpolation.");
  return CallbackReturn::SUCCESS;
}

// ===================== ACTIVATE =====================
CallbackReturn JointVelocityExampleController::on_activate(
    const rclcpp_lifecycle::State&) {

  std::lock_guard<std::mutex> lock(command_mutex_);

  velocity_commands_.fill(0.0);
  start_velocity_ = velocity_commands_;
  target_velocity_ = velocity_commands_;
  filtered_velocity_commands_ = velocity_commands_;

  interp_step_ = kInterpSteps;

  velocity_sub_ =
      get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
          "~/commands", 10,
          [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {

            if (static_cast<int>(msg->data.size()) != num_joints) {
              return;
            }

            std::lock_guard<std::mutex> lock(command_mutex_);

            for (int i = 0; i < num_joints; ++i) {
              velocity_commands_[i] =
                  std::clamp(msg->data[i], -kMaxVelocity, kMaxVelocity);

              // NEW: snapshot for interpolation
              start_velocity_[i] = filtered_velocity_commands_[i];
              target_velocity_[i] = velocity_commands_[i];
            }

            interp_step_ = 0;  // restart ramp
          });

  RCLCPP_INFO(get_node()->get_logger(),
              "Controller activated with cosine ramp smoothing.");
  return CallbackReturn::SUCCESS;
}

// ===================== DEACTIVATE =====================
CallbackReturn JointVelocityExampleController::on_deactivate(
    const rclcpp_lifecycle::State&) {

  std::lock_guard<std::mutex> lock(command_mutex_);

  velocity_commands_.fill(0.0);
  filtered_velocity_commands_.fill(0.0);

  for (int i = 0; i < num_joints; i++) {
    command_interfaces_[i].set_value(0.0);
  }

  return CallbackReturn::SUCCESS;
}

}  // namespace franka_example_controllers
