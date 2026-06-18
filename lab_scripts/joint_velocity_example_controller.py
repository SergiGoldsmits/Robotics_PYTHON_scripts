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

// ===================== UPDATE (ACCELERATION-LIMITED RECONSTRUCTION) =====================
// Runs at the controller_manager rate (e.g. 1 kHz); commands arrive slower (e.g. 100 Hz).
// Each tick we move the output toward the latest command by at most max_joint_accel*dt,
// so the signal sent to the FR3 has continuous velocity and bounded acceleration ->
// prevents the motion-generator velocity/acceleration discontinuity reflexes.
// A watchdog (reusing interp_step_ as a "ticks since last command" counter) ramps the
// output to zero if the command stream stalls, instead of holding a velocity.
// ===================== UPDATE (JERK-LIMITED RECONSTRUCTION) =====================
// Bounds velocity slope (accel) AND accel slope (jerk), so the signal sent to the FR3
// has continuous velocity AND continuous acceleration -> no accel-discontinuity reflex.
// State: filtered_velocity_commands_ = velocity, interpolated_velocity_ = acceleration.
controller_interface::return_type JointVelocityExampleController::update(
    const rclcpp::Time&,
    const rclcpp::Duration& period) {

  std::lock_guard<std::mutex> lock(command_mutex_);

  double dt = period.seconds();
  if (dt <= 0.0 || dt > 0.1) {
    dt = 0.001;
  }

  // --- tunables ---
  const double max_accel = 2.0;    // [rad/s^2]
  const double max_jerk  = 40.0;   // [rad/s^3]  <-- the new, key limit
  const int    watchdog_ticks = 100;

  if (interp_step_ < 1000000) {
    interp_step_++;
  }
  const bool stale = (interp_step_ > watchdog_ticks);

  const double da_max = max_jerk * dt;   // max change in acceleration this tick

  for (int i = 0; i < num_joints; i++) {
    const double v_goal = stale ? 0.0 : velocity_commands_[i];

    // velocity error -> desired acceleration, capped at max_accel
    double a_des = (v_goal - filtered_velocity_commands_[i]) / dt;
    if (a_des >  max_accel) a_des =  max_accel;
    if (a_des < -max_accel) a_des = -max_accel;

    // jerk-limit: move current acceleration toward a_des by at most da_max
    double a = interpolated_velocity_[i];   // current acceleration (state)
    double da = a_des - a;
    if (da >  da_max) da =  da_max;
    if (da < -da_max) da = -da_max;
    a += da;
    interpolated_velocity_[i] = a;

    // integrate acceleration -> velocity
    filtered_velocity_commands_[i] += a * dt;
    command_interfaces_[i].set_value(filtered_velocity_commands_[i]);
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
              "JointVelocityExampleController ready with acceleration-limited reconstruction.");
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

  // start "stale" so the arm stays at zero until a real command arrives (soft start)
  interp_step_ = kInterpSteps + 1000;

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
            }

            interp_step_ = 0;  // feed the watchdog (ticks-since-command counter)
          });

  RCLCPP_INFO(get_node()->get_logger(),
              "Controller activated: soft start, acceleration-limited tracking.");
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
