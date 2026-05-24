#pragma once

#include <array>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <hardware_interface/loaned_command_interface.hpp>
#include <hardware_interface/loaned_state_interface.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>

namespace arm_controllers {

class CartesianImpedanceController : public controller_interface::ControllerInterface {
public:
  CartesianImpedanceController() = default;
  ~CartesianImpedanceController() override = default;

  // ── ros2_control lifecycle ────────────────────────────────────────────────
  controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(
      const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn on_activate(
      const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
      const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::return_type update(
      const rclcpp::Time& time,
      const rclcpp::Duration& period) override;

private:
  // ── constants ────────────────────────────────────────────────────────────
  static constexpr int kNumJoints = 7;
  static constexpr int kCartDof   = 6;   // 3 translation + 3 rotation
  // outer loop runs every kOuterDecimation ticks (1000 / 50 = 20)
  static constexpr int kOuterDecimation = 5;
  // ticks to wait at startup before running FK — lets Gazebo settle joints
  // into their initial values from the xacro (500 ticks = 0.5s)
  static constexpr int kSettleTicks = 500;

  // ── joint names ──────────────────────────────────────────────────────────
  std::vector<std::string> joint_names_;
  std::string              end_effector_frame_;

  // ── Pinocchio model ──────────────────────────────────────────────────────
  pinocchio::Model model_;
  pinocchio::Data  data_;
  pinocchio::FrameIndex ee_frame_id_;

  // ── hardware interface handles ───────────────────────────────────────────
  // indexed 0..6 matching joint_names_
  std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>>
      effort_command_interfaces_;
  std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>>
      position_state_interfaces_;
  std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>>
      velocity_state_interfaces_;
  std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>>
      effort_state_interfaces_;   // τ_ext readable — inner force loop

  // ── joint state (refreshed every tick) ───────────────────────────────────
  Eigen::Matrix<double, kNumJoints, 1> q_;
  Eigen::Matrix<double, kNumJoints, 1> dq_;
  Eigen::Matrix<double, kNumJoints, 1> tau_ext_raw_;   // raw from state iface

  // ── desired Cartesian setpoint (written by ROS subscriber) ───────────────
  // protected by atomic copy — subscriber runs in a separate executor thread
  Eigen::Affine3d                      x_des_;
  Eigen::Matrix<double, kCartDof, 1>   xdot_des_;     // desired EE velocity
  bool                                 setpoint_received_{false};

  // ── outer loop outputs (computed at 50 Hz, read by inner loop) ───────────
  Eigen::Matrix<double, kCartDof, kNumJoints> jacobian_;       // J(q)  6×7
  Eigen::Matrix<double, kCartDof, 1>          F_des_;           // desired wrench
  Eigen::Matrix<double, kNumJoints, 1>        tau_ff_;          // Jᵀ·F_des
  Eigen::Matrix<double, kNumJoints, 1>        tau_gravity_;     // g(q)
  Eigen::Matrix<double, kNumJoints, 1>        tau_coriolis_;    // C(q,dq)·dq
  Eigen::Affine3d                             x_current_;       // FK result
  double                                      sigma_min_{1.0};  // smallest SV of J

  // ── inner loop outputs (computed at 1000 Hz) ─────────────────────────────
  Eigen::Matrix<double, kNumJoints, 1>  tau_corr_;    // force feedback correction
  Eigen::Matrix<double, kCartDof, 1>    F_ext_;       // estimated contact wrench
  Eigen::Matrix<double, kCartDof, 1>    force_error_; // F_des − F_ext
  // disturbance wrench for compliance testing — published externally via topic
  // maps through Jᵀ and adds directly to torque sum
  Eigen::Matrix<double, kCartDof, 1>    F_disturbance_;

  // ── null-space torque (computed at outer rate) ────────────────────────────
  Eigen::Matrix<double, kNumJoints, 1> tau_null_;
  Eigen::Matrix<double, kNumJoints, 1> q_null_target_;  // desired null-space config

  // ── gains (loaded from YAML) ─────────────────────────────────────────────
  Eigen::Matrix<double, kCartDof, kCartDof>   K_;   // Cartesian stiffness  6×6
  Eigen::Matrix<double, kCartDof, kCartDof>   D_;   // Cartesian damping    6×6
  Eigen::Matrix<double, kCartDof, kCartDof>   Kf_;  // force feedback gain  6×6
  Eigen::Matrix<double, kNumJoints, kNumJoints> K_null_;  // null-space stiffness 7×7
  double lambda_base_{1e-6};       // damped pseudoinverse base regularisation
  double lambda_max_{1e-2};        // max regularisation near singularity
  double sigma_threshold_{0.1};    // singular value threshold for avoidance

  // ── torque saturation limits (Nm, per joint) ─────────────────────────────
  std::array<double, kNumJoints> tau_max_{87, 87, 87, 87, 12, 12, 12};
    // low-pass filtered joint velocity — used in damping term instead of raw dq_
    Eigen::Matrix<double, kNumJoints, 1> dq_filtered_{
    Eigen::Matrix<double, kNumJoints, 1>::Zero()};
    double dq_filter_alpha_{0.1};  // 0.0 = no update, 1.0 = no filter
  // ── tick counter ─────────────────────────────────────────────────────────
  int  tick_{0};
  bool initialized_{false};   // bumpless transfer flag

  // ── ROS subscribers ───────────────────────────────────────────────────────
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_sub_;

  // ── private methods ───────────────────────────────────────────────────────

  // read all joint state interfaces into q_, dq_, tau_ext_raw_
  void readJointState();

  // write saturated torques to effort command interfaces
  void writeEffortCommands(const Eigen::Matrix<double, kNumJoints, 1>& tau);

  // ── outer loop (50 Hz) ───────────────────────────────────────────────────
  void outerLoopUpdate();

  // forward kinematics + Jacobian via Pinocchio → x_current_, jacobian_
  void computeFKAndJacobian();

  // 6D Cartesian error: [e_pos; e_rot] where e_rot uses SO(3) log map
  Eigen::Matrix<double, kCartDof, 1> computeCartesianError() const;

  // orientation error: log( R_des · R_current^T )  →  ℝ³
  Eigen::Vector3d computeOrientationError() const;

  // impedance wrench  F = K·e + D·(ẋ_des − J·dq)
  void computeImpedanceWrench(const Eigen::Matrix<double, kCartDof, 1>& error);

  // gravity + Coriolis via Pinocchio  → tau_gravity_, tau_coriolis_
  void computeDynamicsCompensation();

  // null-space torque to avoid singularities and keep preferred config
  void computeNullSpaceTorque();

  // SVD of J to get sigma_min_ and adapt lambda
  double computeAdaptiveLambda() const;

  // ── inner loop (1000 Hz) ─────────────────────────────────────────────────
  void innerLoopUpdate();

  // strip dynamics from τ_ext_raw, map to Cartesian via damped pseudoinverse
  // F_ext = J^{-T}_damped · (tau_ext_raw - tau_gravity - tau_coriolis)
  void estimateContactForce(double lambda);

  // force feedback correction  τ_corr = Jᵀ · Kf · (F_des − F_ext)
  void computeForceCorrection();

  // damped pseudoinverse  J · (JᵀJ + λI)^{-1}   →  7×6
  Eigen::Matrix<double, kNumJoints, kCartDof>
  dampedPseudoinverse(const Eigen::Matrix<double, kCartDof, kNumJoints>& J,
                      double lambda) const;

  // saturate torque vector element-wise
  Eigen::Matrix<double, kNumJoints, 1>
  saturateTorques(const Eigen::Matrix<double, kNumJoints, 1>& tau) const;

  // callback: update x_des_ from incoming PoseStamped
  void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
};

}  // namespace arm_controllers