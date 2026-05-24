#include "arm_controllers/cartesian_impedance_controller.hpp"

#include <chrono>
#include <fstream>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace arm_controllers {

// ═══════════════════════════════════════════════════════════════════════════
// Interface configuration
// Builds state and command interface strings and compares with the ones
// available — if they exist the controller proceeds
// ═══════════════════════════════════════════════════════════════════════════

controller_interface::InterfaceConfiguration
CartesianImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration cfg;
  cfg.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto& name : joint_names_) {
    cfg.names.push_back(name + "/effort");
  }
  return cfg;
}

controller_interface::InterfaceConfiguration
CartesianImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration cfg;
  cfg.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto& name : joint_names_) {
    cfg.names.push_back(name + "/position");
    cfg.names.push_back(name + "/velocity");
    cfg.names.push_back(name + "/effort");
  }
  return cfg;
}

// ═══════════════════════════════════════════════════════════════════════════
// Lifecycle — on_init
// Declares all ROS2 parameters with default values.
// auto_declare registers the parameter name + default on the parameter server.
// on_configure() will read them and fill the C++ member variables.
// ═══════════════════════════════════════════════════════════════════════════

controller_interface::CallbackReturn CartesianImpedanceController::on_init() {
  try {
    auto_declare<std::vector<std::string>>("joints", {});
    auto_declare<std::string>("end_effector_frame", "fr3_link8");
    auto_declare<std::vector<double>>("stiffness",
        {400.0, 400.0, 400.0, 30.0, 30.0, 30.0});
    auto_declare<std::vector<double>>("damping",
        {40.0, 40.0, 40.0, 10.0, 10.0, 10.0});
    auto_declare<std::vector<double>>("force_gain",
        {0.1, 0.1, 0.1, 0.01, 0.01, 0.01});
    auto_declare<double>("dq_filter_alpha", 0.1);
    auto_declare<std::vector<double>>("null_stiffness",
        {10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0});
    auto_declare<std::vector<double>>("null_target",
        {0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854});
    auto_declare<double>("lambda_base",     1e-6);
    auto_declare<double>("lambda_max",      1e-2);
    auto_declare<double>("sigma_threshold", 0.1);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "on_init failed: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

// ═══════════════════════════════════════════════════════════════════════════
// Lifecycle — on_configure
// Reads parameters, builds gain matrices, loads Pinocchio model from URDF,
// resolves end-effector frame index, creates subscribers.
// ═══════════════════════════════════════════════════════════════════════════

controller_interface::CallbackReturn CartesianImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {

  // ── read joint names and end-effector frame ───────────────────────────────
  joint_names_        = get_node()->get_parameter("joints").as_string_array();
  end_effector_frame_ = get_node()->get_parameter("end_effector_frame").as_string();

  if (static_cast<int>(joint_names_.size()) != kNumJoints) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Expected %d joints, got %zu", kNumJoints, joint_names_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  // ── build 6×6 diagonal gain matrices from YAML 6-element arrays ──────────
  // Lambda reads a named parameter and fills a diagonal 6×6 Eigen matrix.
  // Off-diagonal entries stay zero — decoupled Cartesian axes assumption.
  auto to_diag6 = [&](const std::string& param)
      -> Eigen::Matrix<double, kCartDof, kCartDof> {
    auto v = get_node()->get_parameter(param).as_double_array();
    Eigen::Matrix<double, kCartDof, kCartDof> M =
        Eigen::Matrix<double, kCartDof, kCartDof>::Zero();
    for (int i = 0; i < kCartDof; ++i) M(i, i) = v[i];
    return M;
  };

  K_  = to_diag6("stiffness");
  D_  = to_diag6("damping");
  Kf_ = to_diag6("force_gain");

  // ── 7×7 null space stiffness and home configuration ───────────────────────
  auto ns_v = get_node()->get_parameter("null_stiffness").as_double_array();
  K_null_ = Eigen::Matrix<double, kNumJoints, kNumJoints>::Zero();
  for (int i = 0; i < kNumJoints; ++i) K_null_(i, i) = ns_v[i];

  auto nt_v = get_node()->get_parameter("null_target").as_double_array();
  for (int i = 0; i < kNumJoints; ++i) q_null_target_(i) = nt_v[i];

  // ── scalar parameters for adaptive damping ────────────────────────────────
  lambda_base_     = get_node()->get_parameter("lambda_base").as_double();
  lambda_max_      = get_node()->get_parameter("lambda_max").as_double();
  sigma_threshold_ = get_node()->get_parameter("sigma_threshold").as_double();
  dq_filter_alpha_ = get_node()->get_parameter("dq_filter_alpha").as_double();

  // 
// ── Pinocchio model from /tmp/fr3_resolved.urdf ───────────────────────────
// Written by the launch file synchronously before any node starts.
// Replaces the previous shell extraction which was subject to a race
// condition on first launch.
  try {
    pinocchio::urdf::buildModel("/tmp/fr3_resolved.urdf", model_);
    data_ = pinocchio::Data(model_);
    RCLCPP_INFO(get_node()->get_logger(),
                "Pinocchio model built: %d joints", model_.njoints);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Pinocchio model build failed: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  // ── resolve end-effector frame to integer index ───────────────────────────
  // String lookup is slow — convert once to index, use index at 50Hz
  if (!model_.existFrame(end_effector_frame_)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Frame '%s' not found in URDF", end_effector_frame_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  ee_frame_id_ = model_.getFrameId(end_effector_frame_);

  // ── subscribers ───────────────────────────────────────────────────────────
  // Created last — after all parameters and model are ready.
  // Callbacks run on a separate ROS executor thread, not in update().
  pose_sub_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
      "~/target_pose", rclcpp::SystemDefaultsQoS(),
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        poseCallback(msg);
      });

  wrench_sub_ = get_node()->create_subscription<geometry_msgs::msg::WrenchStamped>(
      "~/disturbance_wrench", rclcpp::SystemDefaultsQoS(),
      [this](const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
        F_disturbance_(0) = msg->wrench.force.x;
        F_disturbance_(1) = msg->wrench.force.y;
        F_disturbance_(2) = msg->wrench.force.z;
        F_disturbance_(3) = msg->wrench.torque.x;
        F_disturbance_(4) = msg->wrench.torque.y;
        F_disturbance_(5) = msg->wrench.torque.z;
      });

  RCLCPP_INFO(get_node()->get_logger(),
              "CartesianImpedanceController configured. EE frame: %s",
              end_effector_frame_.c_str());
  return controller_interface::CallbackReturn::SUCCESS;
}

// ═══════════════════════════════════════════════════════════════════════════
// Lifecycle — on_activate
// Binds hardware interfaces, zeros all state.
// x_des_ is NOT initialized here — we wait for Gazebo to settle first.
// The first outer loop tick (after kSettleTicks) initializes x_des_ to the
// actual FK pose, preventing the startup torque spike.
// ═══════════════════════════════════════════════════════════════════════════

controller_interface::CallbackReturn CartesianImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {

  effort_command_interfaces_.clear();
  position_state_interfaces_.clear();
  velocity_state_interfaces_.clear();
  effort_state_interfaces_.clear();

  for (const auto& name : joint_names_) {
    // ── bind effort command interface ──────────────────────────────────────
    auto it_cmd = std::find_if(
        command_interfaces_.begin(), command_interfaces_.end(),
        [&](const auto& iface) {
          return iface.get_prefix_name() == name &&
                 iface.get_interface_name() == "effort";
        });
    if (it_cmd == command_interfaces_.end()) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Effort command interface not found for %s", name.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    effort_command_interfaces_.emplace_back(*it_cmd);

    // ── bind state interfaces ──────────────────────────────────────────────
    auto bind_state = [&](const std::string& iface_name,
                          auto& container) -> bool {
      auto it = std::find_if(
          state_interfaces_.begin(), state_interfaces_.end(),
          [&](const auto& iface) {
            return iface.get_prefix_name() == name &&
                   iface.get_interface_name() == iface_name;
          });
      if (it == state_interfaces_.end()) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "%s state interface not found for %s",
                     iface_name.c_str(), name.c_str());
        return false;
      }
      container.emplace_back(*it);
      return true;
    };

    if (!bind_state("position", position_state_interfaces_)) {
      return controller_interface::CallbackReturn::ERROR;
    }
    if (!bind_state("velocity", velocity_state_interfaces_)) {
      return controller_interface::CallbackReturn::ERROR;
    }
    if (!bind_state("effort", effort_state_interfaces_)) {
      return controller_interface::CallbackReturn::ERROR;
    }
  }

  // ── zero all internal state ───────────────────────────────────────────────
  q_.setZero();
  dq_.setZero();
  dq_filtered_.setZero();
  tau_ext_raw_.setZero();
  tau_ff_.setZero();
  tau_corr_.setZero();
  tau_null_.setZero();
  tau_gravity_.setZero();
  tau_coriolis_.setZero();
  F_des_.setZero();
  F_ext_.setZero();
  force_error_.setZero();
  F_disturbance_.setZero();
  jacobian_.setZero();
  x_des_     = Eigen::Affine3d::Identity();
  x_current_ = Eigen::Affine3d::Identity();
  xdot_des_.setZero();
  tick_            = 0;
  initialized_     = false;
  setpoint_received_ = false;

  RCLCPP_INFO(get_node()->get_logger(),
              "CartesianImpedanceController activated. "
              "Waiting %d ticks for Gazebo to settle...", kSettleTicks);
  return controller_interface::CallbackReturn::SUCCESS;
}

// ═══════════════════════════════════════════════════════════════════════════
// Lifecycle — on_deactivate
// Send zero torques before releasing hardware interfaces.
// ═══════════════════════════════════════════════════════════════════════════

controller_interface::CallbackReturn CartesianImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  writeEffortCommands(Eigen::Matrix<double, kNumJoints, 1>::Zero());
  return controller_interface::CallbackReturn::SUCCESS;
}

// ═══════════════════════════════════════════════════════════════════════════
// Main update — 1000 Hz
// Called by controller_manager every physics tick.
// Startup phase: zero torques for kSettleTicks to let Gazebo settle joints.
// First outer loop after settling: initializes x_des_ to actual FK pose.
// Normal operation: outer loop at 50Hz, inner loop at 1000Hz.
// ═══════════════════════════════════════════════════════════════════════════

controller_interface::return_type CartesianImpedanceController::update(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {

  readJointState();

  // ── NaN guard — protect against bad state interface values ───────────────
  if (!q_.allFinite() || !dq_.allFinite()) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(),
        *get_node()->get_clock(), 1000, "Non-finite joint state, skipping");
    writeEffortCommands(Eigen::Matrix<double, kNumJoints, 1>::Zero());
    ++tick_;
    return controller_interface::return_type::OK;
  }

  // ── startup settle phase ──────────────────────────────────────────────────
  // Hold zero torques for kSettleTicks (0.5s) to let Gazebo apply the
  // initial joint values from the xacro before we run FK for the first time.
  // Without this, FK runs on an all-zero q_ and sets x_des_ to the vertical
  // fully-extended pose, causing the robot to shoot upward on activation.
  if (tick_ < kSettleTicks) {
    writeEffortCommands(Eigen::Matrix<double, kNumJoints, 1>::Zero());
    ++tick_;
    return controller_interface::return_type::OK;
  }

  // ── outer loop at 50 Hz ───────────────────────────────────────────────────
  if (tick_ % kOuterDecimation == 0) {
    outerLoopUpdate();
  }

  // ── inner loop at 1000 Hz ────────────────────────────────────────────────
  if (initialized_) {
    innerLoopUpdate();
  }

  // ── torque sum ────────────────────────────────────────────────────────────
  // τ = τ_ff + τ_corr + τ_null + Jᵀ·F_disturbance
  // τ_gravity excluded — no-gravity simulation mirrors real FR3 firmware
  // which compensates gravity internally before exposing the torque interface
  Eigen::Matrix<double, kNumJoints, 1> tau_cmd =
      tau_ff_ + tau_corr_ + tau_null_ +
      jacobian_.transpose() * F_disturbance_;

  // ── output NaN guard ──────────────────────────────────────────────────────
  if (!tau_cmd.allFinite()) {
    RCLCPP_ERROR_THROTTLE(get_node()->get_logger(),
        *get_node()->get_clock(), 1000, "Non-finite torque command, zeroing");
    writeEffortCommands(Eigen::Matrix<double, kNumJoints, 1>::Zero());
    ++tick_;
    return controller_interface::return_type::OK;
  }

  writeEffortCommands(saturateTorques(tau_cmd));
  ++tick_;
  return controller_interface::return_type::OK;
}

// ═══════════════════════════════════════════════════════════════════════════
// Hardware I/O
// ═══════════════════════════════════════════════════════════════════════════

void CartesianImpedanceController::readJointState() {
  for (int i = 0; i < kNumJoints; ++i) {
    q_(i)           = position_state_interfaces_[i].get().get_value();
    dq_(i)          = velocity_state_interfaces_[i].get().get_value();
    tau_ext_raw_(i) = effort_state_interfaces_[i].get().get_value();
  }
  
  
}

void CartesianImpedanceController::writeEffortCommands(
    const Eigen::Matrix<double, kNumJoints, 1>& tau) {
  for (int i = 0; i < kNumJoints; ++i) {
    effort_command_interfaces_[i].get().set_value(tau(i));
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Outer loop — 50 Hz
// On the first call after settling: initializes x_des_ to actual FK pose
// (bumpless start — robot holds current position with zero spring force).
// Subsequently: computes FK, dynamics, Cartesian error, impedance wrench,
// null space torque.
// ═══════════════════════════════════════════════════════════════════════════

void CartesianImpedanceController::outerLoopUpdate() {
  dq_filtered_ = dq_filter_alpha_ * dq_ + 
                 (1.0 - dq_filter_alpha_) * dq_filtered_;
  computeFKAndJacobian();
  computeDynamicsCompensation();

  // ── first outer loop tick after settling — initialize x_des_ ─────────────
  // x_des_ is set to the actual current EE pose from FK.
  // This guarantees zero Cartesian error on the first impedance tick,
  // preventing any startup torque spike regardless of initial configuration.
  if (!initialized_) {
    x_des_.linear()      = x_current_.linear();
    x_des_.translation() = x_current_.translation();
    setpoint_received_   = true;
    initialized_         = true;
    RCLCPP_INFO(get_node()->get_logger(),
                "x_des initialized to current EE pose: [%.3f %.3f %.3f]",
                x_des_.translation().x(),
                x_des_.translation().y(),
                x_des_.translation().z());
    return;   // skip impedance this tick — x_des_ just set, error is zero
  }

  auto error = computeCartesianError();
  computeImpedanceWrench(error);
  computeNullSpaceTorque();
}

void CartesianImpedanceController::computeFKAndJacobian() {
  Eigen::VectorXd q_pin  = q_;
  Eigen::VectorXd dq_pin = dq_;

  // forward kinematics — computes positions and velocities for all frames
  pinocchio::forwardKinematics(model_, data_, q_pin, dq_pin);
  // propagate FK results to all named frames including fr3_link8
  pinocchio::updateFramePlacements(model_, data_);

  // extract current EE pose from Pinocchio data
  // oMf = "origin to frame" — homogeneous transform from world to EE
  const auto& oMf          = data_.oMf[ee_frame_id_];
  x_current_.linear()      = oMf.rotation();
  x_current_.translation() = oMf.translation();

  // compute 6×7 geometric Jacobian in world-aligned frame
  // LOCAL_WORLD_ALIGNED: centered at EE, axes aligned with world — consistent
  // with Cartesian error which is also expressed in world coordinates
  pinocchio::Data::Matrix6x J_pin(6, model_.nv);
  J_pin.setZero();
  pinocchio::computeFrameJacobian(model_, data_, q_pin, ee_frame_id_,
                                   pinocchio::LOCAL_WORLD_ALIGNED, J_pin);
  jacobian_ = J_pin;

  // SVD to get smallest singular value — measures proximity to singularity
  // ComputeThinU/V: economical decomposition, 6 singular values only
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      jacobian_, Eigen::ComputeThinU | Eigen::ComputeThinV);
  sigma_min_ = svd.singularValues().minCoeff();
}

Eigen::Matrix<double, CartesianImpedanceController::kCartDof, 1>
CartesianImpedanceController::computeCartesianError() const {
  Eigen::Matrix<double, kCartDof, 1> error;
  // position error — straightforward vector subtraction in world frame
  error.head(3) = x_des_.translation() - x_current_.translation();
  // orientation error — SO(3) log map gives axis-angle vector in ℝ³
  error.tail(3) = computeOrientationError();
  return error;
}

Eigen::Vector3d CartesianImpedanceController::computeOrientationError() const {
    Eigen::Matrix3d R_err = x_des_.rotation() * x_current_.rotation().transpose();

    // angle from the trace: θ = arccos((trace(R) - 1) / 2)
    double cos_theta = (R_err.trace() - 1.0) / 2.0;
    cos_theta = std::clamp(cos_theta, -1.0, 1.0);  // numerical safety
    double theta = std::acos(cos_theta);

    // case 1: R = I → no error
    if (theta < 1e-6) {
        return Eigen::Vector3d::Zero();
    }

    // case 2: θ = π → 180° rotation, axis not unique — pick any valid axis
    if (std::abs(theta - M_PI) < 1e-6) {
        // extract axis from diagonal of (R + I) / 2
        Eigen::Matrix3d B = (R_err + Eigen::Matrix3d::Identity()) / 2.0;
        // axis is the column with largest norm
        Eigen::Vector3d axis = B.col(0);
        if (B.col(1).norm() > axis.norm()) axis = B.col(1);
        if (B.col(2).norm() > axis.norm()) axis = B.col(2);
        axis.normalize();
        return theta * axis;
    }

    // case 3: general — extract axis from skew-symmetric part
    // [n]× = (R - Rᵀ) / (2 sin θ)
    // n = [R₃₂ - R₂₃, R₁₃ - R₃₁, R₂₁ - R₁₂]ᵀ / (2 sin θ)
    Eigen::Vector3d axis;
    axis << R_err(2,1) - R_err(1,2),
            R_err(0,2) - R_err(2,0),
            R_err(1,0) - R_err(0,1);
    axis /= (2.0 * std::sin(theta));

    return theta * axis;
}

void CartesianImpedanceController::computeImpedanceWrench(
    const Eigen::Matrix<double, kCartDof, 1>& error) {
  // velocity error: ẋ_des − J·q̇
  // for static targets ẋ_des = 0 → vel_error = −J·q̇ (damper opposes motion)
  Eigen::Matrix<double, kCartDof, 1> vel_error =
      xdot_des_ - jacobian_ * dq_filtered_ ;

  // impedance law: F = K·e + D·ė (virtual spring-damper wrench)
  F_des_  = K_ * error + D_ * vel_error;

  // map Cartesian wrench to joint torques via Jacobian transpose
  // τ_ff = Jᵀ·F — exact mapping from virtual work principle
  tau_ff_ = jacobian_.transpose() * F_des_;
}

void CartesianImpedanceController::computeDynamicsCompensation() {
  Eigen::VectorXd q_pin  = q_;
  Eigen::VectorXd dq_pin = dq_;

  // gravity torques: τ_g = g(q) — torques needed to hold against gravity
  // disabled in no-gravity simulation but computed for real robot readiness
  tau_gravity_ = pinocchio::computeGeneralizedGravity(model_, data_, q_pin);

  // Coriolis matrix C(q,dq) — velocity-dependent coupling torques
  // tau_coriolis = C·dq — used to strip dynamics from tau_ext_raw_
  pinocchio::computeCoriolisMatrix(model_, data_, q_pin, dq_pin);
  tau_coriolis_ = data_.C * dq_pin;
}

void CartesianImpedanceController::computeNullSpaceTorque() {
  double lambda = computeAdaptiveLambda();
  auto J_pinv   = dampedPseudoinverse(jacobian_, lambda);

  // null space projector N = I − Jᵀ·J†ᵀ
  // any torque multiplied by N produces zero end-effector force
  // primary task (impedance) is never compromised by the secondary task
  Eigen::Matrix<double, kNumJoints, kNumJoints> null_proj =
      Eigen::Matrix<double, kNumJoints, kNumJoints>::Identity() -
      jacobian_.transpose() * J_pinv.transpose();

  // secondary task: joint-space spring pulling toward home configuration
  // projected through N — only the null-space component survives
  Eigen::Matrix<double, kNumJoints, 1> q_error = q_null_target_ - q_;
  tau_null_ = null_proj * (K_null_ * q_error);
}

double CartesianImpedanceController::computeAdaptiveLambda() const {
  // exponential ramp: small λ far from singularity (accurate pseudoinverse)
  // large λ near singularity (regularized, bounded pseudoinverse)
  return lambda_base_ +
         lambda_max_ * std::exp(-sigma_min_ / sigma_threshold_);
}

// ═══════════════════════════════════════════════════════════════════════════
// Inner loop — 1000 Hz
// ═══════════════════════════════════════════════════════════════════════════

void CartesianImpedanceController::innerLoopUpdate() {
  double lambda = computeAdaptiveLambda();
  estimateContactForce(lambda);
  computeForceCorrection();
}

void CartesianImpedanceController::estimateContactForce(double /*lambda*/) {
  // In simulation: tau_ext_raw_ from Gazebo contains the total physics torque
  // (commanded torques + inertia + contact + numerical noise) — not a clean
  // external contact signal. Setting F_ext_ to zero disables force estimation.
  // On the real FR3: franka_hardware provides tau_ext after firmware strips
  // gravity and inertia internally. Restore with:
  //   Eigen::Matrix<double,kNumJoints,1> tau_contact = tau_ext_raw_ - tau_coriolis_;
  //   F_ext_ = dampedPseudoinverse(jacobian_, lambda).transpose() * tau_contact;
  F_ext_.setZero();
}

void CartesianImpedanceController::computeForceCorrection() {
  // force error: difference between desired wrench and estimated contact wrench
  // with F_ext_=0 in simulation, force_error_ = F_des_ always
  force_error_ = F_des_ - F_ext_;
  // map force error to joint torques — same Jᵀ mapping as tau_ff_
  tau_corr_    = jacobian_.transpose() * (Kf_ * force_error_);
}

// ═══════════════════════════════════════════════════════════════════════════
// Math helpers
// ═══════════════════════════════════════════════════════════════════════════

// Damped right pseudoinverse: J† = Jᵀ·(J·Jᵀ + λ²·I)⁻¹
// Returns 7×6 matrix mapping Cartesian forces → joint torques (inverse direction)
// λ²·I prevents J·Jᵀ from becoming singular near kinematic singularities
// Uses J·Jᵀ (6×6) not Jᵀ·J (7×7) — smaller inversion, right pseudoinverse
Eigen::Matrix<double, CartesianImpedanceController::kNumJoints,
              CartesianImpedanceController::kCartDof>
CartesianImpedanceController::dampedPseudoinverse(
    const Eigen::Matrix<double, kCartDof, kNumJoints>& J,
    double lambda) const {
  Eigen::Matrix<double, kCartDof, kCartDof> JJt =
      J * J.transpose() +
      lambda * lambda *
          Eigen::Matrix<double, kCartDof, kCartDof>::Identity();
  return J.transpose() * JJt.inverse();
}

// Clamps each joint torque to the FR3 hardware limits (Nm)
// Joints 0-3: 87 Nm (shoulder/elbow), joints 4-6: 12 Nm (wrist)
// Source: Franka FR3 technical datasheet
Eigen::Matrix<double, CartesianImpedanceController::kNumJoints, 1>
CartesianImpedanceController::saturateTorques(
    const Eigen::Matrix<double, kNumJoints, 1>& tau) const {
  Eigen::Matrix<double, kNumJoints, 1> tau_sat = tau;
  for (int i = 0; i < kNumJoints; ++i) {
    tau_sat(i) = std::clamp(tau(i), -tau_max_[i], tau_max_[i]);
  }
  return tau_sat;
}

// ═══════════════════════════════════════════════════════════════════════════
// ROS callbacks — run on ROS executor thread, NOT in update()
// Thread safety note: x_des_ and F_disturbance_ are written here and read
// in update(). Race condition is benign in simulation (one stale read = one
// tick with slightly wrong error). For real robot, protect with mutex or
// lock-free double buffer.
// ═══════════════════════════════════════════════════════════════════════════

void CartesianImpedanceController::poseCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  // update desired position
  x_des_.translation() << msg->pose.position.x,
                           msg->pose.position.y,
                           msg->pose.position.z;

  // convert quaternion to rotation matrix
  // normalized() handles floating point drift in the incoming quaternion
  Eigen::Quaterniond q(msg->pose.orientation.w,
                       msg->pose.orientation.x,
                       msg->pose.orientation.y,
                       msg->pose.orientation.z);
  x_des_.linear() = q.normalized().toRotationMatrix();

  // static target — no desired velocity
  xdot_des_.setZero();
  setpoint_received_ = true;
}

}  // namespace arm_controllers

PLUGINLIB_EXPORT_CLASS(arm_controllers::CartesianImpedanceController,
                       controller_interface::ControllerInterface)