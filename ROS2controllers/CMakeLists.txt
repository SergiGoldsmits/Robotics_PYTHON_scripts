cmake_minimum_required(VERSION 3.8)
project(arm_controllers_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(urdf REQUIRED)
set(BUILD_WITH_PINOCCHIO_PYTHON OFF)
set(PINOCCHIO_WITH_PYTHON_BINDINGS OFF)
find_package(pinocchio REQUIRED)

add_library(arm_controllers_cpp SHARED
  src/joint_position_controller.cpp
  src/joint_effort_controller.cpp
  src/cartesian_impedance_controller.cpp
)

target_include_directories(arm_controllers_cpp PUBLIC
  include
)

ament_target_dependencies(arm_controllers_cpp
  controller_interface
  hardware_interface
  rclcpp
  rclcpp_lifecycle
  pluginlib
  urdf
  std_msgs
  rclcpp_action
  control_msgs
  geometry_msgs
  Eigen3
  orocos_kdl
  kdl_parser
  trajectory_msgs
)

target_link_libraries(arm_controllers_cpp
  pinocchio::pinocchio
)

pluginlib_export_plugin_description_file(controller_interface arm_controllers.xml)

install(TARGETS arm_controllers_cpp
  DESTINATION lib
)
install(
  DIRECTORY config backup include src
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)
install(DIRECTORY include/
  DESTINATION include
)

install(PROGRAMS
  scripts/cartesian_pid_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()