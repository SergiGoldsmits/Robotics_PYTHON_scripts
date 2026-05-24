"""
fr3_launch_hardware.py
------------------
Bringup for the real Franka FR3 using franka_hardware.
Mirrors the simulation architecture exactly — same controllers,
same topics, same command architecture — only the hardware plugin
and the way controller_manager is started differ from fr3_sim.launch.py.

Key differences vs fr3_sim.launch.py:
  1. No Gazebo — ros2_control_node runs as a standalone node
  2. URDF is fr3_hardware.xacro — uses franka_hardware/FrankaHardwareInterface
  3. Controllers YAML is Franka's fr3_ros_controllers.yaml (effort mode, franka_robot_state_broadcaster)
  4. franka_robot_state_broadcaster is spawned — required by franka_hardware
  5. use_sim_time is explicitly false — real wall clock, not Gazebo clock

Usage:
  ros2 launch arm_description fr3_launch_hardware.py robot_ip:=<IP>
  ros2 launch arm_description fr3_launch_hardware.py robot_ip:=192.168.1.1 use_rviz:=true

Prerequisites:
  - FCI must be enabled on the robot web UI: https://<robot_ip> → Activate FCI
  - PC must be on the same network as the robot
  - libfranka version must match robot firmware
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    # ── Launch arguments ──────────────────────────────────────────────────────
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        description='IP address of the real FR3 robot (e.g. 192.168.1.1)',
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Launch RViz2 for visualisation',
    )
    
    # Which arm controller to load. Declared as a launch argument so you can
    # swap controllers at launch time without editing the file, e.g.:
    #   ros2 launch arm_description fr3_real.launch.py controller:=fr3_cartesian_impedance_controller
    # Default matches the effort JointTrajectoryController in Franka's YAML.
    declare_controller = DeclareLaunchArgument(
        'controller',
        default_value='cartesian_impedance_controller',
        description='Name of the arm controller to load (must exist in fr3_ros_controllers.yaml)',
    )

    robot_ip   = LaunchConfiguration('robot_ip')
    use_rviz   = LaunchConfiguration('use_rviz')
    controller = LaunchConfiguration('controller')

    # ── Paths ─────────────────────────────────────────────────────────────────
    arm_share    = get_package_share_directory('arm_description')
    franka_moveit_share = get_package_share_directory('franka_fr3_moveit_config')
    arm_controllers_share = get_package_share_directory('arm_description')
    # CHANGED from simulation:
    # We use Franka's own fr3_ros_controllers.yaml instead of our sim yaml.
    # This YAML:
    #   - declares franka_robot_state_broadcaster (mandatory for franka_hardware)
    #   - configures fr3_arm_controller in effort mode with PD gains
    #   - sets update_rate: 1000 Hz to match the real robot's FCI loop
    franka_controllers_yaml = os.path.join(
        franka_moveit_share, 'config', 'fr3_ros_controllers.yaml'
    )
    my_controllers_yaml = os.path.join(
        arm_controllers_share, 'config', 'fr3_controllers_franka.yaml'
)

    rviz_config = os.path.join(arm_share, 'rviz', 'fr3.rviz')

    # ── Robot description (xacro → URDF string at launch time) ───────────────
    # use_gazebo:=false  → activates franka_hardware/FrankaHardwareInterface
    #                      and excludes the <gazebo> plugin block from the URDF
    # use_fake_hardware:=false → connects to the real robot, not a mock
    # robot_ip is forwarded into the <hardware> block so franka_hardware
    # knows which robot to connect to over the network.
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ',
        os.path.join(arm_share, 'urdf', 'fr3_hardware.xacro'),
        ' hand:=false',
        ' robot_ip:=', robot_ip,
    ])
    robot_description = {'robot_description': ParameterValue(robot_description_content, value_type=str)}
    # ── robot_state_publisher ─────────────────────────────────────────────────
    # Identical role to simulation — publishes /robot_description and TF.
    # use_sim_time MUST be false on the real robot (wall clock, not Gazebo clock).
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            robot_description,
            {'use_sim_time': False},   # CHANGED: was True in simulation
        ],
    )

    # ── ros2_control_node (standalone controller manager) ────────────────────
    # CHANGED from simulation:
    # In simulation, gz_ros2_control creates controller_manager implicitly
    # as a Gazebo plugin. On the real robot there is no Gazebo — we launch
    # the controller_manager directly as a ROS2 node.
    #
    # It loads franka_hardware/FrankaHardwareInterface from the URDF,
    # which opens the libfranka connection to the real robot over ethernet.
    #
    # use_sim_time MUST be false here too — franka_hardware uses wall clock
    # for its 1 kHz control loop timestamps.
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        output='screen',
        parameters=[
            franka_controllers_yaml, my_controllers_yaml,         # CHANGED: Franka's yaml, not sim yaml
            {'use_sim_time': False},   # CHANGED: real wall clock
        ],
    )

    # ── Controller spawners ───────────────────────────────────────────────────
    # Sequenced with OnProcessExit — each spawner only fires after the previous
    # one's process exits with success (exit code 0 = controller active).
    #
    # This is strictly better than TimerAction:
    #   TimerAction  — fires after a fixed number of seconds regardless of
    #                  whether the previous step actually finished. Too short
    #                  and controllers race. Too long and bringup is slow.
    #   OnProcessExit — fires only when the target process exits successfully.
    #                   Zero guessing, works on fast and slow machines equally.
    #
    # Sequence: franka_state_broadcaster → joint_state_broadcaster → arm_controller
    # franka_robot_state_broadcaster must come first: franka_hardware will not
    # activate joint interfaces until it confirms this broadcaster is running.
    # Each subsequent spawner is registered as an OnProcessExit handler on the
    # previous one — the chain fires automatically as each step completes.

    # Step 1 — launched directly (no dependency, starts as soon as
    # controller_manager is ready — spawner blocks internally until it is).
    # ADDED vs simulation: not present in fr3_sim.launch.py.
    franka_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'franka_robot_state_broadcaster',
            '--controller-manager-timeout', '30',
            '--controller-manager', '/controller_manager',
        ],
        output='screen',
    )

    # Step 2 — fires only after franka_state_broadcaster_spawner exits (code 0).
    # Publishes /joint_states — same topic your scripts and RViz subscribe to.
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager-timeout', '30',
            '--controller-manager', '/controller_manager',
        ],
        output='screen',
    )

    # Step 3 — fires only after joint_state_broadcaster_spawner exits (code 0).
    # Uses the 'controller' LaunchConfiguration — defaults to fr3_arm_controller
    # but can be overridden at launch time:
    #   ros2 launch ... controller:=fr3_cartesian_impedance_controller
    # fr3_ros_controllers.yaml configures it in effort mode with PD gains.
    # Your command scripts and task nodes do not need to change.
    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            controller,                            # ← LaunchConfiguration, not hardcoded
            '--controller-manager-timeout', '30',
            '--controller-manager', '/controller_manager',
        ],
        output='screen',
    )

    # ── RViz (optional) ───────────────────────────────────────────────────────
    # Identical to simulation. Subscribes to /joint_states and /robot_description
    # — same topics, so the same saved .rviz config works unchanged.
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        parameters=[{'use_sim_time': False}],
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        robot_ip_arg,
        use_rviz_arg,
        declare_controller,
        robot_state_publisher,
        ros2_control_node,
        franka_state_broadcaster_spawner,   # Step 1 — starts immediately
        # Step 2 — fires when Step 1 exits successfully
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=franka_state_broadcaster_spawner,
                on_exit=[joint_state_broadcaster_spawner],
            )
        ),
        # Step 3 — fires when Step 2 exits successfully
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[arm_controller_spawner],
            )
        ),
        rviz,
    ])