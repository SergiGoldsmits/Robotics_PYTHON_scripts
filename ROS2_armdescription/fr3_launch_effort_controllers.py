import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    DeclareLaunchArgument,
    TimerAction,
)
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder

# EFFORT CONTROLLER LAUNCH — no gravity world
# ─────────────────────────────────────────────────────────────────────────────
# Uses franka_gazebo_bringup/worlds/empty_no_gravity.sdf so the PD effort
# controller does not need gravity compensation — matches real FR3 behaviour
# where Franka firmware handles gravity at the hardware level.
#
# Timeline:
# t=0s  → robot_state_publisher, clock_bridge, static_tf, spawn_robot, rviz
# t=2s  → Gazebo starts (no gravity world)
# t~3s  → robot spawned → joint_state_broadcaster → effort controller
# t=6s  → move_group starts
# ─────────────────────────────────────────────────────────────────────────────


def generate_launch_description():

    arm_share = get_package_share_directory('arm_description')

    # ── Controller selection ──────────────────────────────────────────────────
    declare_controller = DeclareLaunchArgument(
        'controller',
        default_value='cartesian_impedance_controller',
        description='Options: joint_effort_controller, '
                    'joint_position_controller, '
                    'joint_trajectory_controller'
                    'cartesian_impedance_controller',
    )
    controller = LaunchConfiguration('controller')

    # ── URDF ─────────────────────────────────────────────────────────────────
    xacro_path = os.path.join(arm_share, 'urdf', 'fr3_effort_controller.xacro')
    robot_description_content = xacro.process_file(
        xacro_path,
        mappings={'hand': 'false'}
    ).toxml()
    robot_description = {'robot_description': robot_description_content}

    with open('/tmp/fr3_resolved.urdf', 'w') as f:
        f.write(robot_description_content)
    # ── MoveIt2 config ────────────────────────────────────────────────────────
    moveit_config = (
        MoveItConfigsBuilder("fr3", package_name="fr3_moveit_config")
        .robot_description(file_path=xacro_path, mappings={"hand": "false"})
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base'],
        output='screen',
    )

    # ── Gazebo resource paths ─────────────────────────────────────────────────
    # 1. franka_description parent → resolves URDF mesh paths
    # 2. franka_gazebo_bringup/worlds → lets Gazebo find empty_no_gravity.sdf
    os.environ['GZ_SIM_RESOURCE_PATH'] = (
        os.path.dirname(get_package_share_directory('franka_description'))
        + ':' + os.path.join(
            get_package_share_directory('franka_gazebo_bringup'), 'worlds')
    )

    # ── Gazebo — no gravity world ─────────────────────────────────────────────
    gazebo = TimerAction(
        period=2.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('ros_gz_sim'),
                    'launch', 'gz_sim.launch.py',
                )
            ),
            launch_arguments={'gz_args': '-r empty_no_gravity.sdf'}.items(),
        )]
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', '/robot_description', '-name', 'fr3', '-z', '0.0'],
        output='screen',
    )

    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager-timeout', '30'],
        output='screen',
    )

    load_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[controller, '--controller-manager-timeout', '30'],
        output='screen',
    )

    rviz_config = os.path.join(arm_share, 'rviz', 'fr3.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
    )

    move_group = TimerAction(
        period=6.0,
        actions=[Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[
                moveit_config.to_dict(),
                {'use_sim_time': True},
                {'planning_plugin': 'ompl_interface/OMPLPlanner'},
                {'trajectory_execution.allowed_start_tolerance': 0.0},
            ],
        )]
    )

    gz_shutdown = RegisterEventHandler(
        OnShutdown(on_shutdown=[ExecuteProcess(
            cmd=['pkill', '-SIGINT', '-f', 'gz sim'],
            name='gz_sim_graceful_shutdown',
        )])
    )

    # ── Launch ────────────────────────────────────────────────────────────────
    return LaunchDescription([
        declare_controller,
        robot_state_publisher,
        clock_bridge,
        static_tf,
        gazebo,
        spawn_robot,
        rviz,
        move_group,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_robot,
                on_exit=[TimerAction(
                period=3.0,
                actions=[load_joint_state_broadcaster],
                    )]
                )
            ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=[load_arm_controller],
                )
            ),
        gz_shutdown,
    ])