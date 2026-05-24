**arm\_description**

FR3 Franka robot simulation · ROS2 Humble · Ignition Gazebo ·
ros2\_control

Single-command bringup that starts Gazebo, loads the FR3 robot model,
activates joint controllers, and opens RViz. Built for the 1.0 Thesis
workspace on Ubuntu 22.04.

**Prerequisites**

**System**

-   Ubuntu 22.04 (Jammy Jellyfish)

-   ROS2 Humble desktop-full

-   Ignition Fortress (installed with ros-humble-ros-gz)

**ROS2 packages**

+-------------------------------------+
| sudo apt install \\                 |
|                                     |
| ros-humble-ros-gz \\                |
|                                     |
| ros-humble-ign-ros2-control \\      |
|                                     |
| ros-humble-ros2-controllers \\      |
|                                     |
| ros-humble-robot-state-publisher \\ |
|                                     |
| ros-humble-xacro \\                 |
|                                     |
| ros-humble-rviz2                    |
+-------------------------------------+

**External dependencies**

-   franka\_description --- built and sourced in a separate workspace
    (provides meshes and YAML configs)

-   libfranka --- built from source, matching the version used on your
    host

Verify franka\_description is sourced:

  -------------------------------------------
  ros2 pkg list \| grep franka\_description
  -------------------------------------------

**Environment setup**

Add this to your \~/.bashrc so Gazebo can find the Franka mesh files.
Without it the robot spawns invisible.

+----------------------------------------------------------------------+
| echo \'export GZ\_SIM\_RESOURCE\_PATH=\$(ros2 pkg prefix             |
| franka\_description)/share\' \>\> \~/.bashrc                         |
|                                                                      |
| source \~/.bashrc                                                    |
+----------------------------------------------------------------------+

The launch file also sets this automatically via SetEnvironmentVariable,
so it works even if your shell does not have it.

**Package structure**

+------------------------------------------------------------------------+
| arm\_description/                                                      |
|                                                                        |
| config/                                                                |
|                                                                        |
| fr3\_controllers.yaml *\# PID gains and joint names for ros2\_control* |
|                                                                        |
| launch/                                                                |
|                                                                        |
| fr3\_sim.launch.py *\# single-command bringup --- edit this file*      |
|                                                                        |
| meshes/ *\# visual and collision geometry*                             |
|                                                                        |
| rviz/                                                                  |
|                                                                        |
| fr3.rviz *\# saved RViz config (generated on first run)*               |
|                                                                        |
| urdf/                                                                  |
|                                                                        |
| panda\_wrapper.urdf.xacro *\# source --- edit this, then regenerate*   |
|                                                                        |
| expanded\_fr3.urdf *\# generated --- do not edit by hand*              |
|                                                                        |
| scripts/                                                               |
|                                                                        |
| fr3\_joint\_commander.py *\# joint position publisher script*          |
|                                                                        |
| CMakeLists.txt                                                         |
|                                                                        |
| package.xml                                                            |
+------------------------------------------------------------------------+

**Build**

**1. Regenerate the URDF**

Run this whenever you change panda\_wrapper.urdf.xacro. The expanded
URDF must be regenerated so the controller config path resolves
correctly on your machine.

+----------------------------------------------------------------------+
| cd \~/ros2\_ws/src/1.0\_Thesis/arm\_description/urdf                 |
|                                                                      |
| ros2 run xacro xacro panda\_wrapper.urdf.xacro -o expanded\_fr3.urdf |
+----------------------------------------------------------------------+

**2. Build the package**

+--------------------------------------------------+
| cd \~/ros2\_ws                                   |
|                                                  |
| colcon build \--packages-select arm\_description |
|                                                  |
| source install/setup.bash                        |
+--------------------------------------------------+

**Launch**

  ------------------------------------------------------------------ ---------------------------------------------------
  **Command**                                                        **Effect**
  ros2 launch arm\_description fr3\_sim.launch.py                    Full bringup: Gazebo + robot + controllers + RViz
  ros2 launch arm\_description fr3\_sim.launch.py use\_rviz:=false   Same without RViz
  ros2 launch arm\_description fr3\_sim.launch.py \--show-args       List all available arguments
  ------------------------------------------------------------------ ---------------------------------------------------

**Startup sequence**

  ---------- ----------------------------------------------------------------------------------------------
  **Time**   **What happens**
  t = 0 s    robot\_state\_publisher and Gazebo start. RViz opens if enabled.
  t \~ 2 s   FR3 model is spawned into the Gazebo world.
  t = 5 s    joint\_state\_broadcaster activates. /joint\_states begins publishing. RViz shows the robot.
  t = 7 s    fr3\_arm\_controller activates. Ready to receive joint trajectory commands.
  ---------- ----------------------------------------------------------------------------------------------

**Verify everything is running**

+----------------------------------------------------------------------+
| ros2 control list\_controllers *\# both controllers should show      |
| \'active\'*                                                          |
|                                                                      |
| ros2 topic echo /joint\_states \--once *\# should print current      |
| joint positions*                                                     |
|                                                                      |
| ros2 topic list \| grep fr3 *\# should show                          |
| /fr3\_arm\_controller/joint\_trajectory*                             |
+----------------------------------------------------------------------+

**Sending joint commands**

**Quick test --- topic pub**

Move to the standard Franka home pose (Gazebo and RViz update
simultaneously):

+----------------------------------------------------------------------+
| ros2 topic pub \--once /fr3\_arm\_controller/joint\_trajectory \\    |
|                                                                      |
| trajectory\_msgs/msg/JointTrajectory \"{                             |
|                                                                      |
| joint\_names: \[fr3\_joint1, fr3\_joint2, fr3\_joint3, fr3\_joint4,  |
| fr3\_joint5, fr3\_joint6, fr3\_joint7\],                             |
|                                                                      |
| points: \[{                                                          |
|                                                                      |
| positions: \[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785\],          |
|                                                                      |
| time\_from\_start: {sec: 2}                                          |
|                                                                      |
| }\]                                                                  |
|                                                                      |
| }\"                                                                  |
+----------------------------------------------------------------------+

**Joint commander script**

A Python script that replaces the topic pub command with named presets,
custom durations, and optional goal confirmation:

+----------------------------------------------------------------------+
| python3 scripts/fr3\_joint\_commander.py \--preset home *\# standard |
| home pose*                                                           |
|                                                                      |
| python3 scripts/fr3\_joint\_commander.py \--preset ready *\# test    |
| pose*                                                                |
|                                                                      |
| python3 scripts/fr3\_joint\_commander.py \--positions 1.5 0 0 0 0    |
| 1.5 0 *\# explicit radians*                                          |
|                                                                      |
| python3 scripts/fr3\_joint\_commander.py \--preset home \--duration  |
| 4.0 *\# slower motion*                                               |
|                                                                      |
| python3 scripts/fr3\_joint\_commander.py \--preset home \--wait *\#  |
| block until goal reached*                                            |
+----------------------------------------------------------------------+

**RViz first-time setup**

On the first launch RViz opens with no configuration. Do this once and
save:

1.  In Global Options, set Fixed Frame to world

2.  Click Add → By display type → RobotModel → set Description Topic to
    /robot\_description

3.  Click Add → By display type → TF (to see coordinate frames)

4.  File → Save Config --- saved to rviz/fr3.rviz, loaded automatically
    on next launch

**Docker**

**Build**

Check your libfranka version first:

  ---------------------------
  dpkg -l \| grep libfranka
  ---------------------------

  ---------------------------------------------------------------------------
  docker build \--build-arg LIBFRANKA\_VERSION=0.13.3 -t fr3\_ros2:humble .
  ---------------------------------------------------------------------------

**Run**

+-----------------------------------------------+
| xhost +local:docker                           |
|                                               |
| docker run -it \--rm \\                       |
|                                               |
| \--env DISPLAY=\$DISPLAY \\                   |
|                                               |
| \--volume /tmp/.X11-unix:/tmp/.X11-unix:rw \\ |
|                                               |
| \--volume \$HOME/ros2\_ws:/root/ros2\_ws \\   |
|                                               |
| \--network host \\                            |
|                                               |
| fr3\_ros2:humble                              |
+-----------------------------------------------+

**First run inside the container**

+-------------------------------------------------+
| cd /root/ros2\_ws                               |
|                                                 |
| colcon build \--symlink-install                 |
|                                                 |
| source install/setup.bash                       |
|                                                 |
| ros2 launch arm\_description fr3\_sim.launch.py |
+-------------------------------------------------+

**Troubleshooting**

  ----------------------------------------------------- --------------------------------------------------------------------------------------------------------------------
  **Symptom**                                           **Fix**
  Robot in Entity Tree but invisible                    GZ\_SIM\_RESOURCE\_PATH not set. Run: export GZ\_SIM\_RESOURCE\_PATH=\$(ros2 pkg prefix franka\_description)/share
  Controller fails: controller\_manager not available   Increase load\_jsb period in launch file from 5.0 to 8.0
  Gazebo window does not open                           Use IncludeLaunchDescription with gz\_sim.launch.py, not ExecuteProcess with ros2 run ros\_gz\_sim ros\_gz\_sim
  Robot not visible in RViz                             Check Fixed Frame is set to \'world\'. Run: ros2 control list\_controllers --- both must show \'active\'
  colcon build fails: package not found                 Source ROS2 first: source /opt/ros/humble/setup.bash, then source franka\_description workspace
  URDF path error in Gazebo                             Regenerate the URDF from xacro: ros2 run xacro xacro panda\_wrapper.urdf.xacro -o expanded\_fr3.urdf
  ----------------------------------------------------- --------------------------------------------------------------------------------------------------------------------

**Converting this file to README.md**

Install pandoc if not already available:

  -------------------------
  sudo apt install pandoc
  -------------------------

Convert:

  ---------------------------------
  pandoc README.docx -o README.md
  ---------------------------------

After converting, review code blocks --- pandoc may render them as
indented text rather than fenced blocks. Replace indented blocks with
triple backtick fences for clean GitHub rendering.
