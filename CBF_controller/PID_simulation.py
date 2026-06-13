#!/usr/bin/env python3
"""
fr3_simulator.py — FR3 kinematic simulator for CBF testing

Simulates the FR3 robot by integrating velocity commands to update
joint positions. Publishes joint states so the PID+CBF controller
has feedback. Visualises in RViz with obstacle marker.

No Gazebo required — pure kinematic integration.

Subscribes:
  /joint_velocity_example_controller/commands  (std_msgs/Float64MultiArray)

Publishes:
  /franka/joint_states   (sensor_msgs/JointState)    — 50 Hz
  /joint_states          (sensor_msgs/JointState)    — 50 Hz (for RViz)
  /robot_description     (std_msgs/String)            — latched
  /obstacle_marker       (visualization_msgs/Marker)  — 10 Hz

Usage:
  Terminal 1: python3 fr3_simulator.py
  Terminal 2: python3 cartesian_pid_controller.py
  Terminal 3: rviz2  (add RobotModel + Marker displays)
  Terminal 4: ros2 topic pub /cartesian_pid/goal_pose ...
  Terminal 5: ros2 topic echo /cbf_status
"""

import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.qos import QoSProfile, DurabilityPolicy
import os


# FR3 ready pose — safe starting configuration
FR3_READY_POSE = np.array([0.0, -0.908, 0.0, -2.601, 0.0, 1.6, 0.785])

# FR3 joint limits [rad]
JOINT_LIMITS_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718,
                               -2.8973, -0.0175, -2.8973])
JOINT_LIMITS_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,
                                2.8973,  3.7525,  2.8973])


class FR3Simulator(Node):

    def __init__(self):
        super().__init__('fr3_simulator')

        # ── Obstacle parameters — must match cartesian_pid_controller.py ──────
        self.obs_center = [0.45, 0.25]
        self.obs_radius = 0.06

        # ── Robot state ───────────────────────────────────────────────────────
        self.q    = FR3_READY_POSE.copy()
        self.dq   = np.zeros(7)

        # ── Timing ────────────────────────────────────────────────────────────
        self.dt_sim    = 0.02    # 50 Hz simulation
        self.dt_marker = 0.1     # 10 Hz marker

        # ── Subscriber — velocity commands ───────────────────────────────────
        self.create_subscription(
            Float64MultiArray,
            '/joint_velocity_example_controller/commands',
            self.cmd_callback, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.js_pub     = self.create_publisher(
            JointState, '/franka/joint_states', 10)
        self.js_pub2    = self.create_publisher(
            JointState, '/joint_states', 10)
        self.marker_pub = self.create_publisher(
            Marker, '/obstacle_marker', 10)

        # robot_description with TRANSIENT_LOCAL for RViz
        qos_latched = QoSProfile(depth=1,
                                  durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.desc_pub = self.create_publisher(
            String, '/robot_description', qos_latched)
        # publish URDF
        from std_msgs.msg import String as StringMsg
        desc_msg = StringMsg()
        desc_msg.data = open('/tmp/fr3_resolved.urdf').read()
        self.create_timer(0.5, lambda: self.desc_pub.publish(desc_msg))

        # robot_description with TRANSIENT_LOCAL for RViz
        qos_latched = QoSProfile(depth=1,
                                  durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.desc_pub = self.create_publisher(
            String, '/robot_description', qos_latched)
        # publish URDF
        from std_msgs.msg import String as StringMsg
        desc_msg = StringMsg()
        desc_msg.data = open('/tmp/fr3_resolved.urdf').read()
        self.create_timer(0.5, lambda: self.desc_pub.publish(desc_msg))

        # TF broadcaster for RViz
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Timers ────────────────────────────────────────────────────────────
        self.create_timer(self.dt_sim,    self.simulation_step)
        self.create_timer(self.dt_marker, self.publish_obstacle_marker)

        self.get_logger().info('FR3 Simulator started')
        self.get_logger().info(
            f'Starting pose: ready pose  '
            f'{np.round(np.degrees(self.q), 1)} deg')
        self.get_logger().info(
            f'Obstacle: centre={self.obs_center}  radius={self.obs_radius} m')
        self.get_logger().info(
            'Open RViz, add RobotModel (topic /robot_description) '
            'and Marker (topic /obstacle_marker)')

    # ── Velocity command callback ─────────────────────────────────────────────
    def cmd_callback(self, msg):
        if len(msg.data) >= 7:
            self.dq = np.array(msg.data[:7])
        else:
            self.dq = np.zeros(7)

    # ── Simulation step — 50 Hz ───────────────────────────────────────────────
    def simulation_step(self):
        # integrate: q = q + dq * dt
        self.q = self.q + self.dq * self.dt_sim

        # enforce joint limits
        self.q = np.clip(self.q, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX)

        # publish joint states
        now = self.get_clock().now().to_msg()
        js  = JointState()
        js.header.stamp = now
        js.name         = [f'fr3_joint{i}' for i in range(1, 8)]
        js.position     = self.q.tolist()
        js.velocity     = self.dq.tolist()
        js.effort       = [0.0] * 7

        self.js_pub.publish(js)
        self.js_pub2.publish(js)

        # broadcast TF for RViz
        self._broadcast_base_tf(now)

    # ── TF base frame ─────────────────────────────────────────────────────────
    def _broadcast_base_tf(self, stamp):
        t                        = TransformStamped()
        t.header.stamp           = stamp
        t.header.frame_id        = 'world'
        t.child_frame_id         = 'fr3_link0'
        t.transform.rotation.w   = 1.0
        self.tf_broadcaster.sendTransform(t)

    # ── Obstacle marker for RViz ──────────────────────────────────────────────
    def publish_obstacle_marker(self):
        m                    = Marker()
        m.header.frame_id    = 'fr3_link0'
        m.header.stamp       = self.get_clock().now().to_msg()
        m.ns                 = 'obstacle'
        m.id                 = 0
        m.type               = Marker.SPHERE
        m.action             = Marker.ADD

        # obstacle in XZ plane — y=0
        m.pose.position.x    = float(self.obs_center[0])
        m.pose.position.y    = 0.0
        m.pose.position.z    = float(self.obs_center[1])
        m.pose.orientation.w = 1.0

        # diameter = 2 * radius
        m.scale.x = self.obs_radius * 2
        m.scale.y = self.obs_radius * 2
        m.scale.z = self.obs_radius * 2

        # red semi-transparent
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 0.5

        self.marker_pub.publish(m)

        # also publish safety boundary marker (slightly larger, wireframe)
        m2                    = Marker()
        m2.header             = m.header
        m2.ns                 = 'obstacle_boundary'
        m2.id                 = 1
        m2.type               = Marker.SPHERE
        m2.action             = Marker.ADD
        m2.pose               = m.pose
        m2.scale.x            = (self.obs_radius + 0.05) * 2  # + link radius
        m2.scale.y            = (self.obs_radius + 0.05) * 2
        m2.scale.z            = (self.obs_radius + 0.05) * 2
        m2.color.r            = 1.0
        m2.color.g            = 0.5
        m2.color.b            = 0.0
        m2.color.a            = 0.15   # very transparent boundary
        self.marker_pub.publish(m2)


def main():
    rclpy.init()
    node = FR3Simulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
