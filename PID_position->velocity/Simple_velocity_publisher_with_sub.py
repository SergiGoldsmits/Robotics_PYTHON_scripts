#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class SafeVelocityController(Node):

    def __init__(self):
        super().__init__('safe_velocity_controller')

        # Publisher to robot
        self.pub = self.create_publisher(
            Float64MultiArray,
            '/joint_velocity_example_controller/commands',
            10
        )

        # Subscriber for external commands
        self.sub = self.create_subscription(
            Float64MultiArray,
            '/cmd_joint_velocity',
            self.cmd_callback,
            10
        )

        # current + target velocity
        self.v = 0.0
        self.v_target_cmd = 0.0

        self.v_max = 0.02
        self.ramp_rate = 0.0005

        # control loop (50 Hz)
        self.timer = self.create_timer(0.02, self.update)

        # startup hold
        self.initialized = False
        self.hold_count = 0

    def cmd_callback(self, msg: Float64MultiArray):
        """
        Expecting:
        msg.data[0] = desired base joint velocity
        """
        if len(msg.data) == 0:
            return

        self.v_target_cmd = msg.data[0]

        # clamp for safety
        if self.v_target_cmd > self.v_max:
            self.v_target_cmd = self.v_max
        elif self.v_target_cmd < -self.v_max:
            self.v_target_cmd = -self.v_max

    def update(self):
        msg = Float64MultiArray()

        # ---- startup hold ----
        if not self.initialized:
            msg.data = [0.0] * 7
            self.pub.publish(msg)

            self.hold_count += 1
            if self.hold_count > 50:
                self.initialized = True

            return

        # ---- smooth tracking of external command ----
        if self.v < self.v_target_cmd:
            self.v += self.ramp_rate
            if self.v > self.v_target_cmd:
                self.v = self.v_target_cmd
        elif self.v > self.v_target_cmd:
            self.v -= self.ramp_rate
            if self.v < self.v_target_cmd:
                self.v = self.v_target_cmd

        # publish
        msg.data = [self.v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SafeVelocityController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        stop = Float64MultiArray()
        stop.data = [0.0] * 7
        node.pub.publish(stop)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()