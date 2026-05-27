#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class SafeVelocityController(Node):

    def __init__(self):
        super().__init__('safe_velocity_controller')

        self.pub = self.create_publisher(
            Float64MultiArray,
            '/joint_velocity_example_controller/commands',
            10
        )
        self.v = 0.0
        self.v_target = 0.02     # max velocity (rad/s)
        self.ramp_rate = 0.0005  # acceleration step per cycle

        # 50 Hz _ to create the control loop
        self.timer = self.create_timer(0.02, self.update)

        # zero start
        self.initialized = False
        self.hold_count = 0

    def update(self):
        msg = Float64MultiArray()

        # ---- Step 1: hold zero briefly on startup ----
        if not self.initialized:
            msg.data = [0.0] * 7
            self.pub.publish(msg)

            self.hold_count += 1
            if self.hold_count > 50:  # ~1 second
                self.initialized = True

            return

        # ---- Step 2: smooth ramp-up ----
        self.v += self.ramp_rate
        if self.v > self.v_target:
            self.v = self.v_target

        # apply to base joint
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
        # safe stop
        stop = Float64MultiArray()
        stop.data = [0.0] * 7
        node.pub.publish(stop)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()