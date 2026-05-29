#!/usr/bin/env python3
"""
clock_bridge.py — Publish /clock from the robot's message timestamps.

Workaround for a workstation<->robot clock skew you can't fix on the robot
(no ssh). It mirrors the robot's clock onto the ROS sim-time topic /clock by
copying header.stamp off a steady robot-published topic (/T8/odom).

Run this, then start every OTHER workstation node with use_sim_time:=true
(map_server, amcl, goal_nav). They will all adopt the robot's clock, so scans
and the map->odom TF share one time base and AMCL stops dropping scans.

This node itself must run on REAL time (use_sim_time stays false) — it is the
clock source.

RUN
    python3 clock_bridge.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry

ODOM_TOPIC = '/T8/odom'   # any steady robot-stamped topic; odom is high-rate


class ClockBridge(Node):
    def __init__(self):
        super().__init__('clock_bridge')
        # Best-effort sensor QoS matches typical robot publishers.
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.pub = self.create_publisher(Clock, '/clock', 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self.cb, qos)
        self.last = -1.0
        self.get_logger().info(f'clock_bridge: mirroring {ODOM_TOPIC} stamps -> /clock')

    def cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t <= self.last:        # keep /clock monotonic
            return
        self.last = t
        out = Clock()
        out.clock = msg.header.stamp
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ClockBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
