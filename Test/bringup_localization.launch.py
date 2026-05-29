#!/usr/bin/env python3
"""
bringup_localization.launch.py — map_server + AMCL + auto-activation, one command.

Brings up everything goal_nav.py needs for localization on a saved map:
  - clock_bridge      : mirrors the robot's timestamps onto /clock (embedded
                        below as an inline process — no separate file) so all
                        nodes share the robot's clock and AMCL stops dropping scans
  - nav2_map_server   : publishes the static map on /T7/map
  - nav2_amcl         : publishes the map->odom TF correction (localization)
  - lifecycle_manager : auto-configures + activates both (so you don't run
                        `ros2 lifecycle set ...` by hand)

This is localization only — NOT the Nav2 planner/controller/bt_navigator stack
you had comms trouble with. Your goal_nav.py node is launched separately and
just reads the map->base_link TF this provides.

RUN
    ros2 launch /path/to/bringup_localization.launch.py
    # override defaults if needed:
    ros2 launch ./bringup_localization.launch.py \
        namespace:=T7 map:=$HOME/Desktop/lab_map.yaml \
        init_x:=0.0 init_y:=0.0 init_yaw:=0.0

THEN (separate terminal) verify the TF exists, then start your node:
    ros2 run tf2_ros tf2_echo map T7/base_link
    ros2 run tb4_sensor_reader demo_test
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

# Clock bridge, embedded. Run via `python3 -c`. argv[1] = SCAN topic to mirror.
# Mirrors the laser-scan timestamps onto /clock so the sim-time nodes adopt the
# laser's clock. AMCL gates each scan on the scan->odom transform, so matching
# AMCL's clock to the scan is what stops the "scan earlier than transform cache"
# drops. Only runs when use_sim_time:=true (see IfCondition below).
CLOCK_BRIDGE_CODE = r'''
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan

SCAN_TOPIC = sys.argv[1] if len(sys.argv) > 1 else "/T7/scan"

class ClockBridge(Node):
    def __init__(self):
        super().__init__("clock_bridge")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.pub = self.create_publisher(Clock, "/clock", 10)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.cb, qos)
        self.last = -1.0
        self.get_logger().info("clock_bridge: mirroring %s stamps -> /clock" % SCAN_TOPIC)
    def cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t <= self.last:        # keep /clock monotonic
            return
        self.last = t
        out = Clock()
        out.clock = msg.header.stamp
        self.pub.publish(out)

rclpy.init()
node = ClockBridge()
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
'''

# Scan re-stamp relay, embedded. argv[1]=in scan, argv[2]=out scan, argv[3]=odom.
# Root problem: the robot's odom (Create3) and scan (RPi) are on DIFFERENT clocks
# (~1.4 s apart), so AMCL can never get the odom transform at the scan's time.
# Fix: stamp each scan with the LATEST odom timestamp -> scan and odom now share
# the odom clock, so AMCL's tf lookup always lands on a real odom transform.
# Only runs when restamp:=true.
RESTAMP_CODE = r'''
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

IN_TOPIC   = sys.argv[1] if len(sys.argv) > 1 else "/T7/scan"
OUT_TOPIC  = sys.argv[2] if len(sys.argv) > 2 else "/T7/scan_restamped"
ODOM_TOPIC = sys.argv[3] if len(sys.argv) > 3 else "/T7/odom"

class Restamp(Node):
    def __init__(self):
        super().__init__("scan_restamp")
        scan_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=10)
        odom_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=10)
        self.last_odom = None
        self.pub = self.create_publisher(LaserScan, OUT_TOPIC, scan_qos)
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_cb, odom_qos)
        self.create_subscription(LaserScan, IN_TOPIC, self.scan_cb, scan_qos)
        self.get_logger().info(
            "scan_restamp: %s stamped to latest %s time -> %s" % (IN_TOPIC, ODOM_TOPIC, OUT_TOPIC))
    def odom_cb(self, msg):
        self.last_odom = msg.header.stamp
    def scan_cb(self, msg):
        if self.last_odom is None:
            return                       # wait until we have an odom time
        msg.header.stamp = self.last_odom
        self.pub.publish(msg)

rclpy.init()
node = Restamp()
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
'''


def generate_launch_description():
    ns        = LaunchConfiguration('namespace')
    map_yaml  = LaunchConfiguration('map')
    use_sim   = LaunchConfiguration('use_sim_time')
    restamp   = LaunchConfiguration('restamp')
    init_x    = LaunchConfiguration('init_x')
    init_y    = LaunchConfiguration('init_y')
    init_yaw  = LaunchConfiguration('init_yaw')

    # Robot TF frames are namespaced (e.g. T7/odom, T7/base_link); the global
    # localization frame is the plain 'map'.
    base_frame = PythonExpression(["'", ns, "/base_link'"])
    odom_frame = PythonExpression(["'", ns, "/odom'"])

    args = [
        DeclareLaunchArgument('namespace', default_value='T7'),
        DeclareLaunchArgument(
            'map', default_value=os.path.expanduser('~/Desktop/lab_map.yaml')),
        # Clocks measured roughly synced -> real time. Only flip to true (and
        # re-enable clock_bridge in the return list) if you confirm a real skew.
        # Real time (the clock bridge alone can't fix an odom<->scan offset).
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        # Default true: re-stamp scans onto the odom clock (fixes the Create3<->RPi
        # offset). This is the working path. Set false after the robot clock synced.
        DeclareLaunchArgument('restamp', default_value='true'),
        DeclareLaunchArgument('init_x',   default_value='0.0'),
        DeclareLaunchArgument('init_y',   default_value='0.0'),
        DeclareLaunchArgument('init_yaw', default_value='0.0'),
    ]

    # Clock bridge (embedded): mirrors /<ns>/scan timestamps onto /clock. Only
    # active when use_sim_time:=true. Scan topic passed as argv (namespace-aware).
    scan_topic = ['/', ns, '/scan']
    clock_bridge = ExecuteProcess(
        cmd=['python3', '-c', CLOCK_BRIDGE_CODE, scan_topic],
        output='screen',
        condition=IfCondition(use_sim),
    )

    # Scan re-stamp relay (only when restamp:=true). When on, AMCL reads the
    # restamped topic instead of the raw scan.
    restamp_out  = ['/', ns, '/scan_restamped']
    restamp_odom = ['/', ns, '/odom']
    scan_restamp = ExecuteProcess(
        cmd=['python3', '-c', RESTAMP_CODE, scan_topic, restamp_out, restamp_odom],
        output='screen',
        condition=IfCondition(restamp),
    )
    amcl_scan = PythonExpression(
        ["'scan_restamped' if '", restamp, "' == 'true' else 'scan'"])

    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server',
        namespace=ns, output='screen',
        parameters=[{
            'use_sim_time': use_sim,
            'yaml_filename': map_yaml,
            'frame_id': 'map',
            'topic_name': 'map',
        }],
    )

    amcl = Node(
        package='nav2_amcl', executable='amcl', name='amcl',
        namespace=ns, output='screen',
        parameters=[{
            'use_sim_time': use_sim,
            'global_frame_id': 'map',
            'odom_frame_id': odom_frame,
            'base_frame_id': base_frame,
            'scan_topic': amcl_scan,         # 'scan', or 'scan_restamped' if restamp:=true
            'laser_model_type': 'likelihood_field',
            'transform_tolerance': 3.0,      # absorb WiFi latency + clock offset
            'set_initial_pose': True,        # seed pose so AMCL publishes immediately
            'initial_pose.x': init_x,
            'initial_pose.y': init_y,
            'initial_pose.z': 0.0,
            'initial_pose.yaw': init_yaw,
        }],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_localization', namespace=ns, output='screen',
        parameters=[{
            'use_sim_time': use_sim,
            'autostart': True,                       # configure + activate automatically
            'node_names': ['map_server', 'amcl'],    # in this namespace
            'bond_timeout': 0.0,                      # tolerate slow/laggy nodes
        }],
    )

    # clock_bridge / scan_restamp self-gate on their args (IfCondition); both
    # default OFF. Enable with use_sim_time:=true (bridge) or restamp:=true (relay).
    return LaunchDescription(
        args + [clock_bridge, scan_restamp, map_server, amcl, lifecycle_manager])
