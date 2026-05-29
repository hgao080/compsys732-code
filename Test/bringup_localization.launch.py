#!/usr/bin/env python3
"""
bringup_localization.launch.py — map_server + AMCL + auto-activation, one command.

Brings up everything goal_nav.py needs for localization on a saved map:
  - clock_bridge      : mirrors the robot's timestamps onto /clock (embedded
                        below as an inline process — no separate file) so all
                        nodes share the robot's clock and AMCL stops dropping scans
  - nav2_map_server   : publishes the static map on /T8/map
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
        namespace:=T8 map:=$HOME/Desktop/lab_map.yaml \
        init_x:=0.0 init_y:=0.0 init_yaw:=0.0

THEN (separate terminal) verify the TF exists, then start your node:
    ros2 run tf2_ros tf2_echo map T8/base_link
    ros2 run tb4_sensor_reader demo_test
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

# Clock bridge, embedded. Run via `python3 -c`. argv[1] = odom topic to mirror.
# Copies the robot's message timestamps onto /clock so the sim-time nodes adopt
# the robot's clock (fixes the workstation<->robot skew that drops AMCL scans).
CLOCK_BRIDGE_CODE = r'''
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry

ODOM_TOPIC = sys.argv[1] if len(sys.argv) > 1 else "/T8/odom"

class ClockBridge(Node):
    def __init__(self):
        super().__init__("clock_bridge")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.pub = self.create_publisher(Clock, "/clock", 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self.cb, qos)
        self.last = -1.0
        self.get_logger().info("clock_bridge: mirroring %s stamps -> /clock" % ODOM_TOPIC)
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


def generate_launch_description():
    ns        = LaunchConfiguration('namespace')
    map_yaml  = LaunchConfiguration('map')
    use_sim   = LaunchConfiguration('use_sim_time')
    init_x    = LaunchConfiguration('init_x')
    init_y    = LaunchConfiguration('init_y')
    init_yaw  = LaunchConfiguration('init_yaw')

    # Robot TF frames are namespaced (e.g. T8/odom, T8/base_link); the global
    # localization frame is the plain 'map'.
    base_frame = PythonExpression(["'", ns, "/base_link'"])
    odom_frame = PythonExpression(["'", ns, "/odom'"])

    args = [
        DeclareLaunchArgument('namespace', default_value='T8'),
        DeclareLaunchArgument(
            'map', default_value=os.path.expanduser('~/Desktop/lab_map.yaml')),
        # Clocks measured roughly synced -> real time. Only flip to true (and
        # re-enable clock_bridge in the return list) if you confirm a real skew.
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('init_x',   default_value='0.0'),
        DeclareLaunchArgument('init_y',   default_value='0.0'),
        DeclareLaunchArgument('init_yaw', default_value='0.0'),
    ]

    # Clock bridge (embedded): runs the inline code above on real time, mirroring
    # /<ns>/odom timestamps onto /clock. odom topic passed as argv (namespace-aware).
    odom_topic = ['/', ns, '/odom']
    clock_bridge = ExecuteProcess(
        cmd=['python3', '-c', CLOCK_BRIDGE_CODE, odom_topic],
        output='screen',
    )

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
            'scan_topic': 'scan',            # relative -> /T8/scan
            'laser_model_type': 'likelihood_field',
            'transform_tolerance': 1.0,      # absorb WiFi latency
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

    # clock_bridge disabled — clocks measured ~synced. Re-add it to this list
    # (and set use_sim_time:=true) only if a real skew is confirmed.
    return LaunchDescription(args + [map_server, amcl, lifecycle_manager])
