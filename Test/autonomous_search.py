"""Phase 2 autonomous run — map-based cube finder (localization + custom planner).

This replaces the old odometry teach-and-repeat. Instead of replaying a drifting
breadcrumb trail, the robot localizes against the Phase-1 SLAM map and plans its
own paths online:

    LOCALIZATION  slam_toolbox runs in `mode: localization`, loading the Phase-1
                  posegraph (~/Desktop/demo_map). It publishes the /map occupancy
                  grid and map→odom TF on /{NAMESPACE}/tf. This node subscribes to
                  /{NAMESPACE}/tf and /{NAMESPACE}/tf_static directly and feeds the
                  transforms into a tf2 Buffer — no launch-time /tf remap required.
    PLANNING      a custom A* planner over the occupancy grid (occupancy_planner)
                  + a waypoint-chasing follower. No Nav2 planner/controller.
    SEARCH        coverage-search: visit generated viewpoints over the mapped free
                  space while the camera scans for the red cube; stop on detection.
    OBSTACLES     the two unmapped 20 cm cylinders are detected on /scan, injected
                  into a working grid, and routed around by local replan; a front
                  safety net stops/replans if something gets too close.
    VISUALISATION a second OpenCV window ('Search Progress') shows the occupancy
                  grid with robot position, planned path, remaining coverage goals,
                  and cube location updated every control tick.

States: WAIT_FIX → SEARCHING ⇄ SCAN_SPIN → REPORTING → RETURNING → DONE.

Launch:
    ~/ros2_venv/bin/python3 -m tb4_sensor_reader.autonomous_search
"""

import math
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import OccupancyGrid
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import tf2_ros

try:                       # works as `python -m tb4_sensor_reader.autonomous_search`
    from .. import occupancy_planner as op
except ImportError:        # and as a plain script during development
    import occupancy_planner as op

# ── Robot config ────────────────────────────────────────────────────────────
NAMESPACE      = 'T7'     # ← change to your robot namespace

# ── Motion ────────────────────────────────────────────────────────────────────
FORWARD_SPEED  = 0.15      # m/s
TURN_SPEED     = 0.6       # rad/s
HEADING_TOL    = 0.15      # rad — rotate in place above this heading error
HEADING_KP     = 0.8       # proportional gain on heading error
WAYPOINT_TOL   = 0.20      # m — close enough to advance to next path waypoint
ORIGIN_THRESHOLD = 0.25    # m — close enough to count as home

# ── Planning / map ──────────────────────────────────────────────────────────
MAP_YAML_PATH    = os.path.expanduser('~/Desktop/demo_map.yaml')  # set None to use topic
INFLATION_RADIUS = 0.10    # m — robot radius + margin
COVERAGE_SPACING = 1.5     # m — spacing between coverage viewpoints
DYN_MAX_RANGE    = 2.0     # m — only project nearby scan returns as obstacles
REPLAN_PERIOD    = 2.0     # s — opportunistic replan cadence while moving

# ── Reactive safety net ───────────────────────────────────────────────────────
FRONT_ARC_DEG    = 30      # ± degrees around forward for the front-clearance check
FORWARD_OFFSET_DEG = 0.0   # bearing of "forward" in the scan frame (tune if needed)
HARD_STOP_DIST   = 0.15    # m — too close in front: stop, back off, replan
BACKOFF_TICKS    = 5       # control ticks to reverse after a hard stop

# ── Search timing ─────────────────────────────────────────────────────────────
SEARCH_TIME_LIMIT = 360.0  # s — give up searching, return home
FIX_SETTLE_TICKS  = 15     # ticks to let localization settle before moving
SCAN_SPIN_REVS    = 1.0    # full rotations to scan at each coverage goal (0 = off)

# ── Red cube HSV thresholds (matched to demo_test.py) ─────────────────────────
RED_LOW1  = np.array([0,   95,  95])
RED_HIGH1 = np.array([8,  255, 255])
RED_LOW2  = np.array([177, 95,  95])
RED_HIGH2 = np.array([180, 255, 255])
MIN_PIXELS_CENTRE = 8500              # red pixels to trigger a detection

# ── Centring / capture ──────────────────────────────────────────────────────
CENTRE_TOLERANCE_PX  = 15
CENTRE_SPIN_KP       = 0.3
CENTRE_TIMEOUT_TICKS = 100            # force capture after this many ticks centring
CAPTURE_TICKS        = 50             # hold still during capture (~2 s at 10 Hz)
SNAPSHOT_PATH        = os.path.expanduser('~/detection_snapshot.jpg')

# ── States ────────────────────────────────────────────────────────────────────
WAIT_FIX       = 'WAIT_FIX'
SEARCHING      = 'SEARCHING'
SCAN_SPIN      = 'SCAN_SPIN'
CENTRE_ON_CUBE = 'CENTRE_ON_CUBE'
REPORTING      = 'REPORTING'
RETURNING      = 'RETURNING'
DONE           = 'DONE'


class AutonomousSearch(Node):

    def __init__(self):
        super().__init__('autonomous_search')

        # Publisher / subscriptions
        self.publisher = self.create_publisher(Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.create_subscription(LaserScan, f'{NAMESPACE}/scan', self.scan_callback, 10)
        self.create_subscription(
            CompressedImage, f'{NAMESPACE}/oakd/rgb/image_raw/compressed',
            self.image_callback, 10)

        # Map / planning state
        self.gridmap        = None       # op.GridMap
        self.blocked_static = None       # inflated static obstacles (bool array)
        self.blocked_work   = None       # static + dynamic (rebuilt each tick)

        # Load the planning map from the saved .pgm/.yaml for consistency across
        # runs — AMCL's published grid can vary in size between sessions.
        if MAP_YAML_PATH and os.path.exists(MAP_YAML_PATH):
            self.gridmap = op.GridMap.from_pgm_yaml(MAP_YAML_PATH)
            occ = self.gridmap.obstacle_mask()
            self.blocked_static = op.inflate(occ, self.gridmap.resolution, INFLATION_RADIUS)
            self.get_logger().info(
                f'Map loaded from file: {self.gridmap.width}x{self.gridmap.height} '
                f'@ {self.gridmap.resolution:.3f} m/cell')
        else:
            if MAP_YAML_PATH:
                self.get_logger().warn(f'Map file not found: {MAP_YAML_PATH} — falling back to topic')
            map_qos = QoSProfile(
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                history=QoSHistoryPolicy.KEEP_LAST)
            self.create_subscription(OccupancyGrid, f'{NAMESPACE}/map',
                                     self.map_callback, map_qos)

        # TF: subscribe to both global and namespaced topics.
        # AMCL publishes map→odom on /tf (global); the robot's odom→base_link and
        # all static transforms may be on /T7/tf and /T7/tf_static.
        self.tf_buffer = tf2_ros.Buffer()
        static_qos = QoSProfile(
            depth=100,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST)
        for topic in ('/tf', f'/{NAMESPACE}/tf'):
            self.create_subscription(TFMessage, topic, self._on_tf, 100)
        for topic in ('/tf_static', f'/{NAMESPACE}/tf_static'):
            self.create_subscription(TFMessage, topic, self._on_tf_static, static_qos)
        self._tf_frames = None      # cached (map_frame, base_frame) once found

        # Tell AMCL the robot starts at (0, 0) facing +x — resent every 2 s until accepted.
        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, f'/{NAMESPACE}/initialpose', 10)


        self.coverage       = []         # queue of goal cells to visit
        self.current_goal_cell = None    # active goal cell (coverage or origin)
        self.path           = []         # list of map-frame (x, y) waypoints
        self.path_idx       = 0
        self.last_replan    = 0.0

        # Pose (map frame) — updated by _lookup_pose each tick
        self.pose = None                 # (x, y, yaw) or None

        # Scan state
        self.scan_msg      = None
        self.nearest_front = float('inf')

        # Camera / detection state
        self.bridge      = CvBridge()
        self.cube_cx     = None
        self.image_width = None
        self.latest_img  = None
        self.red_pixels  = 0

        # Per-state counters
        self.settle_ticks  = 0
        self.spin_ticks    = 0
        self.centre_ticks  = 0
        self.capture_ticks = 0
        self.backoff_ticks = 0

        # Mission tracking
        self.start_time      = time.time()
        self.returning       = False
        self.cube_world_pos  = None
        self.photo_robot_pos = None
        self.return_pos      = None
        self.mission_logged  = False

        self.state = WAIT_FIX
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            'AutonomousSearch started — waiting for map + TF pose + scan...')

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _yaw_from_quat(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _normalize(a):
        return math.atan2(math.sin(a), math.cos(a))

    def _publish(self, lin, ang):
        msg = Twist()
        msg.linear.x  = lin
        msg.angular.z = ang
        self.publisher.publish(msg)

    def stop(self):
        self.publisher.publish(Twist())

    # ── Callbacks ────────────────────────────────────────────────────────────

    def map_callback(self, msg):
        self.gridmap = op.GridMap.from_occupancy_grid(msg)
        occ = self.gridmap.obstacle_mask()
        self.blocked_static = op.inflate(occ, self.gridmap.resolution, INFLATION_RADIUS)
        self.get_logger().info(
            f'Map received {self.gridmap.width}x{self.gridmap.height} '
            f'@ {self.gridmap.resolution:.3f} m/cell', once=True)

    def scan_callback(self, msg):
        self.scan_msg = msg
        # Front clearance from true beam bearings (mount-independent).
        fwd = math.radians(FORWARD_OFFSET_DEG)
        arc = math.radians(FRONT_ARC_DEG)
        best = float('inf')
        a = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max and abs(self._normalize(a - fwd)) <= arc:
                best = min(best, r)
            a += msg.angle_increment
        self.nearest_front = best

    def image_callback(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        self.image_width = img.shape[1]
        self.latest_img  = img

        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2))
        self.red_pixels = cv2.countNonZero(mask)
        M = cv2.moments(mask)
        self.cube_cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else None

        overlay = img.copy()
        overlay[mask > 0] = [0, 0, 255]
        cv2.putText(overlay, f'Red: {self.red_pixels}  state: {self.state}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Detection', overlay)
        cv2.waitKey(1)

    # ── Localization (TF via direct subscription) ─────────────────────────────

    def _on_tf(self, msg):
        for t in msg.transforms:
            self.tf_buffer.set_transform(t, 'default')

    def _on_tf_static(self, msg):
        for t in msg.transforms:
            self.tf_buffer.set_transform_static(t, 'default')

    def _lookup_pose(self):
        """Return (x, y, yaw) in the map frame, or None if TF not yet available.

        Auto-detects whether frame ids are bare ('base_link') or namespace-prefixed
        ('T20/base_link') and caches the working pair after the first success.
        """
        if self._tf_frames:
            candidates = [self._tf_frames]
        else:
            candidates = [
                ('map', 'base_link'),
                ('map', f'{NAMESPACE}/base_link'),
                (f'{NAMESPACE}/map', f'{NAMESPACE}/base_link'),
            ]
        for mf, bf in candidates:
            try:
                t = self.tf_buffer.lookup_transform(mf, bf, rclpy.time.Time())
            except tf2_ros.TransformException:
                continue
            self._tf_frames = (mf, bf)
            tr = t.transform.translation
            return tr.x, tr.y, self._yaw_from_quat(t.transform.rotation)
        return None

    # ── Working grid (static + dynamic obstacles) ──────────────────────────────

    def _update_working_grid(self):
        """Rebuild blocked_work = inflated static map + dynamic scan obstacles.

        Only scan returns that land in *free* static space become dynamic
        obstacles, so known walls aren't double-marked and a wall hit near the
        body can't wrongly seal the corridor.
        """
        if self.blocked_static is None:
            return
        work = self.blocked_static.copy()
        gm = self.gridmap
        if self.scan_msg is not None and self.pose is not None:
            x, y, yaw = self.pose
            radius_cells = max(1, int(round(INFLATION_RADIUS / gm.resolution)))
            a = self.scan_msg.angle_min
            for r in self.scan_msg.ranges:
                if self.scan_msg.range_min < r < min(self.scan_msg.range_max, DYN_MAX_RANGE):
                    bearing = yaw + a
                    px = x + r * math.cos(bearing)
                    py = y + r * math.sin(bearing)
                    row, col = gm.world_to_cell(px, py)
                    if gm.in_bounds(row, col) and not self.blocked_static[row, col]:
                        self._mark_disk(work, row, col, radius_cells)
                a += self.scan_msg.angle_increment
        self.blocked_work = work

    @staticmethod
    def _mark_disk(grid, row, col, radius_cells):
        h, w = grid.shape
        r2 = radius_cells * radius_cells
        for dr in range(-radius_cells, radius_cells + 1):
            rr = row + dr
            if 0 <= rr < h:
                for dc in range(-radius_cells, radius_cells + 1):
                    cc = col + dc
                    if 0 <= cc < w and dr * dr + dc * dc <= r2:
                        grid[rr, cc] = True

    # ── Planning ───────────────────────────────────────────────────────────────

    def _plan_to(self, goal_cell):
        """Plan a path from the current pose to goal_cell over the working grid.

        Stores a sparse list of map-frame waypoints in self.path. Returns success.
        """
        if self.pose is None or self.blocked_work is None:
            return False
        gm = self.gridmap
        start = gm.world_to_cell(self.pose[0], self.pose[1])
        # If the start cell sits inside an inflated zone (close to a wall / robot
        # straddling the band), nudge to the nearest free cell so A* can begin.
        start = self._nearest_free(start, self.blocked_work)
        goal  = self._nearest_free(goal_cell, self.blocked_work)
        if start is None or goal is None:
            return False
        cells = op.astar(self.blocked_work, start, goal)
        if cells is None:
            # Scan noise may be sealing the corridor — retry on the static map.
            start = self._nearest_free(gm.world_to_cell(self.pose[0], self.pose[1]),
                                       self.blocked_static)
            goal  = self._nearest_free(goal_cell, self.blocked_static)
            if start is None or goal is None:
                return False
            cells = op.astar(self.blocked_static, start, goal)
        if cells is None:
            return False
        sparse = op.simplify_path(self.blocked_work, cells)
        self.path = [gm.cell_to_world(r, c) for (r, c) in sparse]
        self.path_idx = 0
        self.current_goal_cell = goal_cell
        self.last_replan = time.time()
        return True

    def _nearest_free(self, cell, blocked):
        """Return cell itself if free, else the closest free cell within a small
        radius (handles the robot starting just inside the inflation band)."""
        r, c = cell
        h, w = blocked.shape
        if 0 <= r < h and 0 <= c < w and not blocked[r, c]:
            return cell
        for rad in range(1, max(1, int(0.5 / self.gridmap.resolution)) + 1):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc]:
                        return (rr, cc)
        return None

    def _path_ahead_blocked(self):
        """True if the line to the current target waypoint crosses a blocked cell
        (e.g. a newly seen obstacle) — triggers a replan."""
        if not self.path or self.path_idx >= len(self.path):
            return False
        gm = self.gridmap
        here = gm.world_to_cell(self.pose[0], self.pose[1])
        tgt  = gm.world_to_cell(*self.path[self.path_idx])
        if not (gm.in_bounds(*here) and gm.in_bounds(*tgt)):
            return False
        return not op._line_clear(self.blocked_work, here, tgt)

    # ── Follower ───────────────────────────────────────────────────────────────

    def _follow_path(self):
        """Chase the current path waypoint-by-waypoint. Returns True when the
        final waypoint is reached."""
        x, y, yaw = self.pose
        # Advance past any waypoints we're already on top of.
        while self.path_idx < len(self.path):
            tx, ty = self.path[self.path_idx]
            if math.hypot(tx - x, ty - y) < WAYPOINT_TOL:
                self.path_idx += 1
            else:
                break
        if self.path_idx >= len(self.path):
            return True

        tx, ty = self.path[self.path_idx]
        err = self._normalize(math.atan2(ty - y, tx - x) - yaw)
        if abs(err) > HEADING_TOL:
            self._publish(0.0, max(-TURN_SPEED, min(TURN_SPEED, HEADING_KP * err)))
        else:
            self._publish(FORWARD_SPEED, max(-TURN_SPEED, min(TURN_SPEED, HEADING_KP * err)))
        return False

    def _safety_triggered(self):
        """Front too close: stop, reverse briefly, force a replan."""
        if self.nearest_front < HARD_STOP_DIST:
            self.backoff_ticks = BACKOFF_TICKS
        if self.backoff_ticks > 0:
            self.backoff_ticks -= 1
            self._publish(-0.10, 0.0)
            self.path = []           # invalidate path → replan on resume
            return True
        return False

    # ── Map visualisation ─────────────────────────────────────────────────────

    def _draw_map_viz(self):
        """Render a top-down view of search progress in an OpenCV window.

        Legend:
          dark grey  — inflated obstacle
          light grey — free space
          cyan dots  — remaining coverage goals
          cyan ring  — current goal
          orange     — planned path ahead
          green      — robot (circle + heading arrow)
          red        — detected cube
        """
        if self.gridmap is None or self.blocked_static is None:
            return
        gm = self.gridmap
        h, w = gm.height, gm.width

        canvas = np.full((h, w, 3), 200, dtype=np.uint8)
        canvas[self.blocked_static] = [60, 60, 60]

        # Remaining coverage goals
        for cell in self.coverage:
            r, c = cell
            if 0 <= r < h and 0 <= c < w:
                cv2.circle(canvas, (c, r), 3, (0, 220, 220), -1)

        # Active goal
        if self.current_goal_cell is not None:
            gr, gc = self.current_goal_cell
            if 0 <= gr < h and 0 <= gc < w:
                cv2.circle(canvas, (gc, gr), 6, (0, 220, 220), 2)

        # Planned path (robot → waypoints)
        if self.path and self.path_idx < len(self.path) and self.pose:
            pr0, pc0 = gm.world_to_cell(self.pose[0], self.pose[1])
            prev = (pc0, pr0)
            for wx, wy in self.path[self.path_idx:]:
                pr, pc = gm.world_to_cell(wx, wy)
                cur = (pc, pr)
                cv2.line(canvas, prev, cur, (0, 120, 255), 1)
                prev = cur

        # Robot position + heading arrow
        if self.pose:
            x, y, yaw = self.pose
            pr, pc = gm.world_to_cell(x, y)
            if 0 <= pr < h and 0 <= pc < w:
                cv2.circle(canvas, (pc, pr), 5, (0, 200, 0), -1)
                alen = max(6, int(0.25 / gm.resolution))
                tip = (pc + int(alen * math.cos(yaw)),
                       pr + int(alen * math.sin(yaw)))
                cv2.arrowedLine(canvas, (pc, pr), tip, (0, 255, 0), 2, tipLength=0.4)

        # Detected cube
        if self.cube_world_pos:
            pr, pc = gm.world_to_cell(*self.cube_world_pos)
            if 0 <= pr < h and 0 <= pc < w:
                cv2.circle(canvas, (pc, pr), 8, (0, 0, 255), -1)
                cv2.putText(canvas, 'CUBE', (pc + 6, pr - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

        # HUD: state, goals remaining, elapsed time
        elapsed = int(time.time() - self.start_time)
        hud = (f'{self.state}   goals={len(self.coverage)}   t={elapsed}s'
               + (f'   pos=({self.pose[0]:.2f},{self.pose[1]:.2f})' if self.pose else ''))
        cv2.putText(canvas, hud, (4, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Scale up small maps so the window is readable
        scale = max(1, min(6, 600 // max(h, w, 1)))
        if scale > 1:
            canvas = cv2.resize(canvas, (w * scale, h * scale),
                                interpolation=cv2.INTER_NEAREST)

        # Flip vertically: ROS row-0 = lowest Y; standard map view has Y up
        canvas = cv2.flip(canvas, 0)
        cv2.imshow('Search Progress', canvas)
        cv2.waitKey(1)

    # ── Control loop ───────────────────────────────────────────────────────────

    def control_loop(self):
        self.pose = self._lookup_pose() or self.pose

        # Global detection interrupt during the search legs.
        if (not self.returning and self.state in (SEARCHING, SCAN_SPIN)
                and self.red_pixels >= MIN_PIXELS_CENTRE):
            self.centre_ticks = 0
            self.state = CENTRE_ON_CUBE
            self.get_logger().info(
                f'Red cube detected ({self.red_pixels} px) → CENTRE_ON_CUBE')

        if self.state == WAIT_FIX:
            self.do_wait_fix()
        elif self.state == SEARCHING:
            self.do_searching()
        elif self.state == SCAN_SPIN:
            self.do_scan_spin()
        elif self.state == CENTRE_ON_CUBE:
            self.do_centre_on_cube()
        elif self.state == REPORTING:
            self.do_reporting()
        elif self.state == RETURNING:
            self.do_returning()
        elif self.state == DONE:
            self.stop()
            if not self.mission_logged:
                self.mission_logged = True
                self.log_mission_summary()

        self._draw_map_viz()

    # ── State behaviours ─────────────────────────────────────────────────────

    def _send_initial_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.orientation.w = 1.0   # yaw = 0 → facing +x
        msg.pose.covariance[0]  = 0.25      # x variance
        msg.pose.covariance[7]  = 0.25      # y variance
        msg.pose.covariance[35] = 0.1       # yaw variance
        self._initial_pose_pub.publish(msg)
        self.get_logger().info('Sent initial pose (0, 0, 0) to AMCL')

    def do_wait_fix(self):
        self.stop()
        # Resend every 2 s until AMCL accepts it (it may not be active yet on the first tick).
        if self.pose is None and time.time() - getattr(self, '_last_pose_send', 0) > 2.0:
            self._send_initial_pose()
            self._last_pose_send = time.time()
        ready = (self.blocked_static is not None and self.pose is not None
                 and self.scan_msg is not None)
        if not ready:
            self.get_logger().warn(
                'WAIT_FIX: waiting for '
                f"{'map ' if self.blocked_static is None else ''}"
                f"{'TF-pose ' if self.pose is None else ''}"
                f"{'scan' if self.scan_msg is None else ''}",
                throttle_duration_sec=2.0)
            return
        self.settle_ticks += 1
        if self.settle_ticks < FIX_SETTLE_TICKS:
            return

        # Generate the coverage plan from the current (start) pose.
        self._update_working_grid()
        start_cell = self.gridmap.world_to_cell(self.pose[0], self.pose[1])
        start_cell = self._nearest_free(start_cell, self.blocked_static)
        if start_cell is None:
            self.get_logger().warn(
                f'Start pose ({self.pose[0]:.2f}, {self.pose[1]:.2f}) is inside '
                'inflated obstacles — reduce INFLATION_RADIUS or reposition robot',
                throttle_duration_sec=2.0)
            return
        reach = op.flood_reachable(self.blocked_static, start_cell)
        self.coverage = op.coverage_goals(
            self.blocked_static, reach, self.gridmap.resolution,
            COVERAGE_SPACING, start_cell)
        self.get_logger().info(
            f'Localized at ({self.pose[0]:.2f}, {self.pose[1]:.2f}); '
            f'{len(self.coverage)} coverage goals → SEARCHING')
        self.state = SEARCHING

    def do_searching(self):
        if time.time() - self.start_time > SEARCH_TIME_LIMIT:
            self.get_logger().warn('Search time limit reached — returning home')
            self.begin_return()
            return

        self._update_working_grid()
        if self._safety_triggered():
            return

        # Need a path? Either none yet, or the one ahead just got blocked, or it's
        # time for an opportunistic replan.
        need_plan = (not self.path
                     or self._path_ahead_blocked()
                     or time.time() - self.last_replan > REPLAN_PERIOD)
        if need_plan and self.current_goal_cell is not None:
            if not self._plan_to(self.current_goal_cell) and not self.path:
                # Current goal unreachable (e.g. obstacle sealed it) — drop it.
                self.current_goal_cell = None

        if self.current_goal_cell is None:
            if not self.coverage:
                self.get_logger().info('Coverage exhausted — returning home')
                self.begin_return()
                return
            while self.coverage:
                goal = self.coverage.pop(0)
                if self._plan_to(goal):
                    self.get_logger().info(
                        f'New coverage goal {goal}; {len(self.coverage)} left')
                    break
            else:
                self.get_logger().warn('All remaining goals unreachable — returning home')
                self.begin_return()
                return

        if self._follow_path():
            # Reached the coverage goal — scan in place, then move on.
            self.current_goal_cell = None
            self.path = []
            if SCAN_SPIN_REVS > 0:
                self.spin_ticks = int(SCAN_SPIN_REVS * 2 * math.pi
                                      / (TURN_SPEED * 0.1))
                self.state = SCAN_SPIN

    def do_scan_spin(self):
        """Rotate in place at a coverage goal so the camera sweeps the area."""
        self.spin_ticks -= 1
        if self.spin_ticks <= 0:
            self.state = SEARCHING
            return
        self._publish(0.0, TURN_SPEED)

    def do_centre_on_cube(self):
        """Spin to put the red cube in the image centre before reporting."""
        if self.cube_cx is None or self.image_width is None:
            self.stop()      # lost the centroid mid-spin — hold, don't drift
            return
        error = self.cube_cx - self.image_width / 2     # positive = cube is right
        self.centre_ticks += 1
        if abs(error) <= CENTRE_TOLERANCE_PX or self.centre_ticks >= CENTRE_TIMEOUT_TICKS:
            self.capture_ticks = 0
            self.state = REPORTING
            reason = 'timeout' if self.centre_ticks >= CENTRE_TIMEOUT_TICKS else f'{error:.0f}px'
            self.get_logger().info(f'Cube centred ({reason}) → REPORTING')
            return
        self._publish(0.0, -CENTRE_SPIN_KP * (error / (self.image_width / 2)))

    def do_reporting(self):
        """Stop, log the map-frame (x, y), save the snapshot, then return."""
        self.stop()
        self.capture_ticks += 1
        if self.capture_ticks == CAPTURE_TICKS // 2 and self.pose is not None:
            x, y, yaw = self.pose
            self.photo_robot_pos = (x, y)
            self.get_logger().info(f'[REPORT] Cube reported at map ({x:.3f}, {y:.3f}) m')
            if self.latest_img is not None:
                cv2.imwrite(SNAPSHOT_PATH, self.latest_img)
                self.get_logger().info(f'Snapshot saved → {SNAPSHOT_PATH}')
            else:
                self.get_logger().warn('No camera image — snapshot NOT saved')
            if self.nearest_front != float('inf'):
                self.cube_world_pos = (x + self.nearest_front * math.cos(yaw),
                                       y + self.nearest_front * math.sin(yaw))
        if self.capture_ticks >= CAPTURE_TICKS:
            self.begin_return()

    def begin_return(self):
        self.returning = True
        self.path = []
        self.current_goal_cell = None
        self.state = RETURNING
        self.get_logger().info('RETURNING — planning path to origin (0, 0)')

    def do_returning(self):
        if self.pose is None:
            return
        if math.hypot(self.pose[0], self.pose[1]) < ORIGIN_THRESHOLD:
            self.return_pos = (self.pose[0], self.pose[1])
            self.stop()
            self.state = DONE
            self.get_logger().info('HOME REACHED — mission complete')
            return

        self._update_working_grid()
        if self._safety_triggered():
            return

        if self.current_goal_cell is None:
            self.current_goal_cell = self.gridmap.world_to_cell(0.0, 0.0)
        need_plan = (not self.path or self._path_ahead_blocked()
                     or time.time() - self.last_replan > REPLAN_PERIOD)
        if need_plan:
            self._plan_to(self.current_goal_cell)
        if self.path:
            self._follow_path()

    def log_mission_summary(self):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(elapsed, 60)
        self.get_logger().info('=' * 55)
        self.get_logger().info('[MISSION SUMMARY]')
        self.get_logger().info(f'  Duration       : {int(mins)}m {secs:.1f}s')
        if self.photo_robot_pos:
            self.get_logger().info(
                f'  Robot at report: ({self.photo_robot_pos[0]:.3f}, {self.photo_robot_pos[1]:.3f}) m')
        else:
            self.get_logger().info('  Robot at report: unknown (no detection)')
        if self.cube_world_pos:
            self.get_logger().info(
                f'  Cube world pos : ({self.cube_world_pos[0]:.3f}, {self.cube_world_pos[1]:.3f}) m')
        else:
            self.get_logger().info('  Cube world pos : unknown')
        if self.return_pos:
            self.get_logger().info(
                f'  Return pos     : ({self.return_pos[0]:.3f}, {self.return_pos[1]:.3f}) m')
        else:
            self.get_logger().info('  Return pos     : did not reach origin')
        self.get_logger().info('=' * 55)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousSearch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
