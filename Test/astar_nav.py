#!/usr/bin/env python3
"""
astar_nav.py — Map-based autonomous navigation for TurtleBot4 (ROS2), no Nav2.

WHAT IT DOES
  1. Loads a pre-built occupancy map (.pgm + .yaml saved during your mapping run).
  2. Localises on that map (AMCL via TF map->base_link, or raw odometry fallback).
  3. Plans a path to a manually-specified goal (GOAL_X, GOAL_Y) with a custom
     A* planner running on the static map.
  4. While driving, projects live LiDAR returns into the map as a "dynamic
     obstacle layer", so obstacles that were NOT present during mapping get
     avoided. The path is re-planned whenever it becomes blocked.
  5. Follows the path with a pure-pursuit controller, with an emergency stop
     if something is right in front.

WHY NOT NAV2
  Nav2's planner/controller/bt_navigator action servers are avoided (you report
  unreliable comms). Everything here is plain pub/sub + TF + a self-contained
  A* — no action servers, no lifecycle manager for the planning side.

LOCALISATION NOTE
  POSE_SOURCE = 'amcl' reads the AMCL-corrected pose from the map->base_link TF.
  AMCL itself still needs to be running (it publishes that TF). If you have not
  got AMCL working yet, set POSE_SOURCE = 'odom' to drive using wheel odometry
  only — good for validating the planner / obstacle avoidance first. Odom drifts,
  so the goal will be less accurate over a long traverse; switch to 'amcl' for
  the real run.

  Minimal standalone AMCL bring-up (separate terminals, adjust namespace):
    ros2 run nav2_map_server map_server --ros-args \
        -p yaml_filename:=<your_map>.yaml -p frame_id:=map
    ros2 run nav2_amcl amcl --ros-args -r __ns:=/T24 \
        -p global_frame_id:=map -p odom_frame_id:=T24/odom \
        -p base_frame_id:=T24/base_link -p scan_topic:=/T24/scan
    # then activate both lifecycle nodes:
    ros2 lifecycle set /map_server configure && ros2 lifecycle set /map_server activate
    ros2 lifecycle set /T24/amcl configure && ros2 lifecycle set /T24/amcl activate
  This node also publishes an initial pose at (START_X, START_Y, START_YAW) so
  you don't have to set the "2D Pose Estimate" by hand in RViz.

RUN
    python3 astar_nav.py
"""

import os, math, time, heapq, yaml
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# ── Robot / Topic Config ──────────────────────────────────────────────────────
NAMESPACE   = 'T24'                       # ← your robot namespace ('' for none)
CMD_VEL     = f'{NAMESPACE}/cmd_vel'
SCAN_TOPIC  = f'{NAMESPACE}/scan'
ODOM_TOPIC  = f'{NAMESPACE}/odom'
INITIALPOSE = f'{NAMESPACE}/initialpose'
MAP_FRAME   = 'map'
BASE_FRAME  = f'{NAMESPACE}/base_link'

# ── Map Config ────────────────────────────────────────────────────────────────
MAP_YAML_PATH = os.path.expanduser('~/maps/course.yaml')  # ← saved map .yaml
ALLOW_UNKNOWN = False     # treat unknown (-1) map cells as obstacles (safer = False)

# ── Pose Source ───────────────────────────────────────────────────────────────
POSE_SOURCE = 'amcl'      # 'amcl' (map->base_link TF) or 'odom' (wheel odom only)
SEED_INITIAL_POSE = True  # publish AMCL initialpose on startup (only used for 'amcl')
START_X, START_Y, START_YAW = 0.0, 0.0, 0.0   # robot's true start on the map
INITIAL_POSE_DELAY_S = 2.0

# ── Goal ──────────────────────────────────────────────────────────────────────
GOAL_X, GOAL_Y  = 3.0, 0.0    # ← read these off your map (metres, map frame)
GOAL_TOLERANCE  = 0.18        # metres — close enough to count as arrived

# ── Footprint / Planning Config ───────────────────────────────────────────────
ROBOT_RADIUS     = 0.18       # TurtleBot4 ~0.17 m radius
INFLATION_RADIUS = 0.25       # metres — obstacles grown by this for planning
PLAN_RESOLUTION  = 0.05       # metres/cell for the planner (map is coarsened to this)
REPLAN_PERIOD_S  = 1.0        # seconds — periodic replan cadence
MAX_RECOVERIES   = 6          # consecutive plan failures before giving up

# ── LiDAR / Dynamic Obstacle Config ───────────────────────────────────────────
# Beam angle in the BASE frame = (angle_min + i*angle_increment) + LIDAR_YAW_OFFSET.
# The ACF_demo scripts treat forward as +90deg in the raw scan, i.e. the laser
# frame is rotated -90deg vs base — hence the default below. If projected
# obstacles look rotated in the debug view, adjust this first.
LIDAR_YAW_OFFSET   = math.radians(-90.0)
OBSTACLE_MAX_RANGE = 3.0      # metres — ignore returns beyond this for mapping
OBSTACLE_DECAY     = 0.80     # per-scan decay of the dynamic layer (clears moved obstacles)
OBSTACLE_HIT       = 1.0      # value written for a fresh hit
OBSTACLE_THRESH    = 0.40     # dynamic-layer value above which a cell counts as blocked

# ── Control Config ────────────────────────────────────────────────────────────
FORWARD_SPEED  = 0.15         # m/s
MAX_TURN       = 1.0          # rad/s cap
HEADING_KP     = 1.6          # proportional gain on heading error
LOOKAHEAD      = 0.40         # metres — pure-pursuit lookahead
TURN_IN_PLACE  = math.radians(45)  # |heading err| above this -> rotate, don't drive
SAFE_STOP_DIST = 0.30         # metres — emergency stop if forward arc closer than this
FRONT_ARC_DEG  = 30           # degrees either side of forward for the emergency check

# ── Visualisation ─────────────────────────────────────────────────────────────
SHOW_VIS = True               # cv2 debug window (needs a display); set False if headless
VIS_SCALE = 4                 # pixels per planning cell in the debug window

# ── States ────────────────────────────────────────────────────────────────────
WAIT_FOR_POSE = 'WAIT_FOR_POSE'
NAVIGATE      = 'NAVIGATE'
RECOVERY      = 'RECOVERY'
GOAL_REACHED  = 'GOAL_REACHED'
GIVE_UP       = 'GIVE_UP'


# ── Map Loading ─────────────────────────────────────────────────────────────--
def load_map(yaml_path):
    """Load a ROS map_server .yaml/.pgm pair.

    Returns occupied/unknown boolean grids stored Y-UP (row 0 = lowest y, so
    row increases with world +y), resolution (m/cell) and origin (x, y) of the
    lower-left cell. Drops the origin yaw (assumes 0, the usual case).
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    img_rel = cfg['image']
    img_path = img_rel if os.path.isabs(img_rel) \
        else os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_rel)

    res      = float(cfg['resolution'])
    origin   = cfg['origin']                       # [x, y, yaw]
    negate   = int(cfg.get('negate', 0))
    occ_th   = float(cfg.get('occupied_thresh', 0.65))
    free_th  = float(cfg.get('free_thresh', 0.196))

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Could not read map image: {img_path}')

    p = img.astype(np.float64)
    occ = p / 255.0 if negate else (255.0 - p) / 255.0   # 1 = occupied, 0 = free
    occupied = occ > occ_th
    free     = occ < free_th
    unknown  = ~occupied & ~free

    # Image row 0 is the TOP (max y); flip so row 0 becomes the BOTTOM (min y).
    occupied = np.flipud(occupied)
    unknown  = np.flipud(unknown)
    return occupied, unknown, res, (float(origin[0]), float(origin[1]))


def coarsen(mask, f):
    """Max-pool a boolean grid by integer factor f (occupied if any sub-cell is).

    Pads at the high-index end (high x / high y) so the lower-left origin stays
    anchored.
    """
    if f <= 1:
        return mask
    h, w = mask.shape
    hp = ((h + f - 1) // f) * f
    wp = ((w + f - 1) // f) * f
    padded = np.zeros((hp, wp), dtype=bool)
    padded[:h, :w] = mask
    return padded.reshape(hp // f, f, wp // f, f).max(axis=(1, 3))


class AStarNav(Node):

    def __init__(self):
        super().__init__('astar_nav')

        # ── Load + prepare the map ──
        occ, unk, res0, origin = load_map(MAP_YAML_PATH)
        factor = max(1, int(round(PLAN_RESOLUTION / res0)))
        self.occupied = coarsen(occ, factor)
        self.unknown  = coarsen(unk, factor)
        self.pres     = res0 * factor                 # planning resolution (m/cell)
        self.ox, self.oy = origin
        self.H, self.W = self.occupied.shape          # rows (y), cols (x)

        # Inflation kernel (circular) sized to robot radius + margin.
        infl_cells = max(1, int(math.ceil(INFLATION_RADIUS / self.pres)))
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * infl_cells + 1, 2 * infl_cells + 1))

        # Dynamic obstacle layer (live LiDAR), same shape, Y-UP.
        self.obstacle_grid = np.zeros((self.H, self.W), dtype=np.float32)
        self.blocked = self._build_blocked()         # latest inflated cost grid

        # ── Pose state ──
        self.current_x = START_X
        self.current_y = START_Y
        self.current_yaw = START_YAW
        self.pose_ok = (POSE_SOURCE == 'odom')        # odom assumed available; amcl waits for TF

        # ── LiDAR state ──
        self.front_min = float('inf')
        self.have_scan = False

        # ── Plan / control state ──
        self.path = []            # list of (wx, wy) world waypoints, start -> goal
        self.path_idx = 0
        self.last_plan_time = 0.0
        self.need_replan = True
        self.recovery_ticks = 0
        self.recovery_count = 0
        self.state = WAIT_FOR_POSE
        self.start_time = time.time()
        self.done_logged = False

        # ── ROS interfaces ──
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL, 10)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.scan_callback, 10)

        if POSE_SOURCE == 'amcl':
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            if SEED_INITIAL_POSE:
                self.initial_pose_pub = self.create_publisher(
                    PoseWithCovarianceStamped, INITIALPOSE, 10)
                self._seeded = False
                self.create_timer(INITIAL_POSE_DELAY_S, self._seed_initial_pose)
        else:
            self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        gr, gc = self.world_to_cell(GOAL_X, GOAL_Y)
        self.get_logger().info(
            f'astar_nav up | map {self.W}x{self.H} @ {self.pres:.3f} m/cell '
            f'origin=({self.ox:.2f},{self.oy:.2f}) | pose_source={POSE_SOURCE} | '
            f'goal=({GOAL_X:.2f},{GOAL_Y:.2f}) cell=({gr},{gc})')

    # ── Coordinate helpers (grids are Y-UP) ─────────────────────────────────--
    def world_to_cell(self, wx, wy):
        col = int((wx - self.ox) / self.pres)
        row = int((wy - self.oy) / self.pres)
        return row, col

    def cell_to_world(self, row, col):
        wx = self.ox + (col + 0.5) * self.pres
        wy = self.oy + (row + 0.5) * self.pres
        return wx, wy

    def in_bounds(self, row, col):
        return 0 <= row < self.H and 0 <= col < self.W

    def dist_to_goal(self):
        return math.hypot(GOAL_X - self.current_x, GOAL_Y - self.current_y)

    # ── AMCL initial pose ────────────────────────────────────────────────────
    def _seed_initial_pose(self):
        if self._seeded:
            return
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = MAP_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(START_X)
        msg.pose.pose.position.y = float(START_Y)
        msg.pose.pose.orientation.z = math.sin(START_YAW / 2.0)
        msg.pose.pose.orientation.w = math.cos(START_YAW / 2.0)
        msg.pose.covariance[0]  = 0.05   # x
        msg.pose.covariance[7]  = 0.05   # y
        msg.pose.covariance[35] = 0.05   # yaw
        self.initial_pose_pub.publish(msg)
        self._seeded = True
        self.get_logger().info(
            f'AMCL initial pose seeded at ({START_X:.2f},{START_Y:.2f}) yaw={START_YAW:.2f}')

    # ── Pose updates ───────────────────────────────────────────────────────--
    def update_pose_from_tf(self):
        try:
            t = self.tf_buffer.lookup_transform(MAP_FRAME, BASE_FRAME, rclpy.time.Time())
            self.current_x = t.transform.translation.x
            self.current_y = t.transform.translation.y
            q = t.transform.rotation
            self.current_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.pose_ok = True
        except (LookupException, ConnectivityException, ExtrapolationException):
            pass  # keep last known pose; AMCL not ready yet

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose_ok = True

    # ── LiDAR: emergency arc + dynamic obstacle projection ─────────────────────
    def scan_callback(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        idx = np.arange(ranges.size)
        base_ang = msg.angle_min + idx * msg.angle_increment + LIDAR_YAW_OFFSET
        valid = (np.isfinite(ranges) & (ranges > msg.range_min) &
                 (ranges < min(msg.range_max, OBSTACLE_MAX_RANGE)))

        # Forward arc minimum (for emergency stop) — uses base-frame angle ~0.
        fa = np.abs(np.arctan2(np.sin(base_ang), np.cos(base_ang)))
        front_mask = valid & (fa < math.radians(FRONT_ARC_DEG))
        self.front_min = float(ranges[front_mask].min()) if front_mask.any() else float('inf')

        # Project valid returns into the map frame -> dynamic obstacle layer.
        self.obstacle_grid *= OBSTACLE_DECAY
        r = ranges[valid]
        a = base_ang[valid] + self.current_yaw
        wx = self.current_x + r * np.cos(a)
        wy = self.current_y + r * np.sin(a)
        cols = ((wx - self.ox) / self.pres).astype(np.int64)
        rows = ((wy - self.oy) / self.pres).astype(np.int64)
        inb = (rows >= 0) & (rows < self.H) & (cols >= 0) & (cols < self.W)
        self.obstacle_grid[rows[inb], cols[inb]] = OBSTACLE_HIT
        self.have_scan = True

    # ── Build inflated cost grid ───────────────────────────────────────────--
    def _build_blocked(self):
        occ = self.occupied.copy()
        if not ALLOW_UNKNOWN:
            occ |= self.unknown
        occ |= (self.obstacle_grid >= OBSTACLE_THRESH)
        inflated = cv2.dilate(occ.astype(np.uint8), self.kernel)
        return inflated > 0

    # ── A* planner (8-connected, octile heuristic, no corner cutting) ───────---
    def _nearest_free(self, row, col, max_r=20):
        """If (row,col) is blocked, spiral outward for the nearest free cell."""
        if self.in_bounds(row, col) and not self.blocked[row, col]:
            return (row, col)
        for rad in range(1, max_r + 1):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad:
                        continue
                    rr, cc = row + dr, col + dc
                    if self.in_bounds(rr, cc) and not self.blocked[rr, cc]:
                        return (rr, cc)
        return None

    def astar(self, start_rc, goal_rc):
        sr, sc = start_rc
        goal = self._nearest_free(*goal_rc)
        if goal is None:
            return None
        gr, gc = goal
        blk = self.blocked

        # 8-connectivity with diagonal cost; (dr, dc, cost).
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]

        def h(r, c):
            dr, dc = abs(r - gr), abs(c - gc)
            return (dr + dc) + (1.4142 - 2.0) * min(dr, dc)

        open_heap = [(h(sr, sc), 0.0, (sr, sc))]
        g_score = {(sr, sc): 0.0}
        came = {}
        counter = 0
        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur == (gr, gc):
                # reconstruct
                cells = [cur]
                while cur in came:
                    cur = came[cur]
                    cells.append(cur)
                cells.reverse()
                return [self.cell_to_world(r, c) for (r, c) in cells]
            cr, cc = cur
            if g > g_score.get(cur, float('inf')):
                continue
            for dr, dc, cost in nbrs:
                nr, nc = cr + dr, cc + dc
                if not self.in_bounds(nr, nc):
                    continue
                # Don't let A* enter a blocked cell (the start is allowed even if blocked).
                if blk[nr, nc] and (nr, nc) != (gr, gc):
                    continue
                # No corner cutting on diagonals.
                if dr != 0 and dc != 0:
                    if blk[cr + dr, cc] or blk[cr, cc + dc]:
                        continue
                ng = g + cost
                if ng < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = ng
                    came[(nr, nc)] = cur
                    counter += 1
                    heapq.heappush(open_heap, (ng + h(nr, nc), ng, (nr, nc)))
        return None

    def _diag(self, start, goal, sfree, gfree):
        """Log why a plan is (un)reachable: grid composition + endpoint status."""
        total = self.H * self.W
        occ = int(self.occupied.sum())
        unk = int(self.unknown.sum())
        blk = int(self.blocked.sum())
        dyn = int((self.obstacle_grid >= OBSTACLE_THRESH).sum())
        self.get_logger().warn(
            f'PLAN DIAG | grid {self.W}x{self.H} '
            f'occ={100*occ/total:.0f}% unknown={100*unk/total:.0f}% '
            f'blocked(after inflate+dyn)={100*blk/total:.0f}% live_obs_cells={dyn}')
        self.get_logger().warn(
            f'  start cell={start} in_bounds={self.in_bounds(*start)} '
            f'blocked={self.in_bounds(*start) and bool(self.blocked[start])} '
            f'-> nearest_free={sfree}  pose=({self.current_x:.2f},{self.current_y:.2f})')
        self.get_logger().warn(
            f'  goal  cell={goal} in_bounds={self.in_bounds(*goal)} '
            f'blocked={self.in_bounds(*goal) and bool(self.blocked[goal])} '
            f'-> nearest_free={gfree}  goal=({GOAL_X:.2f},{GOAL_Y:.2f})')

    def replan(self):
        self.blocked = self._build_blocked()
        start = self.world_to_cell(self.current_x, self.current_y)
        goal = self.world_to_cell(GOAL_X, GOAL_Y)
        self.last_plan_time = time.time()
        self.need_replan = False

        if not self.in_bounds(*start):
            self.get_logger().error(
                f'Start cell {start} OFF-MAP — pose wrong? '
                f'pose=({self.current_x:.2f},{self.current_y:.2f}) '
                f'origin=({self.ox:.2f},{self.oy:.2f}) size={self.W}x{self.H}')
            self.path = []
            return False
        if not self.in_bounds(*goal):
            self.get_logger().error(
                f'Goal cell {goal} OFF-MAP — check GOAL_X/GOAL_Y vs map origin.')
            self.path = []
            return False

        # Relocate BOTH endpoints off blocked cells (was goal-only before).
        sfree = self._nearest_free(*start)
        gfree = self._nearest_free(*goal)

        path = self.astar(sfree, gfree) if (sfree and gfree) else None
        if path is None or len(path) < 1:
            self.get_logger().warn('A* found no path to goal.')
            self._diag(start, goal, sfree, gfree)
            self.path = []
            return False
        self.path = path
        self.path_idx = 0
        return True

    def path_is_blocked(self):
        """Cheap check: is any upcoming waypoint now blocked in the latest grid?"""
        for (wx, wy) in self.path[self.path_idx:self.path_idx + 20]:
            r, c = self.world_to_cell(wx, wy)
            if self.in_bounds(r, c) and self.blocked[r, c]:
                return True
        return False

    # ── Pure-pursuit follower ──────────────────────────────────────────────--
    def follow_path(self):
        msg = Twist()

        # Emergency stop: something close ahead. Stop, route around it.
        if self.front_min < SAFE_STOP_DIST:
            self.need_replan = True
            self.cmd_pub.publish(Twist())   # stop
            self.get_logger().warn(
                f'Obstacle {self.front_min:.2f} m ahead — stopping, replanning')
            return

        # Advance the lookahead index past waypoints we've already passed.
        while (self.path_idx < len(self.path) - 1 and
               math.hypot(self.path[self.path_idx][0] - self.current_x,
                          self.path[self.path_idx][1] - self.current_y) < LOOKAHEAD):
            self.path_idx += 1

        tx, ty = self.path[self.path_idx]
        heading = math.atan2(ty - self.current_y, tx - self.current_x)
        err = math.atan2(math.sin(heading - self.current_yaw),
                         math.cos(heading - self.current_yaw))

        msg.angular.z = max(-MAX_TURN, min(MAX_TURN, HEADING_KP * err))
        msg.linear.x = 0.0 if abs(err) > TURN_IN_PLACE else FORWARD_SPEED
        self.cmd_pub.publish(msg)

    # ── Main control loop ──────────────────────────────────────────────────--
    def control_loop(self):
        if POSE_SOURCE == 'amcl':
            self.update_pose_from_tf()

        if self.state == WAIT_FOR_POSE:
            if self.pose_ok and self.have_scan:
                self.state = NAVIGATE
                self.need_replan = True
                self.get_logger().info('Pose + scan ready — NAVIGATE')
            else:
                self.cmd_pub.publish(Twist())

        elif self.state == NAVIGATE:
            if self.dist_to_goal() < GOAL_TOLERANCE:
                self.state = GOAL_REACHED
                self.cmd_pub.publish(Twist())
            else:
                # Refresh the obstacle grid view, then decide if we must replan.
                self.blocked = self._build_blocked()
                due = (time.time() - self.last_plan_time) > REPLAN_PERIOD_S
                if self.need_replan or due or not self.path or self.path_is_blocked():
                    ok = self.replan()
                    if not ok:
                        self.recovery_count += 1
                        if self.recovery_count >= MAX_RECOVERIES:
                            self.state = GIVE_UP
                        else:
                            self.state = RECOVERY
                            self.recovery_ticks = 0
                        self.cmd_pub.publish(Twist())
                        self._maybe_show()
                        return
                    self.recovery_count = 0
                self.follow_path()

        elif self.state == RECOVERY:
            # Back up briefly + rotate to expose new free space, then retry.
            msg = Twist()
            if self.recovery_ticks < 8 and self.front_min > SAFE_STOP_DIST:
                msg.linear.x = -0.06
            else:
                msg.angular.z = 0.5
            self.cmd_pub.publish(msg)
            self.recovery_ticks += 1
            if self.recovery_ticks >= 20:
                self.state = NAVIGATE
                self.need_replan = True

        elif self.state == GOAL_REACHED:
            self.cmd_pub.publish(Twist())
            if not self.done_logged:
                self.done_logged = True
                dt = time.time() - self.start_time
                self.get_logger().info(
                    f'GOAL REACHED ({GOAL_X:.2f},{GOAL_Y:.2f}) in {dt:.1f}s — done')

        elif self.state == GIVE_UP:
            self.cmd_pub.publish(Twist())
            if not self.done_logged:
                self.done_logged = True
                self.get_logger().error('No path after repeated attempts — giving up.')

        self._maybe_show()

    # ── Debug visualisation ────────────────────────────────────────────────--
    def _maybe_show(self):
        if not SHOW_VIS:
            return
        img = np.full((self.H, self.W, 3), 255, dtype=np.uint8)   # free = white
        img[self.unknown] = (160, 160, 160)                        # unknown = grey
        img[self.blocked] = (180, 180, 255)                        # inflated = light red
        img[self.occupied] = (0, 0, 0)                             # walls = black
        img[self.obstacle_grid >= OBSTACLE_THRESH] = (0, 0, 255)   # live obstacle = red

        for (wx, wy) in self.path:                                 # path = green
            r, c = self.world_to_cell(wx, wy)
            if self.in_bounds(r, c):
                img[r, c] = (0, 200, 0)

        gr, gc = self.world_to_cell(GOAL_X, GOAL_Y)                # goal = blue dot
        rr, rc = self.world_to_cell(self.current_x, self.current_y)
        disp = np.flipud(img)                                      # back to y-down for display
        disp = cv2.resize(disp, (self.W * VIS_SCALE, self.H * VIS_SCALE),
                          interpolation=cv2.INTER_NEAREST)
        if self.in_bounds(gr, gc):
            cv2.circle(disp, (gc * VIS_SCALE, (self.H - 1 - gr) * VIS_SCALE),
                       6, (255, 0, 0), -1)
        if self.in_bounds(rr, rc):
            cv2.circle(disp, (rc * VIS_SCALE, (self.H - 1 - rr) * VIS_SCALE),
                       5, (0, 140, 255), -1)
        cv2.putText(disp, f'{self.state}  d={self.dist_to_goal():.2f}m',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imshow('astar_nav', disp)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = AStarNav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())   # stop the robot
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
