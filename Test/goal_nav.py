#!/usr/bin/env python3
"""
goal_nav.py — Minimal map-based go-to-goal for TurtleBot4 (ROS2), no Nav2.

Does exactly four things:
  1. Loads a pre-built occupancy map (.pgm + .yaml).
  2. Localises on it (AMCL via TF map->base_link; odom fallback for testing).
  3. Plans a path to (GOAL_X, GOAL_Y) with a custom A* on the STATIC map.
  4. Drives there with pure pursuit.

No obstacle avoidance, no LiDAR processing, no replanning loop — the map is
assumed static. (If the robot strays far from the path, e.g. after an AMCL
correction, it replans once.)

RUN
    python3 goal_nav.py
"""

import os, math, time, heapq, yaml
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# ── Robot / Topic Config ──────────────────────────────────────────────────────
NAMESPACE   = 'T8'
CMD_VEL     = f'{NAMESPACE}/cmd_vel'
ODOM_TOPIC  = f'{NAMESPACE}/odom'
INITIALPOSE = f'{NAMESPACE}/initialpose'
MAP_FRAME   = 'map'
BASE_FRAME  = f'{NAMESPACE}/base_link'

# ── Map ───────────────────────────────────────────────────────────────────────
MAP_YAML_PATH = os.path.expanduser('~/Desktop/lab_map.yaml')
ALLOW_UNKNOWN = False     # treat unknown (-1) map cells as obstacles

# ── Pose Source ───────────────────────────────────────────────────────────────
POSE_SOURCE = 'amcl'      # 'amcl' (map->base_link TF) or 'odom' (testing only)
SEED_INITIAL_POSE = True  # publish AMCL initialpose on startup (amcl only)
START_X, START_Y, START_YAW = 0.0, 0.0, 0.0   # robot's true pose on the map
INITIAL_POSE_DELAY_S = 2.0

# ── Goal ──────────────────────────────────────────────────────────────────────
GOAL_X, GOAL_Y = 0.322, -2.56
GOAL_TOLERANCE = 0.18

# ── Planning ──────────────────────────────────────────────────────────────────
INFLATION_RADIUS = 0.15   # metres — grow walls so the path keeps clearance
PLAN_RESOLUTION  = 0.05    # metres/cell for the planner
REPLAN_STRAY_DIST = 0.40   # metres off-path before replanning

# ── Control ───────────────────────────────────────────────────────────────────
FORWARD_SPEED = 0.15
MAX_TURN      = 1.0
HEADING_KP    = 1.6
LOOKAHEAD     = 0.40
TURN_IN_PLACE = math.radians(45)

# ── Visualisation ─────────────────────────────────────────────────────────────
SHOW_VIS  = True
VIS_SCALE = 4

# ── States ────────────────────────────────────────────────────────────────────
WAIT_FOR_POSE = 'WAIT_FOR_POSE'
NAVIGATE      = 'NAVIGATE'
GOAL_REACHED  = 'GOAL_REACHED'
NO_PATH       = 'NO_PATH'


def load_map(yaml_path):
    """Load map_server .yaml/.pgm. Grids returned Y-UP (row 0 = lowest y)."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    img_rel = cfg['image']
    img_path = img_rel if os.path.isabs(img_rel) \
        else os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_rel)
    res     = float(cfg['resolution'])
    origin  = cfg['origin']
    negate  = int(cfg.get('negate', 0))
    occ_th  = float(cfg.get('occupied_thresh', 0.65))
    free_th = float(cfg.get('free_thresh', 0.196))

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Could not read map image: {img_path}')
    p = img.astype(np.float64)
    occ = p / 255.0 if negate else (255.0 - p) / 255.0
    occupied = occ > occ_th
    free     = occ < free_th
    unknown  = ~occupied & ~free
    occupied = np.flipud(occupied)   # row 0 -> bottom (min y)
    unknown  = np.flipud(unknown)
    return occupied, unknown, res, (float(origin[0]), float(origin[1]))


def coarsen(mask, f):
    if f <= 1:
        return mask
    h, w = mask.shape
    hp = ((h + f - 1) // f) * f
    wp = ((w + f - 1) // f) * f
    padded = np.zeros((hp, wp), dtype=bool)
    padded[:h, :w] = mask
    return padded.reshape(hp // f, f, wp // f, f).max(axis=(1, 3))


class GoalNav(Node):

    def __init__(self):
        super().__init__('goal_nav')

        # ── Map + static blocked grid (built once) ──
        occ, unk, res0, origin = load_map(MAP_YAML_PATH)
        factor = max(1, int(round(PLAN_RESOLUTION / res0)))
        self.occupied = coarsen(occ, factor)
        self.unknown  = coarsen(unk, factor)
        self.pres = res0 * factor
        self.ox, self.oy = origin
        self.H, self.W = self.occupied.shape

        infl = max(1, int(math.ceil(INFLATION_RADIUS / self.pres)))
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * infl + 1, 2 * infl + 1))
        occ_in = self.occupied.copy()
        if not ALLOW_UNKNOWN:
            occ_in |= self.unknown
        self.blocked = cv2.dilate(occ_in.astype(np.uint8), self.kernel) > 0

        # ── Pose state ──
        self.current_x, self.current_y, self.current_yaw = START_X, START_Y, START_YAW
        self.pose_ok = (POSE_SOURCE == 'odom')

        # ── Plan / control state ──
        self.path = []
        self.path_idx = 0
        self.state = WAIT_FOR_POSE
        self.start_time = time.time()
        self.logged = False

        # ── ROS interfaces ──
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL, 10)
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

        self.create_timer(0.1, self.control_loop)

        gr, gc = self.world_to_cell(GOAL_X, GOAL_Y)
        self.get_logger().info(
            f'goal_nav up | map {self.W}x{self.H} @ {self.pres:.3f} m/cell '
            f'origin=({self.ox:.2f},{self.oy:.2f}) | pose_source={POSE_SOURCE} | '
            f'goal=({GOAL_X:.2f},{GOAL_Y:.2f}) cell=({gr},{gc})')

    # ── Coordinate helpers (Y-UP) ──
    def world_to_cell(self, wx, wy):
        return int((wy - self.oy) / self.pres), int((wx - self.ox) / self.pres)

    def cell_to_world(self, row, col):
        return self.ox + (col + 0.5) * self.pres, self.oy + (row + 0.5) * self.pres

    def in_bounds(self, row, col):
        return 0 <= row < self.H and 0 <= col < self.W

    def dist_to_goal(self):
        return math.hypot(GOAL_X - self.current_x, GOAL_Y - self.current_y)

    # ── AMCL initial pose ──
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
        msg.pose.covariance[0] = msg.pose.covariance[7] = msg.pose.covariance[35] = 0.05
        self.initial_pose_pub.publish(msg)
        self._seeded = True
        self.get_logger().info(
            f'AMCL initial pose seeded at ({START_X:.2f},{START_Y:.2f}) yaw={START_YAW:.2f}')

    # ── Pose updates ──
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
            pass

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose_ok = True

    # ── A* (8-connected, octile heuristic, no corner cutting) ──
    def _nearest_free(self, row, col, max_r=20):
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
        gr, gc = goal_rc
        blk = self.blocked
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]

        def h(r, c):
            dr, dc = abs(r - gr), abs(c - gc)
            return (dr + dc) + (1.4142 - 2.0) * min(dr, dc)

        open_heap = [(h(sr, sc), 0.0, (sr, sc))]
        g_score = {(sr, sc): 0.0}
        came = {}
        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur == (gr, gc):
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
                if blk[nr, nc] and (nr, nc) != (gr, gc):
                    continue
                if dr != 0 and dc != 0 and (blk[cr + dr, cc] or blk[cr, cc + dc]):
                    continue
                ng = g + cost
                if ng < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = ng
                    came[(nr, nc)] = cur
                    heapq.heappush(open_heap, (ng + h(nr, nc), ng, (nr, nc)))
        return None

    def plan(self):
        start = self.world_to_cell(self.current_x, self.current_y)
        goal = self.world_to_cell(GOAL_X, GOAL_Y)
        if not self.in_bounds(*start) or not self.in_bounds(*goal):
            self.get_logger().error(
                f'Start {start} or goal {goal} off-map ({self.W}x{self.H}). '
                f'Check pose / GOAL vs origin ({self.ox:.2f},{self.oy:.2f}).')
            return False
        sfree = self._nearest_free(*start)
        gfree = self._nearest_free(*goal)
        path = self.astar(sfree, gfree) if (sfree and gfree) else None
        if not path:
            blk_pct = 100 * int(self.blocked.sum()) / (self.H * self.W)
            self.get_logger().error(
                f'No path. blocked={blk_pct:.0f}% start_free={sfree} goal_free={gfree}. '
                f'Try lowering INFLATION_RADIUS or check the map is connected.')
            return False
        self.path = path
        self.path_idx = 0
        self.get_logger().info(f'Path planned: {len(path)} waypoints.')
        return True

    def strayed(self):
        if not self.path:
            return True
        d = min(math.hypot(wx - self.current_x, wy - self.current_y)
                for (wx, wy) in self.path)
        return d > REPLAN_STRAY_DIST

    # ── Pure-pursuit follower ──
    def follow_path(self):
        while (self.path_idx < len(self.path) - 1 and
               math.hypot(self.path[self.path_idx][0] - self.current_x,
                          self.path[self.path_idx][1] - self.current_y) < LOOKAHEAD):
            self.path_idx += 1
        tx, ty = self.path[self.path_idx]
        heading = math.atan2(ty - self.current_y, tx - self.current_x)
        err = math.atan2(math.sin(heading - self.current_yaw),
                         math.cos(heading - self.current_yaw))
        msg = Twist()
        msg.angular.z = max(-MAX_TURN, min(MAX_TURN, HEADING_KP * err))
        msg.linear.x = 0.0 if abs(err) > TURN_IN_PLACE else FORWARD_SPEED
        self.cmd_pub.publish(msg)

    # ── Main loop ──
    def control_loop(self):
        if POSE_SOURCE == 'amcl':
            self.update_pose_from_tf()

        if self.state == WAIT_FOR_POSE:
            if self.pose_ok:
                self.state = NAVIGATE if self.plan() else NO_PATH
            else:
                self.cmd_pub.publish(Twist())

        elif self.state == NAVIGATE:
            if self.dist_to_goal() < GOAL_TOLERANCE:
                self.state = GOAL_REACHED
                self.cmd_pub.publish(Twist())
            else:
                if self.strayed() and not self.plan():
                    self.state = NO_PATH
                    self.cmd_pub.publish(Twist())
                else:
                    self.follow_path()

        elif self.state == GOAL_REACHED:
            self.cmd_pub.publish(Twist())
            if not self.logged:
                self.logged = True
                self.get_logger().info(
                    f'GOAL REACHED ({GOAL_X:.2f},{GOAL_Y:.2f}) in '
                    f'{time.time() - self.start_time:.1f}s')

        elif self.state == NO_PATH:
            self.cmd_pub.publish(Twist())

        self._maybe_show()

    # ── Debug window ──
    def _maybe_show(self):
        if not SHOW_VIS:
            return
        img = np.full((self.H, self.W, 3), 255, dtype=np.uint8)
        img[self.unknown] = (160, 160, 160)
        img[self.blocked] = (180, 180, 255)
        img[self.occupied] = (0, 0, 0)
        for (wx, wy) in self.path:
            r, c = self.world_to_cell(wx, wy)
            if self.in_bounds(r, c):
                img[r, c] = (0, 200, 0)
        disp = cv2.resize(np.flipud(img), (self.W * VIS_SCALE, self.H * VIS_SCALE),
                          interpolation=cv2.INTER_NEAREST)
        gr, gc = self.world_to_cell(GOAL_X, GOAL_Y)
        rr, rc = self.world_to_cell(self.current_x, self.current_y)
        if self.in_bounds(gr, gc):
            cv2.circle(disp, (gc * VIS_SCALE, (self.H - 1 - gr) * VIS_SCALE), 6, (255, 0, 0), -1)
        if self.in_bounds(rr, rc):
            cv2.circle(disp, (rc * VIS_SCALE, (self.H - 1 - rr) * VIS_SCALE), 5, (0, 140, 255), -1)
        cv2.putText(disp, f'{self.state}  d={self.dist_to_goal():.2f}m',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imshow('goal_nav', disp)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = GoalNav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
