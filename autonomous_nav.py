#!/usr/bin/env python3
"""
Autonomous cube finder — map-based A* path planning, no Nav2.

Usage:
    python3 autonomous_nav.py <target_x> <target_y> [<map_yaml>]
    e.g.:  python3 autonomous_nav.py 2.5 0.3 ~/Desktop/demo_map.yaml

    map_yaml defaults to ~/Desktop/demo_map.yaml

Phase-1 SLAM map → A* global path (static walls avoided from the start).
Phase-2 obstacles (judge's cylinders) → LiDAR detects them, adds to a
dynamic obstacle layer, path is re-planned around them.
"""

import rclpy, math, cv2, os, sys, time, heapq
import numpy as np
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from cv_bridge import CvBridge

# ── Robot Config ──────────────────────────────────────────────────────────────
NAMESPACE     = 'T24'
FORWARD_SPEED = 0.15   # m/s
TURN_SPEED    = 0.5    # rad/s
AVOID_DIST    = 0.40   # m — front obstacle threshold
ROBOT_RADIUS  = 0.22   # m — map inflation radius

# ── Waypoint Following ────────────────────────────────────────────────────────
WAYPOINT_ACCEPT   = 0.25   # m — advance to next waypoint when within this
HEADING_THRESHOLD = 0.15   # rad — start driving when yaw error below this
HEADING_KP        = 1.2

# ── LiDAR Geometry (matches demo_test.py) ─────────────────────────────────────
FRONT_BEARING_DEG = 90
FRONT_ARC_DEG     = 30
CUBE_ARC_DEG      = 10

# ── Red Cube HSV ──────────────────────────────────────────────────────────────
RED_LOW1  = np.array([0,   95,  95])
RED_HIGH1 = np.array([8,  255, 255])
RED_LOW2  = np.array([177, 95,  95])
RED_HIGH2 = np.array([180, 255, 255])
MIN_PIXELS_CENTRE    = 8500
CENTRE_TOL_PX        = 15
CENTRE_KP            = 0.3
CENTRE_TIMEOUT_TICKS = 100
LOST_CUBE_TICKS      = 50

# ── Capture / Spin / Time ─────────────────────────────────────────────────────
CAPTURE_TICKS = 50          # 5 s at 10 Hz
SPIN_RATE     = 0.5         # rad/s
SPIN_TOTAL    = 2 * math.pi
TIME_LIMIT    = 480.0       # 8 min

# ── States ────────────────────────────────────────────────────────────────────
PLANNING    = 'PLANNING'    # waiting after initial plan so user can see path
NAVIGATING  = 'NAVIGATING'
AVOIDING    = 'AVOIDING'
SPINNING    = 'SPINNING'
CENTRE_CUBE = 'CENTRE_CUBE'
CAPTURE     = 'CAPTURE'
RETURNING   = 'RETURNING'
DONE        = 'DONE'

PLAN_WAIT_TICKS = 30   # 3 s at 10 Hz — pause so Rviz path is visible before moving


# ─────────────────────────────────────────────────────────────────────────────
# Map loading and A* planner
# ─────────────────────────────────────────────────────────────────────────────

class OccupancyMap:
    """
    Loads a ROS2 map (YAML + PGM from slam_toolbox) and provides A* planning.

    Coordinate convention (standard ROS map_server):
      world(x,y) → col = (x - origin_x) / res
                   row = height - (y - origin_y) / res
    """

    def __init__(self, yaml_path: str, inflation_m: float = ROBOT_RADIUS):
        yaml_path = os.path.expanduser(yaml_path)
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)

        img_path = meta['image']
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(yaml_path), img_path)
        img = cv2.imread(os.path.expanduser(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'Cannot load PGM: {img_path}')

        self.resolution = float(meta['resolution'])
        origin = meta['origin']
        self.origin_x = float(origin[0])
        self.origin_y = float(origin[1])
        self.height, self.width = img.shape

        negate     = int(meta.get('negate', 0))
        occ_thresh = float(meta.get('occupied_thresh', 0.65))

        whiteness = img.astype(float) / 255.0
        if negate:
            whiteness = 1.0 - whiteness
        # probability of occupancy: bright pixel = free (p~0), dark = occupied (p~1)
        self._static  = (1.0 - whiteness) > occ_thresh
        self._dynamic = np.zeros_like(self._static, dtype=bool)

        self._inflate_r = max(1, int(math.ceil(inflation_m / self.resolution)))
        self._inflated  = None
        self._rebuild_inflated()

    def _rebuild_inflated(self):
        combined = self._static | self._dynamic
        r = self._inflate_r
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        self._inflated = cv2.dilate(combined.astype(np.uint8), kernel).astype(bool)

    # ── Coordinate helpers ───────────────────────────────────────────────────

    def world_to_grid(self, wx, wy):
        col = int((wx - self.origin_x) / self.resolution)
        row = self.height - int((wy - self.origin_y) / self.resolution)
        return col, row

    def grid_to_world(self, col, row):
        wx = self.origin_x + (col + 0.5) * self.resolution
        wy = self.origin_y + (self.height - row - 0.5) * self.resolution
        return wx, wy

    def in_bounds(self, col, row):
        return 0 <= col < self.width and 0 <= row < self.height

    def is_free(self, col, row):
        return self.in_bounds(col, row) and not self._inflated[row, col]

    # ── Dynamic obstacle API ─────────────────────────────────────────────────

    def mark_obstacles(self, world_points):
        """Add world-frame obstacle points to dynamic layer; rebuild if changed."""
        changed = False
        for wx, wy in world_points:
            c, r = self.world_to_grid(wx, wy)
            if self.in_bounds(c, r) and not self._static[r, c] and not self._dynamic[r, c]:
                self._dynamic[r, c] = True
                changed = True
        if changed:
            self._rebuild_inflated()

    # ── Path planning ────────────────────────────────────────────────────────

    def plan(self, from_world, to_world):
        """A* from from_world to to_world. Returns list of world (x,y) waypoints."""
        s = self.world_to_grid(*from_world)
        g = self.world_to_grid(*to_world)
        if not self.is_free(*s):
            s = self._nearest_free(s)
        if not self.is_free(*g):
            g = self._nearest_free(g)
        cells = self._astar(s, g)
        if not cells:
            return []
        pts = [self.grid_to_world(c, r) for c, r in cells]
        return self._simplify(pts)

    def _nearest_free(self, cell, max_r=20):
        c0, r0 = cell
        for r in range(1, max_r):
            for dc in range(-r, r+1):
                for dr in range(-r, r+1):
                    if abs(dc) == r or abs(dr) == r:
                        nc, nr = c0+dc, r0+dr
                        if self.in_bounds(nc, nr) and not self._inflated[nr, nc]:
                            return nc, nr
        return cell

    def _astar(self, start, goal):
        gc, gr = goal
        def h(c, r): return math.hypot(c - gc, r - gr)
        DIRS = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1,-1, 1.414), (-1, 1, 1.414), (1,-1, 1.414), (1, 1, 1.414),
        ]
        heap = [(h(*start), start)]
        came = {start: None}
        g    = {start: 0.0}

        while heap:
            _, cur = heapq.heappop(heap)
            if cur == goal:
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = came[cur]
                return path[::-1]
            cc, cr = cur
            for dc, dr, cost in DIRS:
                nb = (cc+dc, cr+dr)
                if not self.is_free(*nb):
                    continue
                # Prevent corner cutting: diagonal move blocked if either
                # adjacent cardinal cell is occupied
                if dc != 0 and dr != 0:
                    if not self.is_free(cc+dc, cr) or not self.is_free(cc, cr+dr):
                        continue
                ng = g[cur] + cost
                if ng < g.get(nb, float('inf')):
                    g[nb] = ng
                    came[nb] = cur
                    heapq.heappush(heap, (ng + h(*nb), nb))
        return []

    def _simplify(self, pts, min_dist=0.30):
        """Keep waypoints at least min_dist apart; always keep last point."""
        if len(pts) <= 1:
            return pts
        out = [pts[0]]
        for p in pts[1:-1]:
            if math.hypot(p[0]-out[-1][0], p[1]-out[-1][1]) >= min_dist:
                out.append(p)
        out.append(pts[-1])
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Main node
# ─────────────────────────────────────────────────────────────────────────────

class AutonomousNav(Node):

    def __init__(self, target_x: float, target_y: float, map_yaml: str):
        super().__init__('autonomous_nav')
        self.target_x = target_x
        self.target_y = target_y

        self.get_logger().info(f'Loading map: {map_yaml}')
        self.omap = OccupancyMap(map_yaml)
        self.get_logger().info(
            f'Map {self.omap.width}x{self.omap.height} '
            f'res={self.omap.resolution:.3f} m/px')

        self.cmd_pub  = self.create_publisher(Twist, f'{NAMESPACE}/cmd_vel', 10)
        # Latched QoS so Rviz gets map/path even if it connects after publish
        _latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE)
        self.map_pub        = self.create_publisher(OccupancyGrid, '/map', _latched)
        self.path_pub       = self.create_publisher(Path, '/planned_path', 10)
        self.robot_pose_pub = self.create_publisher(PoseStamped, '/robot_pose', 10)

        self.create_subscription(LaserScan, f'{NAMESPACE}/scan', self.scan_cb, 10)
        self.create_subscription(
            CompressedImage,
            f'{NAMESPACE}/oakd/rgb/image_raw/compressed',
            self.image_cb, 10)
        self.create_subscription(Odometry, f'{NAMESPACE}/odom', self.odom_cb, 10)

        # LiDAR
        self.nearest_front      = float('inf')
        self.nearest_cube_front = float('inf')

        # Odometry
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0

        # Camera
        self.bridge      = CvBridge()
        self.cube_cx     = None
        self.image_width = None
        self.latest_img  = None
        self.red_pixels  = 0

        # Navigation state
        self.waypoints          = []
        self.nav_target         = (target_x, target_y)
        self.nav_arrive_state   = SPINNING
        self.avoid_return_state = NAVIGATING

        # Spin
        self.spin_last_yaw    = None
        self.spin_accumulated = 0.0

        # Centre on cube
        self.centre_ticks    = 0
        self.lost_cube_ticks = 0

        # Capture
        self.capture_ticks   = 0
        self.photo_robot_pos = None
        self.cube_world_pos  = None

        # Mission
        self.state          = PLANNING
        self.plan_ticks     = 0
        self.start_time     = time.time()
        self.mission_logged = False

        # Initial A* plan to target
        self._replan((target_x, target_y), SPINNING)
        self._publish_map()

        self.timer     = self.create_timer(0.1, self.control_loop)
        self.viz_timer = self.create_timer(1.0, self._publish_path)
        self.get_logger().info(
            f'Started | target=({target_x:.2f},{target_y:.2f}) '
            f'waypoints={len(self.waypoints)} state={NAVIGATING}')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def scan_cb(self, msg):
        inc    = msg.angle_increment
        fwd_i  = int(round(math.radians(FRONT_BEARING_DEG) / inc))
        half_f = int(round(math.radians(FRONT_ARC_DEG) / inc))
        half_c = int(round(math.radians(CUBE_ARC_DEG) / inc))
        n = len(msg.ranges)

        def arc_min(lo, hi):
            idxs = [(lo + d) % n for d in range(hi - lo + 1)]
            vals = [msg.ranges[i] for i in idxs
                    if msg.range_min < msg.ranges[i] < msg.range_max]
            return min(vals) if vals else float('inf')

        self.nearest_front      = arc_min(fwd_i - half_f, fwd_i + half_f)
        self.nearest_cube_front = arc_min(fwd_i - half_c, fwd_i + half_c)

    def image_cb(self, msg):
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
        cv2.putText(overlay, f'px:{self.red_pixels}  {self.state}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Detection', overlay)
        cv2.waitKey(1)

        # Cube found while spinning → centre on it
        if self.state == SPINNING and self.red_pixels >= MIN_PIXELS_CENTRE:
            self.stop()
            self.state        = CENTRE_CUBE
            self.centre_ticks = 0
            self.lost_cube_ticks = 0
            self.get_logger().info(
                f'Cube detected ({self.red_pixels}px) → CENTRE_CUBE')

        # Lost-cube hysteresis during centering
        if self.state == CENTRE_CUBE:
            if self.red_pixels < MIN_PIXELS_CENTRE:
                self.lost_cube_ticks += 1
                if self.lost_cube_ticks >= LOST_CUBE_TICKS:
                    self.lost_cube_ticks = 0
                    self.centre_ticks    = 0
                    self.state = SPINNING
                    self._reset_spin()
                    self.get_logger().warn('Lost cube — resuming SPINNING')
            else:
                self.lost_cube_ticks = 0

    def odom_cb(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z))

    # ── Control Loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        if (self.state not in (PLANNING, RETURNING, DONE)
                and time.time() - self.start_time > TIME_LIMIT):
            self.get_logger().warn('Time limit — returning to origin')
            self._replan((0.0, 0.0), DONE)
            self.state = RETURNING
            return

        self._publish_robot_pose()
        {
            PLANNING:    self.do_planning,
            NAVIGATING:  self.do_navigating,
            AVOIDING:    self.do_avoiding,
            SPINNING:    self.do_spinning,
            CENTRE_CUBE: self.do_centre_cube,
            CAPTURE:     self.do_capture,
            RETURNING:   self.do_returning,
            DONE:        self.do_done,
        }[self.state]()

    # ── State Behaviours ──────────────────────────────────────────────────────

    def do_planning(self):
        """Hold still for PLAN_WAIT_TICKS so the path is visible in Rviz before moving."""
        self.stop()
        self.plan_ticks += 1
        if self.plan_ticks == 1:
            self.get_logger().info(
                f'Path planned ({len(self.waypoints)} waypoints). '
                f'Starting in {PLAN_WAIT_TICKS/10:.0f}s — check Rviz now.')
        if self.plan_ticks >= PLAN_WAIT_TICKS:
            self.state = NAVIGATING
            self.start_time = time.time()  # reset timer — don't count planning wait
            self.get_logger().info('Starting navigation')

    def do_navigating(self):
        self._follow_waypoints()

    def do_avoiding(self):
        """Spin left until front is clear, then re-plan."""
        if self.nearest_front > AVOID_DIST * 1.5:
            self.get_logger().info('Obstacle cleared — re-planning')
            self._replan(self.nav_target, self.nav_arrive_state)
            self.state = self.avoid_return_state
        else:
            msg = Twist()
            msg.angular.z = TURN_SPEED
            self.cmd_pub.publish(msg)
            self.get_logger().info(
                f'AVOIDING: front={self.nearest_front:.2f}m spinning left')

    def do_spinning(self):
        """360° in place; image_cb transitions to CENTRE_CUBE if cube seen."""
        if self.spin_last_yaw is not None:
            self.spin_accumulated += abs(
                self._adiff(self.current_yaw, self.spin_last_yaw))
        self.spin_last_yaw = self.current_yaw

        if self.spin_accumulated >= SPIN_TOTAL:
            self.get_logger().info('360° complete — cube not found, returning')
            self.stop()
            self._replan((0.0, 0.0), DONE)
            self.state = RETURNING
            return

        msg = Twist()
        msg.angular.z = SPIN_RATE
        self.cmd_pub.publish(msg)
        self.get_logger().info(
            f'SPINNING {math.degrees(self.spin_accumulated):.0f}° / 360°  '
            f'px={self.red_pixels}')

    def do_centre_cube(self):
        """Rotate to centre cube in image frame."""
        if self.cube_cx is None or self.image_width is None:
            return
        err = self.cube_cx - self.image_width / 2
        self.centre_ticks += 1
        timed_out = self.centre_ticks >= CENTRE_TIMEOUT_TICKS
        if abs(err) <= CENTRE_TOL_PX or timed_out:
            self.capture_ticks = 0
            self.centre_ticks  = 0
            self.state = CAPTURE
            reason = 'timeout' if timed_out else f'err={err:.0f}px'
            self.get_logger().info(f'Cube centred ({reason}) → CAPTURE')
            return
        msg = Twist()
        msg.angular.z = -CENTRE_KP * (err / (self.image_width / 2))
        self.cmd_pub.publish(msg)
        self.get_logger().info(
            f'CENTRE: cx={self.cube_cx} err={err:.0f}px spin={msg.angular.z:.2f}')

    def do_capture(self):
        """Stop, save snapshot, log cube position, then return."""
        self.stop()
        self.capture_ticks += 1

        if self.capture_ticks == CAPTURE_TICKS // 2:
            self.photo_robot_pos = (self.current_x, self.current_y)
            self.get_logger().info(
                f'[REPORT] Robot odom: '
                f'({self.current_x:.3f},{self.current_y:.3f}) m')
            if self.latest_img is not None:
                path = os.path.expanduser('~/detection_snapshot.jpg')
                cv2.imwrite(path, self.latest_img)
                self.get_logger().info(f'Snapshot → {path}')
            if self.nearest_cube_front != float('inf'):
                cx = self.current_x + self.nearest_cube_front * math.cos(self.current_yaw)
                cy = self.current_y + self.nearest_cube_front * math.sin(self.current_yaw)
                self.cube_world_pos = (cx, cy)
                self.get_logger().info(
                    f'[REPORT] Cube world: ({cx:.3f},{cy:.3f}) m')

        self.get_logger().info(f'CAPTURE {self.capture_ticks}/{CAPTURE_TICKS}')

        if self.capture_ticks >= CAPTURE_TICKS:
            self._replan((0.0, 0.0), DONE)
            self.state = RETURNING

    def do_returning(self):
        self._follow_waypoints()

    def do_done(self):
        self.stop()
        if not self.mission_logged:
            self.mission_logged = True
            self._log_summary()

    # ── Navigation Helpers ─────────────────────────────────────────────────────

    def _follow_waypoints(self):
        """Common waypoint-following used by NAVIGATING and RETURNING."""
        if not self.waypoints:
            self.get_logger().info(
                f'Destination reached → {self.nav_arrive_state}')
            self.stop()
            if self.nav_arrive_state == SPINNING:
                self._reset_spin()
            self.state = self.nav_arrive_state
            return

        # Only avoid if obstacle is NOT on the static map (Phase-2 cylinder).
        # Known walls are already accounted for by A* — reacting to them
        # corrupts the dynamic layer and causes re-plan loops.
        if self.nearest_front < AVOID_DIST and not self._static_obstacle_ahead():
            self._mark_front_obstacle()
            self.avoid_return_state = self.state
            self.state = AVOIDING
            self.get_logger().warn(
                f'New obstacle {self.nearest_front:.2f}m → AVOIDING')
            return

        tx, ty = self.waypoints[0]
        dx = tx - self.current_x
        dy = ty - self.current_y
        dist = math.hypot(dx, dy)

        if dist < WAYPOINT_ACCEPT:
            self.waypoints.pop(0)
            self.get_logger().info(
                f'Waypoint done, {len(self.waypoints)} remaining')
            return

        target_angle = math.atan2(dy, dx)
        heading_err  = self._adiff(target_angle, self.current_yaw)

        msg = Twist()
        if abs(heading_err) > HEADING_THRESHOLD:
            msg.angular.z = TURN_SPEED if heading_err > 0 else -TURN_SPEED
        else:
            msg.linear.x  = FORWARD_SPEED
            msg.angular.z = HEADING_KP * heading_err
        self.cmd_pub.publish(msg)
        self.get_logger().info(
            f'→({tx:.2f},{ty:.2f}) dist={dist:.2f}m '
            f'hdg={math.degrees(heading_err):.1f}° '
            f'pos=({self.current_x:.2f},{self.current_y:.2f})')

    def _replan(self, target, on_arrive):
        """Run A* from current position to target and store waypoints."""
        self.nav_target       = target
        self.nav_arrive_state = on_arrive
        pts = self.omap.plan((self.current_x, self.current_y), target)
        if not pts:
            self.get_logger().error(
                f'A* found no path to {target} — driving direct')
            pts = [target]
        # Drop start point if trivially close to current position
        if pts and math.hypot(pts[0][0] - self.current_x,
                               pts[0][1] - self.current_y) < 0.15:
            pts = pts[1:]
        self.waypoints = pts
        self.get_logger().info(
            f'Planned {len(self.waypoints)} waypoints → {target}')
        self._publish_path()

    def _mark_front_obstacle(self):
        """Mark cells in front of the robot as dynamic obstacles, then rebuild map."""
        d = self.nearest_front if self.nearest_front < float('inf') else AVOID_DIST
        perp = self.current_yaw + math.pi / 2
        pts = []
        for depth in [0.8, 1.0, 1.2]:
            ox = self.current_x + d * depth * math.cos(self.current_yaw)
            oy = self.current_y + d * depth * math.sin(self.current_yaw)
            for offset in [-0.15, 0.0, 0.15]:
                pts.append((ox + offset * math.cos(perp),
                             oy + offset * math.sin(perp)))
        self.omap.mark_obstacles(pts)
        self._publish_map()

    def _static_obstacle_ahead(self):
        """True if the nearest front obstacle is already in the static map."""
        d = self.nearest_front
        if d == float('inf'):
            return False
        obs_x = self.current_x + d * math.cos(self.current_yaw)
        obs_y = self.current_y + d * math.sin(self.current_yaw)
        c, r = self.omap.world_to_grid(obs_x, obs_y)
        if not self.omap.in_bounds(c, r):
            return False
        return bool(self.omap._static[r, c])

    def _publish_robot_pose(self):
        ps = PoseStamped()
        ps.header.frame_id = 'odom'
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.pose.position.x = self.current_x
        ps.pose.position.y = self.current_y
        ps.pose.orientation.z = math.sin(self.current_yaw / 2)
        ps.pose.orientation.w = math.cos(self.current_yaw / 2)
        self.robot_pose_pub.publish(ps)

    def _publish_map(self):
        """Publish the inflated occupancy grid (static + dynamic obstacles)."""
        og = OccupancyGrid()
        og.header.frame_id = 'odom'
        og.header.stamp    = self.get_clock().now().to_msg()
        og.info.resolution = self.omap.resolution
        og.info.width      = self.omap.width
        og.info.height     = self.omap.height
        og.info.origin.position.x  = self.omap.origin_x
        og.info.origin.position.y  = self.omap.origin_y
        og.info.origin.orientation.w = 1.0
        # OccupancyGrid row 0 = y=origin_y (bottom); PGM row 0 = top → flip rows
        data = []
        for og_row in range(self.omap.height):
            pgm_row = self.omap.height - 1 - og_row
            for col in range(self.omap.width):
                if self.omap._static[pgm_row, col]:
                    data.append(100)
                elif self.omap._dynamic[pgm_row, col]:
                    data.append(75)   # dynamic obstacle (Phase-2 cylinder)
                elif self.omap._inflated[pgm_row, col]:
                    data.append(50)   # inflation zone shown as gray
                else:
                    data.append(0)
        og.data = data
        self.map_pub.publish(og)

    def _publish_path(self):
        """Publish current waypoint list as a Path for Rviz."""
        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp    = self.get_clock().now().to_msg()
        for wx, wy in self.waypoints:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x  = wx
            ps.pose.position.y  = wy
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _reset_spin(self):
        self.spin_last_yaw    = None
        self.spin_accumulated = 0.0

    def _adiff(self, a, b):
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))

    def stop(self):
        self.cmd_pub.publish(Twist())

    def _log_summary(self):
        e = time.time() - self.start_time
        m, s = divmod(e, 60)
        self.get_logger().info('=' * 55)
        self.get_logger().info('[MISSION SUMMARY]')
        self.get_logger().info(f'  Duration       : {int(m)}m {s:.1f}s')
        self.get_logger().info(
            f'  Target given   : ({self.target_x:.3f},{self.target_y:.3f}) m')
        if self.photo_robot_pos:
            self.get_logger().info(
                f'  Robot at photo : '
                f'({self.photo_robot_pos[0]:.3f},{self.photo_robot_pos[1]:.3f}) m')
        else:
            self.get_logger().info('  Robot at photo : unknown (no detection)')
        if self.cube_world_pos:
            self.get_logger().info(
                f'  Cube world pos : '
                f'({self.cube_world_pos[0]:.3f},{self.cube_world_pos[1]:.3f}) m')
        else:
            self.get_logger().info('  Cube world pos : not found')
        self.get_logger().info('=' * 55)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main(args=None):
    if len(sys.argv) < 3:
        print('Usage:   python3 autonomous_nav.py <target_x> <target_y> [<map_yaml>]')
        print('Example: python3 autonomous_nav.py 2.5 0.3 ~/Desktop/demo_map.yaml')
        sys.exit(1)

    tx       = float(sys.argv[1])
    ty       = float(sys.argv[2])
    map_yaml = sys.argv[3] if len(sys.argv) > 3 else '~/Desktop/demo_map.yaml'

    rclpy.init(args=args)
    node = AutonomousNav(tx, ty, map_yaml)
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
