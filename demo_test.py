import rclpy, math, cv2, os, time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# ── Robot Config ──────────────────────────────────────────────────────────────
NAMESPACE        = 'T24'  # ← change to your robot namespace
FORWARD_SPEED    = 0.15   # m/s
TURN_SPEED       = 0.6    # rad/s
AVOID_DISTANCE   = 0.35   # metres — obstacle too close
FRONT_ARC_DEG    = 35     # degrees either side of forward
CUBE_RANGE_ARC_DEG = 10  # degrees either side of forward for cube range estimate
FRONT_BEARING_DEG = 90    # degrees — forward direction in scan frame

# ── Wall Follow Config ────────────────────────────────────────────────────────
WALL_LOST_THRESHOLD    = 0.38  # metres — right wall distance to declare wall lost
WALL_LOST_SPEED        = 0.09  # m/s — forward speed when reacquiring right wall
WALL_LOST_TURN         = 0.4   # rad/s — turn speed when reacquiring right wall (was 0.8)
WALL_TARGET_DIST       = 0.28  # metres — desired distance to right wall
WALL_KD                = 1.2   # proportional gain — perpendicular distance error
WALL_KH                = 1.0   # proportional gain — wall heading angle error (new)
WALL_PERP_ANGLE_DEG    = 0     # scan-frame angle for perpendicular-right reading
WALL_DIAG_ANGLE_DEG    = 30    # scan-frame angle for ahead-right diagonal reading (30 reduces corner look-ahead vs 45)
WALL_AVG_HALF_DEG      = 3     # half-arc width used to average each reading

# ── Find-Wall Config ──────────────────────────────────────────────────────────
FIND_WALL_WAIT_TICKS = 10   # ticks to hold still on startup (1 s at 10 Hz)
FIND_WALL_SPIN_TICKS = 52   # ticks for 180° spin (0.6 rad/s × 52 × 0.1 s ≈ π)

# ── Peek Config ───────────────────────────────────────────────────────────────
PEEK_TURN_SPEED    = 0.5   # rad/s — rotation speed during peek
PEEK_YAW_TOLERANCE = 0.08  # rad — close enough to target yaw (≈4.6°)
PEEK_CHECK_TICKS   = 8     # ticks to hold still while camera settles (~0.8 s)

# ── Centre-on-Cube Config ────────────────────────────────────────────────────
CENTRE_TOLERANCE_PX     = 15   # pixels either side of centre = "aligned"
CENTRE_SPIN_KP          = 0.3  # proportional gain for spin-to-centre
LOST_CUBE_TIMEOUT_TICKS  = 50  # consecutive ticks without cube before reverting to WALL_FOLLOW
CENTRE_TIMEOUT_TICKS     = 100  # ticks in CENTRE_ON_CUBE before forcing transition to CAPTURE

CAPTURE_TICKS     = 50    # ticks to hold still during capture (2 s at 10 Hz)

# ── Red Cube HSV Thresholds ───────────────────────────────────────────────────
RED_LOW1  = np.array([0,   95,  95])
RED_HIGH1 = np.array([8,  255, 255])
RED_LOW2  = np.array([177, 95,  95])
RED_HIGH2 = np.array([180, 255, 255])
MIN_PIXELS_SWEEP_FLAG = 2000  # glimpsed cube, arm sweep at next obstacle
MIN_PIXELS_CENTRE     = 8500 # enter CENTRE_ON_CUBE from WALL_FOLLOW

# ── Return Config ─────────────────────────────────────────────────────────────
ORIGIN_THRESHOLD = 0.1    # metres — close enough to count as home
HEADING_KP             = 0.8   # proportional gain for bearing correction
RETURN_NEAR_ORIGIN     = 0.5   # metres — switch from wall-follow to bearing control

# ── States ────────────────────────────────────────────────────────────────────
FIND_WALL      = 'FIND_WALL'
WALL_FOLLOW    = 'WALL_FOLLOW'
PEEK_RIGHT     = 'PEEK_RIGHT'
CENTRE_ON_CUBE = 'CENTRE_ON_CUBE'
CAPTURE        = 'CAPTURE'
RETURN_ORIGIN  = 'RETURN_ORIGIN'
ORIGIN_REACHED = 'ORIGIN_REACHED'


class SearchAndNavigate(Node):

    def __init__(self):
        super().__init__('search_and_navigate')

        # Publishers
        self.publisher = self.create_publisher(
            Twist, f'{NAMESPACE}/cmd_vel', 10)

        # Subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback, 10)

        self.cam_sub = self.create_subscription(
            CompressedImage,
            f'{NAMESPACE}/oakd/rgb/image_raw/compressed',
            self.image_callback, 10)

        self.create_subscription(
            Odometry, f'{NAMESPACE}/odom',
            self.odom_callback, 10)

        # LiDAR state
        self.nearest_front      = float('inf')
        self.nearest_right      = float('inf')
        self.nearest_cube_front = float('inf')
        self.wall_perp          = float('inf')   # range at WALL_PERP_ANGLE_DEG
        self.wall_diag          = float('inf')   # range at WALL_DIAG_ANGLE_DEG

        # Odometry state
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0

        # Camera / detection state
        self.bridge        = CvBridge()
        self.cube_cx       = None
        self.image_width   = None
        self.latest_img    = None

        # Find-wall state
        self.find_wall_wait_ticks = 0   # ticks elapsed during startup hold
        self.find_wall_spin_ticks = 0   # ticks elapsed during initial 180° spin

        # Capture state
        self.capture_ticks    = 0
        self.lost_cube_ticks  = 0   # consecutive frames cube not seen during centering
        self.centre_ticks     = 0   # ticks spent in CENTRE_ON_CUBE
        self.red_pixels       = 0

        # Peek state
        self.peek_yaw_start   = 0.0            # robot yaw when peek began
        self.peek_phase       = 'rotating_right'  # 'rotating_right'|'checking'|'rotating_back'
        self.peek_check_ticks = 0              # ticks elapsed in 'checking' phase
        self.peek_done        = False          # True after peek at current obstacle; cleared when obstacle clears
        self.peek_enabled     = True           # disabled once RETURN_ORIGIN begins

        # State machine
        self.state = FIND_WALL

        # Mission tracking
        self.start_time       = time.time()
        self.cube_world_pos   = None   # set in log_cube_world_position
        self.photo_robot_pos  = None   # set in do_capture on first tick
        self.mission_logged   = False

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('SearchAndNavigate started — state: FIND_WALL')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def dist_to_origin(self):
        return math.hypot(self.current_x, self.current_y)

    def bearing_to_origin(self):
        """Angle (radians) from current position toward (0, 0)."""
        return math.atan2(-self.current_y, -self.current_x)

    def yaw_error_to_origin(self):
        """Signed angular difference between current yaw and bearing to origin."""
        err = self.bearing_to_origin() - self.current_yaw
        # Normalise to [-π, π]
        return math.atan2(math.sin(err), math.cos(err))

    def _wall_errors(self):
        """Dual-P wall errors from two right-side LiDAR readings.

        Returns (dist_err, heading_err):
          dist_err    = d_perp − WALL_TARGET_DIST  (+ve → too far from wall)
          heading_err = ψ in radians               (+ve → converging, turn left)
        Returns (inf, 0.0) when wall is not visible.

        Method: compute wall direction vector from two body-frame points,
        derive angle ψ (= 0 when robot is parallel to wall).
        """
        rp = self.wall_perp   # range ≈ perpendicular to wall
        rd = self.wall_diag   # range at WALL_DIAG_ANGLE_DEG (ahead-right)

        # rp alone determines wall-lost: if wall is gone, no useful control possible
        if rp >= WALL_LOST_THRESHOLD:
            return float('inf'), 0.0

        # Body-frame angles from forward (x-axis): 0 = forward, −90° = directly right
        phi_p = math.radians(WALL_PERP_ANGLE_DEG - FRONT_BEARING_DEG)  # = −90°
        phi_d = math.radians(WALL_DIAG_ANGLE_DEG - FRONT_BEARING_DEG)  # = −60°

        # Perpendicular distance (exact when phi_p = −90°, rp = d_perp)
        d_perp = abs(rp * math.sin(phi_p))

        # rd unavailable or sees past a corner → distance-only control (psi = 0)
        if rd == float('inf'):
            return d_perp - WALL_TARGET_DIST, 0.0

        rd_expected = d_perp / abs(math.sin(phi_d))
        if rd > rd_expected * 1.5:
            # Diagonal is seeing past a corner — suppress heading correction
            return d_perp - WALL_TARGET_DIST, 0.0

        # Wall direction vector: two wall points in body frame
        ab_x = rd * math.cos(phi_d) - rp * math.cos(phi_p)
        ab_y = rd * math.sin(phi_d) - rp * math.sin(phi_p)
        psi  = math.atan2(ab_y, ab_x)   # 0 = parallel; +ve = converging to wall

        return d_perp - WALL_TARGET_DIST, psi

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def scan_callback(self, msg):
        inc    = msg.angle_increment
        arc_r  = math.radians(FRONT_ARC_DEG)

        front_i = int(round(math.radians(FRONT_BEARING_DEG) / inc))
        half_a  = int(round(arc_r  / inc))
        n = len(msg.ranges)

        def arc_min(lo, hi):
            indices = [(lo + d) % n for d in range(hi - lo + 1)]
            vals = [msg.ranges[i] for i in indices
                    if msg.range_min < msg.ranges[i] < msg.range_max]
            return min(vals) if vals else float('inf')

        self.nearest_front = arc_min(front_i - half_a, front_i + half_a)
        self.nearest_right = arc_min(0, front_i - half_a) 
        cube_half_a = int(round(math.radians(CUBE_RANGE_ARC_DEG) / inc))
        self.nearest_cube_front = arc_min(front_i - cube_half_a, front_i + cube_half_a)

        # Two-point right-wall readings for parallel P controller
        perp_i = int(round(math.radians(WALL_PERP_ANGLE_DEG) / inc))
        diag_i = int(round(math.radians(WALL_DIAG_ANGLE_DEG) / inc))
        avg_hw = max(1, int(round(math.radians(WALL_AVG_HALF_DEG) / inc)))
        self.wall_perp = arc_min(max(0, perp_i - avg_hw), perp_i + avg_hw)
        self.wall_diag = arc_min(diag_i - avg_hw, diag_i + avg_hw)

    def image_callback(self, msg):
        """Detect red cube using HSV masking."""
        img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        self.image_width = img.shape[1]
        self.latest_img  = img

        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
        )
        pixels = cv2.countNonZero(mask)
        self.red_pixels = pixels

        M = cv2.moments(mask)
        if M['m00'] > 0:
            self.cube_cx = int(M['m10'] / M['m00'])
        else:
            self.cube_cx = None

        # Debug overlay
        overlay = img.copy()
        overlay[mask > 0] = [0, 0, 255]
        cv2.putText(overlay, f'Red pixels: {pixels}  state: {self.state}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if pixels >= MIN_PIXELS_CENTRE:
            cv2.putText(overlay, 'DETECTED', (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow('Detection', overlay)
        cv2.waitKey(1)

        if self.state in (WALL_FOLLOW, PEEK_RIGHT) and pixels >= MIN_PIXELS_CENTRE:
            self.state = CENTRE_ON_CUBE
            self.centre_ticks = 0
            self.get_logger().info(
                f'Red cube detected ({pixels} px)! Switching → CENTRE_ON_CUBE')

        # Only revert to wall follow if cube is gone for 10 consecutive frames (~1 s)
        if self.state == CENTRE_ON_CUBE:
            if pixels < MIN_PIXELS_CENTRE:
                self.lost_cube_ticks += 1
                if self.lost_cube_ticks >= LOST_CUBE_TIMEOUT_TICKS:
                    self.lost_cube_ticks = 0
                    self.centre_ticks = 0
                    self.state = WALL_FOLLOW
                    self.get_logger().warn('Lost cube for 1 s during centering — resuming WALL_FOLLOW')
            else:
                self.lost_cube_ticks = 0

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    # ── Control Loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        if self.state == FIND_WALL:
            self.do_find_wall()
        elif self.state == WALL_FOLLOW:
            self.do_wall_follow()
        elif self.state == PEEK_RIGHT:
            self.do_peek_right()
        elif self.state == CENTRE_ON_CUBE:
            self.do_centre_on_cube()
        elif self.state == CAPTURE:
            self.do_capture()
        elif self.state == RETURN_ORIGIN:
            self.do_return_origin()
        elif self.state == ORIGIN_REACHED:
            self.stop()
            if not self.mission_logged:
                self.mission_logged = True
                self.log_mission_summary()

    # ── State Behaviours ──────────────────────────────────────────────────────

    def do_find_wall(self):
        """Spin 180° then drive straight until a front obstacle is detected, then enter WALL_FOLLOW."""
        msg = Twist()

        if self.find_wall_wait_ticks < FIND_WALL_WAIT_TICKS:
            self.find_wall_wait_ticks += 1
            self.get_logger().info(
                f'FIND_WALL init hold {self.find_wall_wait_ticks}/{FIND_WALL_WAIT_TICKS}')
            self.publisher.publish(msg)  # zero twist — stopped
            return

        if self.find_wall_spin_ticks < FIND_WALL_SPIN_TICKS:
            msg.angular.z = TURN_SPEED
            self.find_wall_spin_ticks += 1
            self.get_logger().info(
                f'FIND_WALL spin {self.find_wall_spin_ticks}/{FIND_WALL_SPIN_TICKS}')
        elif self.nearest_front < AVOID_DISTANCE:
            self.state = WALL_FOLLOW
            self.get_logger().info(
                f'FIND_WALL: wall found ({self.nearest_front:.2f} m) — switching → WALL_FOLLOW')
            return
        else:
            msg.linear.x = FORWARD_SPEED
            self.get_logger().info(
                f'FIND_WALL driving | front={self.nearest_front:.2f} m')

        self.publisher.publish(msg)

    def do_wall_follow(self):
        """Follow right wall with dual-P parallel controller.

        Two independent P terms:
          WALL_KD × dist_err   — keeps perpendicular distance on target
          WALL_KH × heading_err — keeps robot parallel to wall (ψ = 0)
        """
        msg = Twist()

        if self.nearest_front < AVOID_DISTANCE:
            if self.peek_enabled and not self.peek_done:
                # First encounter with this obstacle — peek right before turning left
                self.peek_yaw_start   = self.current_yaw
                self.peek_phase       = 'rotating_right'
                self.peek_check_ticks = 0
                self.state = PEEK_RIGHT
                self.get_logger().info(
                    f'Front obstacle — peeking right from yaw={math.degrees(self.current_yaw):.1f}°')
                return
            # Peek already done (or disabled in return phase) → turn left as normal
            msg.linear.x  = 0.0
            msg.angular.z = TURN_SPEED
            self.get_logger().warn(
                f'Front wall ({self.nearest_front:.2f} m) — turning LEFT')

        else:
            self.peek_done = False   # obstacle cleared — ready to peek at the next one
            dist_err, heading_err = self._wall_errors()

            if dist_err == float('inf'):
                # Wall not visible — slow curve right to reacquire
                msg.linear.x  = WALL_LOST_SPEED
                msg.angular.z = -WALL_LOST_TURN
                self.get_logger().info(
                    f'Right wall lost (perp={self.wall_perp:.2f} m) — curving RIGHT')
            else:
                # Dual-P: heading correction dominates alignment; distance keeps gap
                msg.linear.x  = FORWARD_SPEED
                msg.angular.z = WALL_KH * heading_err - WALL_KD * dist_err
                self.get_logger().info(
                    f'Wall follow | perp={self.wall_perp:.2f} m '
                    f'dist_err={dist_err:+.3f} m '
                    f'ψ={math.degrees(heading_err):+.1f}° '
                    f'cmd_z={msg.angular.z:+.2f} '
                    f'pos=({self.current_x:.2f},{self.current_y:.2f})')

        self.publisher.publish(msg)

    def do_peek_right(self):
        """90° right peek at a front obstacle to check for red cube.

        Phases:
          rotating_right : rotate CW until yaw has decreased by π/2 from peek_yaw_start
          checking       : hold still for PEEK_CHECK_TICKS; evaluate red_pixels
                           → cube found: CENTRE_ON_CUBE
                           → no cube:    rotating_back
          rotating_back  : rotate CCW back to peek_yaw_start → WALL_FOLLOW (peek_done=True)

        image_callback also monitors PEEK_RIGHT state and can fire CENTRE_ON_CUBE
        transition early if pixels exceed threshold while rotating.
        """
        def norm(a):
            return math.atan2(math.sin(a), math.cos(a))

        msg = Twist()

        if self.peek_phase == 'rotating_right':
            target = norm(self.peek_yaw_start - math.pi / 2)
            err    = norm(self.current_yaw - target)
            if abs(err) < PEEK_YAW_TOLERANCE:
                self.peek_phase       = 'checking'
                self.peek_check_ticks = 0
                self.get_logger().info('Peek: facing right — checking camera')
            else:
                msg.angular.z = -PEEK_TURN_SPEED   # rotate CW (right)

        elif self.peek_phase == 'checking':
            self.peek_check_ticks += 1
            self.get_logger().info(
                f'Peek check {self.peek_check_ticks}/{PEEK_CHECK_TICKS} px={self.red_pixels}')
            if self.peek_check_ticks >= PEEK_CHECK_TICKS:
                if self.red_pixels >= MIN_PIXELS_CENTRE:
                    self.centre_ticks    = 0
                    self.lost_cube_ticks = 0
                    self.state = CENTRE_ON_CUBE
                    self.get_logger().info(
                        f'Peek: cube found ({self.red_pixels} px) → CENTRE_ON_CUBE')
                    return
                self.peek_phase = 'rotating_back'
                self.get_logger().info(
                    f'Peek: no cube ({self.red_pixels} px) — rotating back')

        elif self.peek_phase == 'rotating_back':
            err = norm(self.current_yaw - self.peek_yaw_start)
            if abs(err) < PEEK_YAW_TOLERANCE:
                self.peek_done  = True
                self.peek_phase = 'rotating_right'   # reset for next obstacle
                self.state      = WALL_FOLLOW
                self.get_logger().info('Peek: back to original heading — resuming WALL_FOLLOW')
                return
            msg.angular.z = +PEEK_TURN_SPEED   # rotate CCW (left) back to start

        self.publisher.publish(msg)

    def do_centre_on_cube(self):
        """Spin in place until the red cube centroid is in the centre of the image."""
        if self.cube_cx is None or self.image_width is None:
            return  # no detection yet, wait

        error = self.cube_cx - self.image_width / 2  # positive = cube is right

        self.centre_ticks += 1
        timed_out = self.centre_ticks >= CENTRE_TIMEOUT_TICKS
        if abs(error) <= CENTRE_TOLERANCE_PX or timed_out:
            self.capture_ticks = 0
            self.centre_ticks = 0
            self.state = CAPTURE
            reason = 'timeout forced' if timed_out else f'error={error:.0f} px'
            self.get_logger().info(
                f'Cube centred ({reason}) — switching → CAPTURE')
            return

        msg = Twist()
        # Cube to the right (positive error) → turn right (negative angular.z)
        msg.angular.z = -CENTRE_SPIN_KP * (error / (self.image_width / 2))
        self.publisher.publish(msg)
        self.get_logger().info(
            f'Centering: cube_cx={self.cube_cx} error={error:.0f} px '
            f'spin={msg.angular.z:.2f} rad/s')

    def do_capture(self):
        """Stop for 2 s, save photo, log cube world position, then return."""
        self.stop()
        self.capture_ticks += 1

        if self.capture_ticks == CAPTURE_TICKS // 2:
            self.photo_robot_pos = (self.current_x, self.current_y)
            self.get_logger().info(
                f'[PHOTO] Robot odom at capture: '
                f'({self.current_x:.3f}, {self.current_y:.3f}) m')
            if self.latest_img is not None:
                snap_path = os.path.expanduser('~/detection_demo.png')
                cv2.imwrite(snap_path, self.latest_img)
                self.get_logger().info(f'Photo saved → {snap_path}')
            if self.nearest_cube_front != float('inf'):
                cube_x = self.current_x + self.nearest_cube_front * math.cos(self.current_yaw)
                cube_y = self.current_y + self.nearest_cube_front * math.sin(self.current_yaw)
                self.cube_world_pos = (cube_x, cube_y)

        self.get_logger().info(
            f'CAPTURE tick {self.capture_ticks}/{CAPTURE_TICKS} '
            f'pos=({self.current_x:.2f}, {self.current_y:.2f})')

        if self.capture_ticks >= CAPTURE_TICKS:
            self.peek_enabled = False   # no more peeking during return journey
            self.state = RETURN_ORIGIN
            self.get_logger().info(
                f'Capture complete. Starting return from '
                f'({self.current_x:.2f}, {self.current_y:.2f})')

    def do_return_origin(self):
        """Return to origin (0, 0).

        Far phase  (dist > RETURN_NEAR_ORIGIN): right-wall-follow, same
                   algorithm as search.
        Near phase (dist ≤ RETURN_NEAR_ORIGIN): proportional bearing
                   control straight to origin with obstacle avoidance.
        """
        dist = self.dist_to_origin()

        if dist < ORIGIN_THRESHOLD:
            self.state = ORIGIN_REACHED
            self.get_logger().info('ORIGIN REACHED — mission complete')
            self.stop()
            return

        msg = Twist()

        if dist > RETURN_NEAR_ORIGIN:
            self.do_wall_follow()
            return

        else:
            # ── NEAR: bearing control straight to origin ──────────
            if self.nearest_front < AVOID_DISTANCE:
                msg.linear.x  = 0.0
                msg.angular.z = TURN_SPEED
                self.get_logger().warn(
                    f'Return bearing: obstacle ({self.nearest_front:.2f} m) — turning LEFT')
            else:
                err           = self.yaw_error_to_origin()
                msg.linear.x  = FORWARD_SPEED
                msg.angular.z = HEADING_KP * err
                self.get_logger().info(
                    f'Return bearing | dist={dist:.2f} m '
                    f'bearing_err={math.degrees(err):.1f}° '
                    f'pos=({self.current_x:.2f}, {self.current_y:.2f})')

        self.publisher.publish(msg)

    def log_mission_summary(self):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(elapsed, 60)
        self.get_logger().info('=' * 55)
        self.get_logger().info('[MISSION SUMMARY]')
        self.get_logger().info(f'  Duration       : {int(mins)}m {secs:.1f}s ({elapsed:.1f} s total)')
        if self.photo_robot_pos:
            self.get_logger().info(
                f'  Robot at photo : ({self.photo_robot_pos[0]:.3f}, {self.photo_robot_pos[1]:.3f}) m')
        else:
            self.get_logger().info('  Robot at photo : unknown (no capture)')
        if self.cube_world_pos:
            self.get_logger().info(
                f'  Cube world pos : ({self.cube_world_pos[0]:.3f}, {self.cube_world_pos[1]:.3f}) m')
        else:
            self.get_logger().info('  Cube world pos : unknown (no detection)')
        self.get_logger().info('=' * 55)

    def stop(self):
        self.publisher.publish(Twist())


# ── Entry Point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SearchAndNavigate()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()