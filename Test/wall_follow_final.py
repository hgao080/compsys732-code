import rclpy, math, cv2, os, time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# ── Robot Config ──────────────────────────────────────────────────────────────
NAMESPACE        = 'T12'  # ← change to your robot namespace
FORWARD_SPEED    = 0.15   # m/s
TURN_SPEED       = 0.6    # rad/s
AVOID_DISTANCE   = 0.35   # metres — obstacle too close
FRONT_ARC_DEG    = 35     # degrees either side of forward
GAP_CLEAR_ARC_DEG = 8    # degrees either side of forward — narrow "is straight ahead clear?" cone for gaps
CUBE_RANGE_ARC_DEG = 10  # degrees either side of forward for cube range estimate
FRONT_BEARING_DEG = 90    # degrees — forward direction in scan frame

# ── Wall Follow Config ────────────────────────────────────────────────────────
WALL_LOST_THRESHOLD    = 0.4  # metres — right wall distance to declare wall lost
WALL_LOST_SPEED        = 0.05  # m/s — forward speed when reacquiring right wall
WALL_LOST_TURN         = 0.7  # rad/s — turn speed when reacquiring right wall
WALL_TARGET_DIST       = 0.29  # metres — desired distance to right wall
WALL_KP                = 1.2   # proportional gain for right-wall distance control

# ── Find-Wall Config ──────────────────────────────────────────────────────────
FIND_WALL_WAIT_TICKS = 10   # ticks to hold still on startup (1 s at 10 Hz)
FIND_WALL_SPIN_TICKS = 52   # ticks for 180° spin (0.6 rad/s × 52 × 0.1 s ≈ π)

# ── Peek Config ────────────────────────────────────────────────────────────────
PEEK_TICKS        = 13    # ticks to peek right at each front obstacle (~45° at 0.6 rad/s)

# ── Gap Config ────────────────────────────────────────────────────────────────
WALL_LOST_GAP_TICKS = 15  # ticks of continuous wall-loss before switching to gap-crossing behaviour
GAP_CROSS_SPEED     = 0.08  # m/s — forward speed while threading a gap (slower than FORWARD_SPEED for control)

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
MIN_PIXELS_CENTRE     = 8500  # enter CENTRE_ON_CUBE from WALL_FOLLOW / PEEK

# ── Return Config ─────────────────────────────────────────────────────────────
ORIGIN_THRESHOLD = 0.1    # metres — close enough to count as home
HEADING_KP             = 0.8   # proportional gain for bearing correction
RETURN_NEAR_ORIGIN     = 0.5   # metres — switch from wall-follow to bearing control

# ── States ────────────────────────────────────────────────────────────────────
FIND_WALL      = 'FIND_WALL'
WALL_FOLLOW    = 'WALL_FOLLOW'
PEEK           = 'PEEK'
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
        self.nearest_front = float('inf')
        self.nearest_right = float('inf')
        self.front_narrow  = float('inf')
        self.nearest_cube_front = float('inf')

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
        self.peek_ticks  = 0
        self.peek_done   = False   # True after peeking at the current front obstacle
        self.cube_found  = False   # True once CAPTURE begins; suppresses peek on return

        # Gap state
        self.gap_ticks   = 0      # consecutive ticks with right wall lost

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
        gap_half_a = int(round(math.radians(GAP_CLEAR_ARC_DEG) / inc))
        self.front_narrow = arc_min(front_i - gap_half_a, front_i + gap_half_a)
        cube_half_a = int(round(math.radians(CUBE_RANGE_ARC_DEG) / inc))
        self.nearest_cube_front = arc_min(front_i - cube_half_a, front_i + cube_half_a)

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

        # Directly centre on cube if enough pixels visible during wall follow
        if self.state == WALL_FOLLOW and pixels >= MIN_PIXELS_CENTRE:
            self.centre_ticks = 0
            self.lost_cube_ticks = 0
            self.state = CENTRE_ON_CUBE
            self.get_logger().info(
                f'Red cube detected ({pixels} px) during wall-follow — switching → CENTRE_ON_CUBE')

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
        elif self.state == PEEK:
            self.do_peek()
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
        """Follow the right wall through the C-shaped corridor."""
        msg = Twist()

        if self.nearest_front < AVOID_DISTANCE:
            if not self.cube_found and not self.peek_done:
                self.peek_ticks = 0
                self.peek_done  = True
                self.state = PEEK
                self.stop()   # flush forward velocity before rotation starts
                self.get_logger().info('Front obstacle — peeking right before turning')
                return
            # Cube already found (returning) or already peeked; turn left
            msg.linear.x  = 0.0
            msg.angular.z = TURN_SPEED
            self.get_logger().warn(f'Front wall ({self.nearest_front:.2f} m) — turning LEFT')

        elif self.nearest_right > WALL_LOST_THRESHOLD:
            self.gap_ticks += 1
            if self.gap_ticks > WALL_LOST_GAP_TICKS:
                # Wide gap (doorway) — thread it slowly, straight ahead with a gentle
                # rightward bias to reacquire the right wall on the far side
                if self.front_narrow < AVOID_DISTANCE:
                    # Path dead ahead is genuinely blocked — turn left
                    msg.linear.x  = 0.0
                    msg.angular.z = TURN_SPEED
                    self.get_logger().warn(
                        f'Gap blocked ahead ({self.front_narrow:.2f} m) — turning LEFT')
                else:
                    msg.linear.x  = GAP_CROSS_SPEED
                    msg.angular.z = -0.2
                    self.get_logger().info(
                        f'Gap crossing ({self.gap_ticks} ticks) — forward at {GAP_CROSS_SPEED} m/s')
            else:
                # Short wall-loss (right-turn corner) — tight rightward arc
                msg.linear.x  = WALL_LOST_SPEED
                msg.angular.z = -WALL_LOST_TURN
                self.get_logger().info(
                    f'Right wall lost ({self.nearest_right:.2f} m) — curving RIGHT')

        else:
            self.gap_ticks = 0
            self.peek_done = False   # new obstacle ahead will get its own peek
            error = self.nearest_right - WALL_TARGET_DIST
            msg.linear.x  = FORWARD_SPEED
            msg.angular.z = -WALL_KP * error
            self.get_logger().info(
                f'Wall follow | right={self.nearest_right:.2f} m '
                f'err={error:.2f} m spin={msg.angular.z:.2f} '
                f'front={self.nearest_front:.2f} m '
                f'pos=({self.current_x:.2f}, {self.current_y:.2f})')

        self.publisher.publish(msg)

    def do_peek(self):
        """Rotate right ~45° in place at a corner to check if the cube is just around it."""
        if self.red_pixels >= MIN_PIXELS_CENTRE:
            self.centre_ticks    = 0
            self.lost_cube_ticks = 0
            self.state = CENTRE_ON_CUBE
            self.get_logger().info(
                f'Peek found cube ({self.red_pixels} px) — switching → CENTRE_ON_CUBE')
            return

        self.peek_ticks += 1
        if self.peek_ticks >= PEEK_TICKS:
            self.state = WALL_FOLLOW  # will turn left (peek_done is True)
            self.get_logger().info('Peek complete, no cube — resuming WALL_FOLLOW (turn left)')
            return

        msg = Twist()
        msg.linear.x  = 0.0           # pure rotation — no forward motion
        msg.angular.z = -TURN_SPEED   # peek right
        self.publisher.publish(msg)
        self.get_logger().info(
            f'PEEK tick {self.peek_ticks}/{PEEK_TICKS} px={self.red_pixels}')

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
            self.cube_found = True
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