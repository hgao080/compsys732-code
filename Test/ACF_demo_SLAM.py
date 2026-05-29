import rclpy, math, cv2, os, time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, CompressedImage
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# ── Robot Config ──────────────────────────────────────────────────────────────
NAMESPACE        = 'T24'  # ← change to your robot namespace
FORWARD_SPEED    = 0.15   # m/s
TURN_SPEED       = 0.6    # rad/s
AVOID_DISTANCE   = 0.35   # metres — obstacle too close
FRONT_ARC_DEG    = 35     # degrees either side of forward
FRONT_BEARING_DEG = 90    # degrees — forward direction in scan frame

# ── Wall Follow Config ────────────────────────────────────────────────────────
WALL_LOST_THRESHOLD    = 0.35  # metres — right wall distance to declare wall lost
WALL_LOST_SPEED        = 0.06  # m/s — forward speed when reacquiring right wall
WALL_LOST_TURN         = 0.48  # rad/s — turn speed when reacquiring right wall

# ── Goal Config ───────────────────────────────────────────────────────────────
CAPTURE_TICKS     = 50    # ticks to hold still during capture (5 s at 10 Hz)
SWEEP_TOTAL_TICKS = 110   # ticks for full 360° sweep (0.6 rad/s × 110 × 0.1 s ≈ 2π)

# ── Centre-on-Cube Config ────────────────────────────────────────────────────
CENTRE_TOLERANCE_PX     = 15   # pixels either side of centre = "aligned"
CENTRE_SPIN_KP          = 0.3  # proportional gain for spin-to-centre
LOST_CUBE_TIMEOUT_TICKS  = 50  # consecutive ticks without cube before reverting to WALL_FOLLOW
CENTRE_TIMEOUT_TICKS     = 100 # ticks in CENTRE_ON_CUBE before forcing transition to CAPTURE

# ── Red Cube HSV Thresholds ───────────────────────────────────────────────────
RED_LOW1  = np.array([0,   100,  100])
RED_HIGH1 = np.array([8,   255,  255])
RED_LOW2  = np.array([177, 100,  100])
RED_HIGH2 = np.array([180, 255,  255])
MIN_PIXELS_SWEEP_FLAG = 2000  # glimpsed cube, arm sweep at next obstacle
MIN_PIXELS_CENTRE     = 12000 # enter CENTRE_ON_CUBE from WALL_FOLLOW

# ── Return Config ─────────────────────────────────────────────────────────────
ORIGIN_THRESHOLD   = 0.1  # metres — close enough to count as home
HEADING_KP         = 0.8  # proportional gain for bearing correction
RETURN_NEAR_ORIGIN = 0.5  # metres — switch from wall-follow to bearing control

# ── SLAM Config ───────────────────────────────────────────────────────────────
MAP_FRAME       = 'map'                    # SLAM global frame
BASE_FRAME      = f'{NAMESPACE}/base_link' # robot body frame

# ── States ────────────────────────────────────────────────────────────────────
WALL_FOLLOW    = 'WALL_FOLLOW'
SWEEP          = 'SWEEP'
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

        # Subscriptions — Odometry subscription removed; pose comes from TF2
        self.scan_sub = self.create_subscription(
            LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback, 10)

        self.cam_sub = self.create_subscription(
            CompressedImage,
            f'{NAMESPACE}/oakd/rgb/image_raw/compressed',
            self.image_callback, 10)

        # TF2 — reads SLAM-corrected map→base_link transform each tick
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # LiDAR state
        self.nearest_front = float('inf')
        self.nearest_right = float('inf')

        # Pose state — updated each tick from TF2 instead of odometry
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0

        # Camera / detection state
        self.bridge      = CvBridge()
        self.cube_cx     = None
        self.image_width = None
        self.latest_img  = None

        # Capture state
        self.capture_ticks   = 0
        self.lost_cube_ticks = 0
        self.centre_ticks    = 0
        self.red_pixels      = 0

        # Sweep state
        self.sweep_flag  = False
        self.sweep_ticks = 0

        # State machine
        self.state = WALL_FOLLOW

        # Mission tracking
        self.start_time      = time.time()
        self.cube_world_pos  = None
        self.photo_robot_pos = None
        self.mission_logged  = False

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('SearchAndNavigate started — state: WALL_FOLLOW')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def dist_to_origin(self):
        return math.hypot(self.current_x, self.current_y)

    def bearing_to_origin(self):
        return math.atan2(-self.current_y, -self.current_x)

    def yaw_error_to_origin(self):
        err = self.bearing_to_origin() - self.current_yaw
        return math.atan2(math.sin(err), math.cos(err))

    # ── SLAM Pose Update ──────────────────────────────────────────────────────

    def update_pose_from_tf(self):
        """Read SLAM-corrected robot pose from the map → base_link TF transform."""
        try:
            t = self.tf_buffer.lookup_transform(
                MAP_FRAME, BASE_FRAME, rclpy.time.Time())
            self.current_x = t.transform.translation.x
            self.current_y = t.transform.translation.y
            q = t.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        except (LookupException, ConnectivityException, ExtrapolationException):
            # SLAM not yet initialised or transform not yet available — keep last known pose
            pass

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def scan_callback(self, msg):
        inc    = msg.angle_increment
        arc_r  = math.radians(FRONT_ARC_DEG)

        front_i = int(round(math.radians(FRONT_BEARING_DEG) / inc))
        half_a  = int(round(arc_r / inc))
        n = len(msg.ranges)

        def arc_min(lo, hi):
            indices = [(lo + d) % n for d in range(hi - lo + 1)]
            vals = [msg.ranges[i] for i in indices
                    if msg.range_min < msg.ranges[i] < msg.range_max]
            return min(vals) if vals else float('inf')

        self.nearest_front = arc_min(front_i - half_a, front_i + half_a)
        self.nearest_right = arc_min(0, front_i - half_a)

    def image_callback(self, msg):
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

        overlay = img.copy()
        overlay[mask > 0] = [0, 0, 255]
        cv2.putText(overlay, f'Red pixels: {pixels}  state: {self.state}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if pixels >= MIN_PIXELS_CENTRE:
            cv2.putText(overlay, 'DETECTED', (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow('Detection', overlay)
        cv2.waitKey(1)

        if self.state == WALL_FOLLOW and pixels >= MIN_PIXELS_SWEEP_FLAG:
            self.sweep_flag = True

        if self.state == WALL_FOLLOW and pixels >= MIN_PIXELS_CENTRE:
            self.state = CENTRE_ON_CUBE
            self.centre_ticks = 0
            self.get_logger().info(
                f'Red cube detected ({pixels} px)! Switching → CENTRE_ON_CUBE')

        if self.state == CENTRE_ON_CUBE:
            if pixels < MIN_PIXELS_CENTRE:
                self.lost_cube_ticks += 1
                if self.lost_cube_ticks >= LOST_CUBE_TIMEOUT_TICKS:
                    self.lost_cube_ticks = 0
                    self.centre_ticks    = 0
                    self.state = WALL_FOLLOW
                    self.get_logger().warn('Lost cube during centering — resuming WALL_FOLLOW')
            else:
                self.lost_cube_ticks = 0

    # ── Control Loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        self.update_pose_from_tf()  # refresh pose from SLAM each tick

        if self.state == WALL_FOLLOW:
            self.do_wall_follow()
        elif self.state == SWEEP:
            self.do_sweep()
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

    def do_wall_follow(self):
        msg = Twist()

        if self.nearest_front < AVOID_DISTANCE:
            if self.sweep_flag:
                self.sweep_ticks = 0
                self.state = SWEEP
                self.get_logger().info('Front obstacle + sweep flag — switching → SWEEP')
                return
            msg.linear.x  = 0.0
            msg.angular.z = TURN_SPEED
            self.get_logger().warn(
                f'Front wall ({self.nearest_front:.2f} m) — turning LEFT')

        elif self.nearest_right > WALL_LOST_THRESHOLD:
            msg.linear.x  = WALL_LOST_SPEED
            msg.angular.z = -WALL_LOST_TURN
            self.get_logger().info(
                f'Right wall lost ({self.nearest_right:.2f} m) — curving RIGHT')

        else:
            msg.linear.x  = FORWARD_SPEED
            msg.angular.z = 0.0
            self.get_logger().info(
                f'Wall follow | right={self.nearest_right:.2f} m '
                f'front={self.nearest_front:.2f} m '
                f'pos=({self.current_x:.2f}, {self.current_y:.2f})')

        self.publisher.publish(msg)

    def do_sweep(self):
        if self.red_pixels >= MIN_PIXELS_CENTRE:
            self.centre_ticks    = 0
            self.lost_cube_ticks = 0
            self.state = CENTRE_ON_CUBE
            self.get_logger().info(
                f'Sweep found cube ({self.red_pixels} px) — switching → CENTRE_ON_CUBE')
            return

        self.sweep_ticks += 1
        if self.sweep_ticks >= SWEEP_TOTAL_TICKS:
            self.sweep_flag  = False
            self.sweep_ticks = 0
            self.state = WALL_FOLLOW
            self.get_logger().info('Sweep complete, no cube — resuming WALL_FOLLOW')
            return

        msg = Twist()
        msg.angular.z = TURN_SPEED
        self.publisher.publish(msg)
        self.get_logger().info(
            f'Sweep tick {self.sweep_ticks}/{SWEEP_TOTAL_TICKS} px={self.red_pixels}')

    def do_centre_on_cube(self):
        if self.cube_cx is None or self.image_width is None:
            return

        error = self.cube_cx - self.image_width / 2

        self.centre_ticks += 1
        timed_out = self.centre_ticks >= CENTRE_TIMEOUT_TICKS
        if abs(error) <= CENTRE_TOLERANCE_PX or timed_out:
            self.capture_ticks = 0
            self.centre_ticks  = 0
            self.state = CAPTURE
            reason = 'timeout forced' if timed_out else f'error={error:.0f} px'
            self.get_logger().info(f'Cube centred ({reason}) — switching → CAPTURE')
            return

        msg = Twist()
        msg.angular.z = -CENTRE_SPIN_KP * (error / (self.image_width / 2))
        self.publisher.publish(msg)
        self.get_logger().info(
            f'Centering: cube_cx={self.cube_cx} error={error:.0f} px '
            f'spin={msg.angular.z:.2f} rad/s')

    def do_capture(self):
        self.stop()
        self.capture_ticks += 1

        if self.capture_ticks == 1:
            self.photo_robot_pos = (self.current_x, self.current_y)
            self.get_logger().info(
                f'[PHOTO] Robot map pos at capture: '
                f'({self.current_x:.3f}, {self.current_y:.3f}) m')
            if self.latest_img is not None:
                snap_path = os.path.expanduser('~/detection_demo.png')
                cv2.imwrite(snap_path, self.latest_img)
                self.get_logger().info(f'Photo saved → {snap_path}')

        self.get_logger().info(
            f'CAPTURE tick {self.capture_ticks}/{CAPTURE_TICKS} '
            f'pos=({self.current_x:.2f}, {self.current_y:.2f})')

        if self.capture_ticks >= CAPTURE_TICKS:
            self.log_cube_world_position()
            self.sweep_flag = False
            self.state = RETURN_ORIGIN
            self.get_logger().info(
                f'Capture complete. Starting return from '
                f'({self.current_x:.2f}, {self.current_y:.2f})')

    def do_return_origin(self):
        dist = self.dist_to_origin()

        if dist < ORIGIN_THRESHOLD:
            self.state = ORIGIN_REACHED
            self.get_logger().info('ORIGIN REACHED — mission complete')
            self.stop()
            return

        if dist > RETURN_NEAR_ORIGIN:
            self.do_wall_follow()
            return

        msg = Twist()
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

    def log_cube_world_position(self):
        if self.nearest_front == float('inf'):
            self.get_logger().warn('Cannot estimate cube position: no LiDAR range')
            return

        cube_x = self.current_x + self.nearest_front * math.cos(self.current_yaw)
        cube_y = self.current_y + self.nearest_front * math.sin(self.current_yaw)
        self.cube_world_pos = (cube_x, cube_y)

        self.get_logger().info(
            f'[CUBE POSITION] world=({cube_x:.3f}, {cube_y:.3f}) m  '
            f'| robot=({self.current_x:.3f}, {self.current_y:.3f})  '
            f'| range={self.nearest_front:.3f} m'
        )

    def log_mission_summary(self):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(elapsed, 60)
        self.get_logger().info('=' * 55)
        self.get_logger().info('[MISSION SUMMARY]')
        self.get_logger().info(
            f'  Duration       : {int(mins)}m {secs:.1f}s ({elapsed:.1f} s total)')
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
