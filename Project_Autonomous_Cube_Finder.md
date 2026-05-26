# COMPSYS 732 — Autonomous Robot Project: Autonomous Cube Finder

## Overview

In Tasks 6, 7, and 8 you built three components separately: a SLAM map of the arena with odometry position tracking, a physical reactive controller that avoids obstacles while tracking position, and a camera detection node that stops the robot and logs where the cube was found. This project asks you to combine all three into a single autonomous system.

The demo takes place in a dedicated C-shaped MDF arena. The run has two sequential phases within a single 10-minute window. In Phase 1 you teleoperate the robot to build a SLAM map of the arena, locating the three target cylinders at the far end. In Phase 2 the robot must search the arena autonomously, detect the red cube, log its estimated (x, y) position, and return to the starting point.

There are no step-by-step instructions. You have all the pieces; the project is the integration. You will make design decisions, justify them, and evaluate your results in an invigilated report session two days after the demo.

---

## The Arena

The search area is a C-shaped corridor built from MDF panels (each 100 cm × 30 cm × 1 cm, standing upright on mounts). The corridor is fully enclosed; all LiDAR beams hit walls cleanly, producing a sharp, near-complete SLAM map in a single pass.

**Key physical specifications:**

- Three target cylinders (diameter 12 cm, height 25 cm) are placed at the far end of the C-shape. One has the red cube on top. You will not know which one until you drive there in Phase 1. Their positions are fixed and the same for every group.
- Two obstacle cylinders (diameter 20 cm) are placed by the instructor between Phase 1 and Phase 2. They are not present during Phase 1. Their positions are fixed for the day but not revealed to students beforehand.
- The red cube is matte red, approximately 6 × 6 × 6 cm, placed on top of one of the three target cylinders.
- The starting tape mark (L-shape) is fixed and shared across all groups. It is the SLAM origin (0, 0). The robot must face into the corridor (along the positive x-axis) at this mark.

---

## What the Robot Must Do (Phase 2)

The autonomous run (Phase 2) has four states:

### SEARCHING

The robot executes a deliberate coverage search pattern via `cmd_vel`. It continuously reads `/TXX/scan` for obstacle avoidance.

- → Obstacle detected: temporarily avoid, then resume search pattern
- → Red cube detected: transition to **REPORTING**
- → Time limit exceeded: transition to **RETURNING**

### REPORTING

The robot stops immediately. It logs the current odometry (x, y) as the reported cube position. It saves a camera snapshot image to disk (`detection_snapshot.jpg`) as visual evidence of the detection. This is the primary output of the run.

- → Immediately after logging and saving: transition to **RETURNING**

### RETURNING

The robot navigates back to the starting position (0, 0) using odometry feedback. It reads its current (x, y) from the odometry subscriber, calculates the heading to the origin, rotates to face it, and drives forward until within an acceptable distance.

### DONE

The robot stops and holds position. The node prints a final summary: detected position, return position, and total run time.

---

## The Two Phases

> **Time is shared.** Both phases run within a single 10-minute window. Time spent in Phase 1 directly reduces the time available for Phase 2. Phase 1 should take approximately 1–2 minutes.

### Phase 1 — SLAM Mapping

Before the autonomous run, you must build a SLAM map of the arena with the three target cylinders visible in it. This map is used in your report to calculate ground-truth cylinder positions for error analysis.

1. Place the robot at the tape mark in the correct orientation. Confirm it is undocked.
2. Launch SLAM.
3. Launch teleoperation.
4. Drive through the full C-shape.
5. Return to tape mark and dock.
6. Save the map (while SLAM is still running).
7. Save the posegraph:
   ```
   ros2 service call /T<ID>/slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '$HOME/Desktop/demo_map'}"
   ```
8. Take a screenshot of the Rviz/pgm map. **This is required evidence for the report.**

### Phase 2 — Autonomous Run (remaining time)

After the judge places the two obstacle cylinders, complete this sequence in order:

1. Undock.
2. Confirm robot is at the tape mark in the correct orientation.
3. Reset odometry:
   ```
   ros2 service call /T<ID>/reset_pose irobot_create_msgs/srv/ResetPose {}
   ```
4. Verify reset:
   ```
   ros2 topic echo /T<ID>/odom --field pose.pose.position --once
   ```
   Must show `x: 0.0  y: 0.0`.
5. Activate venv:
   ```
   source ~/ros2_venv/bin/activate
   ```
6. Source workspace:
   ```
   source ~/ros2_ws/install/setup.bash
   ```
7. Get the go signal from the judge.
8. Launch your node:
   ```
   ~/ros2_venv/bin/python3 -m tb4_sensor_reader.autonomous_search
   ```
9. **Do not touch the robot, keyboard, or mouse after this point.**

---

## Building Your Autonomous Search Node

### Starting point

Your `detect_and_stop.py` from Task 8 is the direct starting point. It already has a LiDAR subscriber, camera subscriber, odometry subscriber, `cmd_vel` publisher, and a basic state machine with position logging. What you need to add:

- A deliberate search pattern in the `SEARCHING` state; replace or extend the reactive logic with planned coverage.
- `REPORTING` state: stop, log (x, y), and save a camera snapshot.
- `RETURNING` state: navigate back to (0, 0) using odometry feedback.
- `DONE` state: stop, print final summary.

### Search pattern — your design choice

There is no single correct search pattern. You must choose one, implement it, and justify it in your report.

### Saving a detection snapshot

When your node transitions to `REPORTING`, save a camera image to disk. This is required evidence for assessment.

### Return to start

After detection (or time limit), navigate back to (0, 0) using odometry feedback. Example control logic:

```python
# In RETURNING state control logic:
import math

dx = 0.0 - self.current_x
dy = 0.0 - self.current_y
distance = math.sqrt(dx**2 + dy**2)

if distance < 0.5:  # within 0.5 m — transition to DONE
    self.state = 'DONE'
    self.stop()
else:
    target_angle = math.atan2(dy, dx)
    heading_err  = self._angle_diff(target_angle, self.current_yaw)
    if abs(heading_err) > 0.12:
        # Rotate to face origin
        self._publish_twist(0.0, 0.4 if heading_err > 0 else -0.4)
    else:
        # Drive toward origin
        self._publish_twist(0.15, 0.0)
```

---

## Assessment

### Live Demo: 25 points

The demo is assessed on five components, each worth 5 points. Phase 1 is not scored.

> **Collision definition:** A collision is any contact that displaces a cylinder or MDF panel or requires judge's intervention.

> **Autonomy bands:** There are only three autonomy scores: **5** (fully autonomous), **3** (semi-autonomous: node killed, brief teleop to reposition, node relaunched), or **0** (fully teleoperated).

---

## Practical Advice

### Test before the demo

- Do at least three full test runs in the lab before demo day.
- Record terminal output from each test; this is data for your report.
- Test Phase 1 timing: how long does a full SLAM pass of the arena take?
- Test your return-to-start: does the robot get back within 0.25 m?
- Verify `cv2.imwrite` saves a real file and the cube is visible in it.

### Common pitfalls

- Forgetting to reset odometry before Phase 2 — the position output will be wrong from the start.
- `NAMESPACE` not updated to your robot's ID.
- venv not activated before launching — immediate crash with NumPy error.
- Forgetting to save Phase 1 map before closing SLAM terminal.
- Not taking Rviz or pgm screenshot during Phase 1.
