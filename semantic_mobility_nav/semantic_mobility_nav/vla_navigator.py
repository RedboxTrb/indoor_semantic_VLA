#!/usr/bin/env python3
"""
VLA Navigator — natural language → semantic object navigation.

Flow:
  1. SCAN    — rotate 360°; build sweep map (angle→lidar dist); look for target
  2. ALIGN   — rotate until target bbox is centred
  3. APPROACH — drive straight; lidar triggers REACHED
  4. EXPLORE — if scan fails, pick most open unvisited direction (potential doorway)

Exploration memory:
  - visited_positions: odom (x,y) of every scan location
  - sweep_map: per-scan-position {world_angle→lidar_dist}; open sectors (>2.5m)
    are treated as potential doorways to adjacent rooms

RViz visualisation (topic /vla/markers):
  - Green spheres  = visited scan positions
  - Yellow arrows  = open-direction detections (potential doorways)
  - Blue line      = full robot path
"""

import json
import math
import os
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from std_msgs.msg import String, ColorRGBA, Header
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


# ── tuneable ────────────────────────────────────────────────────────────────
REACH_DIST_M      = 1.0
REACH_BBOX_FRAC   = 0.18   # bbox width ≥ 18% of frame → ~0.6–0.8m for a cube
REACH_CONSEC      = 2
CENTER_TOL        = 0.08
ALIGN_KP          = 0.9
MAX_ALIGN_ANG     = 0.45
APPROACH_SPEED    = 0.14
SCAN_ANGULAR      = 0.30
FRONT_HALF_DEG    = 20
OBSTACLE_DIST     = 0.45
EXPLORE_DIST      = 0.8
VISITED_RADIUS    = 0.6    # m — don't revisit within this radius
MAX_SCAN_ROUNDS   = 8
DETECTION_STALE_S = 2.5
CONTEXT_WINDOW_S  = 3.0
IMG_WIDTH         = 640

DOORWAY_DIST      = 2.5    # m — lidar reading above this = open space / doorway
DOORWAY_SECTOR    = 25     # deg width of a sector counted as "open"


PARSE_PROMPT = """\
Parse the navigation command and return ONLY valid JSON (no markdown):
{
  "target": "<object to navigate to, lowercase>",
  "context": "<spatial constraint object or null>",
  "action": "go_to"
}

Valid target labels: red cube, green cube, blue cube, table, bookshelf
Valid context labels: table, bookshelf, or null

Examples:
  "go to the blue cube on the table"   -> {"target":"blue cube","context":"table","action":"go_to"}
  "find the table"                     -> {"target":"table","context":null,"action":"go_to"}
  "navigate to red cube on bookshelf"  -> {"target":"red cube","context":"bookshelf","action":"go_to"}
  "go to green cube"                   -> {"target":"green cube","context":null,"action":"go_to"}
"""


class VLANavigator(Node):
    def __init__(self):
        super().__init__('vla_navigator')

        api_key = os.environ.get('GEMINI_API_KEY', '')
        if not api_key:
            self.get_logger().error('GEMINI_API_KEY not set!')
        from google import genai
        self._genai = genai.Client(api_key=api_key)

        # state
        self._state         = 'idle'
        self._goal          = None
        self._lock          = threading.Lock()
        self._interrupt     = threading.Event()
        self._busy          = False
        self._stopped_early = False

        # sensors
        self._latest_dets       = []
        self._det_timestamp     = 0.0
        self._context_last_seen = 0.0
        self._scan_ranges       = []
        self._scan_angle_min    = 0.0
        self._scan_angle_inc    = 0.0
        self._x   = 0.0
        self._y   = 0.0
        self._yaw = 0.0   # radians, from odom quaternion

        # exploration memory
        self._visited      = []   # list of (x, y)
        self._open_dirs    = []   # list of (world_angle_rad, x_origin, y_origin)
        self._path_poses   = []   # full path for RViz

        # marker IDs
        self._marker_id = 0

        # ROS
        self.create_subscription(String,    '/task/command',        self._cmd_cb,   10)
        self.create_subscription(String,    '/semantic/detections', self._det_cb,   10)
        self.create_subscription(Odometry,  '/odom',                self._odom_cb,  10)
        self.create_subscription(LaserScan, '/scan',                self._scan_cb,  10)

        self._vel_pub     = self.create_publisher(Twist,       '/cmd_vel',        10)
        self._status_pub  = self.create_publisher(String,      '/task/status',    10)
        self._marker_pub  = self.create_publisher(MarkerArray, '/vla/markers',    10)
        self._path_pub    = self.create_publisher(Path,        '/vla/path',       10)

        # Publish markers at 2 Hz so RViz always has fresh data
        self.create_timer(0.5, self._publish_viz)

        self.get_logger().info('VLA Navigator ready.')
        self.get_logger().info('Add /vla/markers (MarkerArray) and /vla/path (Path) in RViz2.')

    # ── sensor callbacks ────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny, cosy)
        # Record path continuously
        ps = PoseStamped()
        ps.header.frame_id = 'odom'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = msg.pose.pose
        self._path_poses.append(ps)
        if len(self._path_poses) > 2000:
            self._path_poses.pop(0)

    def _scan_cb(self, msg):
        self._scan_ranges    = list(msg.ranges)
        self._scan_angle_min = msg.angle_min
        self._scan_angle_inc = msg.angle_increment

    def _det_cb(self, msg):
        try:
            p = json.loads(msg.data)
            self._latest_dets   = p.get('detections', [])
            self._det_timestamp = p.get('timestamp', time.time())
            if self._goal and self._goal.get('context'):
                for d in self._latest_dets:
                    if d['label'] == self._goal['context'] and d.get('confidence', 0) > 0.20:
                        self._context_last_seen = self._det_timestamp
        except Exception:
            pass

    # ── command handling ─────────────────────────────────────────────────────

    STOP_WORDS = {'stop', 'halt', 'cancel', 'abort', 'freeze', 'emergency stop'}

    def _cmd_cb(self, msg):
        command = msg.data.strip()
        if not command:
            return
        if command.lower() in self.STOP_WORDS:
            self.get_logger().info('STOP received.')
            self._stopped_early = True
            self._interrupt.set()
            self._stop()
            self._state = 'idle'
            self._busy  = False
            self._pub_status('done')
            return
        if self._busy:
            self.get_logger().warn('Busy — send "stop" to interrupt.')
            return
        self._interrupt.clear()
        self._stopped_early = False
        self._busy = True
        threading.Thread(target=self._run, args=(command,), daemon=True).start()

    # ── main thread ──────────────────────────────────────────────────────────

    def _run(self, command: str):
        self.get_logger().info(f'Command: "{command}"')
        self._pub_status(f'planning:{command}')
        try:
            goal = self._parse_goal(command)
            if goal is None:
                self._pub_status('error:could not parse command')
                return
            self._goal = goal
            self._context_last_seen = 0.0
            self._visited.clear()
            self._open_dirs.clear()
            self.get_logger().info(
                f'Goal: target="{goal["target"]}"  context="{goal.get("context")}"')
            self._pub_status(f'searching:{goal["target"]}')
            self._explore_loop()
        except Exception as e:
            self.get_logger().error(f'VLA error: {e}')
            self._pub_status(f'error:{e}')
        finally:
            self._stop()
            self._busy = False
            self._state = 'idle'
            if not self._stopped_early:
                self._pub_status('done')

    # ── Gemini parser ────────────────────────────────────────────────────────

    MODEL_ORDER = ['gemini-2.5-flash', 'gemini-2.5-flash-lite',
                   'gemini-2.0-flash', 'gemini-2.0-flash-lite']

    def _parse_goal(self, command: str):
        from google.genai import types
        last_err = None
        for model in self.MODEL_ORDER:
            for attempt in range(3):
                try:
                    resp = self._genai.models.generate_content(
                        model=model,
                        contents=f'COMMAND: {command}',
                        config=types.GenerateContentConfig(
                            system_instruction=PARSE_PROMPT,
                            max_output_tokens=128,
                            temperature=0.0,
                            response_mime_type='application/json',
                        ),
                    )
                    raw = resp.text.strip()
                    s, e = raw.find('{'), raw.rfind('}')
                    goal = json.loads(raw[s:e+1])
                    assert 'target' in goal
                    return goal
                except Exception as ex:
                    last_err = ex
                    if '503' in str(ex) or 'UNAVAILABLE' in str(ex):
                        time.sleep(2); continue
                    break
        self.get_logger().error(f'Parse failed: {last_err}')
        return None

    # ── exploration loop ─────────────────────────────────────────────────────

    def _explore_loop(self):
        scan_round = 0
        while not self._interrupt.is_set() and scan_round < MAX_SCAN_ROUNDS:
            pos = (self._x, self._y)
            self.get_logger().info(
                f'[SCAN] Round {scan_round+1}/{MAX_SCAN_ROUNDS} '
                f'at ({pos[0]:.2f}, {pos[1]:.2f}) yaw={math.degrees(self._yaw):.0f}°')
            self._visited.append(pos)

            # ── 360° scan: look for target + build sweep map ───────────────
            target, sweep = self._do_scan_360()

            # Record open directions from this position
            open_sectors = self._find_open_sectors(sweep, pos)
            self._open_dirs.extend(open_sectors)
            self.get_logger().info(
                f'[SCAN] Found {len(open_sectors)} open directions '
                f'(potential doorways/rooms): '
                f'{[f"{math.degrees(a):.0f}°@{d:.1f}m" for a, d, _, _ in open_sectors]}')

            if target is not None:
                depth_str = f'{target.get("depth_m"):.2f}m' if target.get("depth_m") else 'no depth'
                self.get_logger().info(
                    f'[FOUND] "{self._goal["target"]}" '
                    f'conf={target.get("confidence", 0):.2f} depth={depth_str}')
                self._pub_status(f'found:{self._goal["target"]}')

                aligned = self._do_align(target)
                if not aligned:
                    self.get_logger().warn('[ALIGN] Lost target — rescanning.')
                    continue

                result = self._do_approach()
                if result == 'reached':
                    self._pub_status(f'reached:{self._goal["target"]}')
                    self.get_logger().info('[REACHED] At target.')
                    return
                elif result == 'lost':
                    self.get_logger().warn('[APPROACH] Lost target — rescanning.')
                    continue

            scan_round += 1
            if scan_round < MAX_SCAN_ROUNDS and not self._interrupt.is_set():
                moved = self._do_explore_move()
                if not moved:
                    self.get_logger().warn('[EXPLORE] No viable direction — giving up.')
                    break

        self._stop()
        if not self._interrupt.is_set():
            self.get_logger().warn(f'[FAILED] "{self._goal["target"]}" not found.')
            self._pub_status('failed:not found')

    # ── PHASE 1: scan 360° + build sweep map ─────────────────────────────────

    def _do_scan_360(self):
        """
        Rotate one full revolution.
        Returns (target_detection | None, sweep_map).
        sweep_map: list of (world_angle_rad, lidar_dist) recorded during rotation.
        """
        turned    = 0.0
        prev_t    = time.time()
        log_tick  = 0
        sweep     = []   # (world_angle_rad, lidar_front_dist)
        sample_every = 4   # sample sweep every N ticks (every ~0.2s)
        tick = 0

        while turned < 2 * math.pi and not self._interrupt.is_set():
            time.sleep(0.05)
            now    = time.time()
            dt     = now - prev_t
            prev_t = now
            turned += SCAN_ANGULAR * dt
            self._vel(0.0, SCAN_ANGULAR)
            tick += 1

            # Sample current heading + lidar front for sweep map
            if tick % sample_every == 0:
                front_dist = self._lidar_front()
                world_angle = self._yaw   # current robot heading in world frame
                sweep.append((world_angle, front_dist))

            log_tick += 1
            if log_tick % 20 == 0:
                visible = [f'{d["label"]}({d["confidence"]:.2f})'
                           for d in self._latest_dets]
                lidar_f = self._lidar_front()
                self.get_logger().info(
                    f'[SCAN] {math.degrees(turned):.0f}° '
                    f'lidar_front={lidar_f:.2f}m | {visible or "nothing"}')

            target = self._find_target()
            if target is not None:
                self._stop()
                return target, sweep

        self._stop()
        return None, sweep

    def _find_open_sectors(self, sweep, origin_pos):
        """
        From a sweep map, find angular sectors where lidar > DOORWAY_DIST.
        These are potential doorways or passages to the next room.
        Returns list of (world_angle_rad, avg_dist, origin_x, origin_y).
        """
        if not sweep:
            return []

        # Group consecutive open readings into sectors
        open_sectors = []
        sector_start = None
        sector_dists = []

        for i, (angle, dist) in enumerate(sweep):
            is_open = (dist > DOORWAY_DIST) or math.isinf(dist)
            if is_open:
                if sector_start is None:
                    sector_start = angle
                sector_dists.append(min(dist, 10.0))
            else:
                if sector_start is not None and len(sector_dists) >= 2:
                    # Valid open sector
                    avg_angle = (sector_start + angle) / 2.0
                    avg_dist  = sum(sector_dists) / len(sector_dists)
                    # Only record if not already visited in that direction
                    dest_x = origin_pos[0] + EXPLORE_DIST * math.cos(avg_angle)
                    dest_y = origin_pos[1] + EXPLORE_DIST * math.sin(avg_angle)
                    if not self._is_visited(dest_x, dest_y):
                        open_sectors.append((avg_angle, avg_dist, origin_pos[0], origin_pos[1]))
                sector_start = None
                sector_dists = []

        # Handle wrap-around
        if sector_start is not None and len(sector_dists) >= 2:
            avg_angle = (sector_start + sweep[-1][0]) / 2.0
            avg_dist  = sum(sector_dists) / len(sector_dists)
            dest_x = origin_pos[0] + EXPLORE_DIST * math.cos(avg_angle)
            dest_y = origin_pos[1] + EXPLORE_DIST * math.sin(avg_angle)
            if not self._is_visited(dest_x, dest_y):
                open_sectors.append((avg_angle, avg_dist, origin_pos[0], origin_pos[1]))

        return open_sectors

    # ── PHASE 2: align ───────────────────────────────────────────────────────

    def _do_align(self, _initial):
        timeout = time.time() + 8.0
        while not self._interrupt.is_set() and time.time() < timeout:
            time.sleep(0.05)
            target = self._find_target()
            if target is None:
                self._stop()
                return False

            bbox = target.get('bbox')
            if not bbox:
                continue

            cx  = (bbox[0] + bbox[2]) / 2.0
            err = (cx - IMG_WIDTH / 2.0) / IMG_WIDTH

            self.get_logger().info(
                f'[ALIGN] bbox_cx={cx:.0f} err={err:+.3f} (need |err|<{CENTER_TOL})')

            if abs(err) < CENTER_TOL:
                self._stop()
                self.get_logger().info('[ALIGN] Centred — starting approach.')
                return True

            angular = max(-MAX_ALIGN_ANG, min(MAX_ALIGN_ANG, -ALIGN_KP * err))
            self._vel(0.0, angular)

        self._stop()
        return False

    # ── PHASE 3: approach (lidar-based reached) ──────────────────────────────

    def _do_approach(self):
        reach_count = 0
        lost_since  = None
        log_tick    = 0

        while not self._interrupt.is_set():
            time.sleep(0.05)
            log_tick += 1

            target = self._find_target()
            if target is None:
                if lost_since is None:
                    lost_since = time.time()
                    self.get_logger().warn('[APPROACH] Target not visible...')
                elif time.time() - lost_since > DETECTION_STALE_S:
                    self._stop()
                    return 'lost'
                self._vel(APPROACH_SPEED * 0.5, 0.0)
                continue
            lost_since = None

            lidar_front = self._lidar_front()

            # Minor heading correction + bbox proximity check
            bbox = target.get('bbox')
            angular = 0.0
            bbox_frac = 0.0
            if bbox:
                cx  = (bbox[0] + bbox[2]) / 2.0
                err = (cx - IMG_WIDTH / 2.0) / IMG_WIDTH
                angular = max(-0.2, min(0.2, -0.4 * err))
                bbox_frac = (bbox[2] - bbox[0]) / IMG_WIDTH

            # Stop if LiDAR sees obstacle OR bbox fills enough of the frame
            # (bbox fallback handles small objects the LiDAR scan plane misses)
            close_enough = (lidar_front <= REACH_DIST_M) or (bbox_frac >= REACH_BBOX_FRAC)

            if close_enough:
                reach_count += 1
                reason = (f'lidar={lidar_front:.2f}m' if lidar_front <= REACH_DIST_M
                          else f'bbox={bbox_frac:.0%}')
                self.get_logger().info(
                    f'[APPROACH] {reason} → REACHED? (confirm {reach_count}/{REACH_CONSEC})')
                if reach_count >= REACH_CONSEC:
                    self._stop()
                    return 'reached'
            else:
                reach_count = 0

            if lidar_front < OBSTACLE_DIST:
                self._stop()
                return 'reached'

            if log_tick % 10 == 0:
                conf = target.get('confidence', 0)
                self.get_logger().info(
                    f'[APPROACH] lidar={lidar_front:.2f}m bbox={bbox_frac:.0%} '
                    f'conf={conf:.2f} ang={angular:+.2f}')

            self._vel(APPROACH_SPEED, angular)

        self._stop()
        return 'interrupted'

    # ── exploration move: prefer open doorway directions ─────────────────────

    def _do_explore_move(self):
        """
        Pick best next direction:
          1. Prefer previously detected open sectors (doorways) that are unvisited
          2. Fall back to trying fixed offsets (0°, ±60°, ±120°, 180°)
        """
        # Sort open dirs by distance (prefer closer / more accessible)
        candidates = sorted(self._open_dirs, key=lambda x: x[1], reverse=True)

        for world_angle, avg_dist, ox, oy in candidates:
            dest_x = self._x + EXPLORE_DIST * math.cos(world_angle)
            dest_y = self._y + EXPLORE_DIST * math.sin(world_angle)
            if self._is_visited(dest_x, dest_y):
                continue

            # Turn to face this direction
            angle_to_turn = self._angle_diff(world_angle, self._yaw)
            self.get_logger().info(
                f'[EXPLORE] Heading toward open sector at '
                f'{math.degrees(world_angle):.0f}° (world), '
                f'turning {math.degrees(angle_to_turn):.0f}°, '
                f'clearance={avg_dist:.1f}m')
            self._turn_by(angle_to_turn)
            time.sleep(0.15)

            if self._lidar_front() < OBSTACLE_DIST + 0.1:
                self.get_logger().info('[EXPLORE] Open sector now blocked — skipping.')
                continue

            self._drive_straight(EXPLORE_DIST)
            return True

        # Fallback: try fixed turn increments
        self.get_logger().info('[EXPLORE] No open sectors — trying fixed directions.')
        for turn_deg in [0, 60, -60, 90, -90, 120, -120, 180]:
            if self._interrupt.is_set():
                return False
            if turn_deg != 0:
                self._turn_by(math.radians(turn_deg))
                time.sleep(0.15)
            lidar_f = self._lidar_front()
            if lidar_f < OBSTACLE_DIST + 0.1:
                continue
            dest_x = self._x + EXPLORE_DIST * math.cos(self._yaw)
            dest_y = self._y + EXPLORE_DIST * math.sin(self._yaw)
            if self._is_visited(dest_x, dest_y):
                continue
            self.get_logger().info(
                f'[EXPLORE] Moving {EXPLORE_DIST}m forward (lidar={lidar_f:.2f}m)')
            self._drive_straight(EXPLORE_DIST)
            return True

        return False

    # ── lidar helpers ────────────────────────────────────────────────────────

    def _lidar_front(self) -> float:
        if not self._scan_ranges:
            return float('inf')
        sector = math.radians(FRONT_HALF_DEG)
        dists  = []
        for i, r in enumerate(self._scan_ranges):
            if r == 0.0 or math.isnan(r) or math.isinf(r):
                continue
            angle = self._scan_angle_min + i * self._scan_angle_inc
            while angle >  math.pi: angle -= 2 * math.pi
            while angle < -math.pi: angle += 2 * math.pi
            if abs(angle) <= sector:
                dists.append(r)
        return min(dists) if dists else float('inf')

    # ── motion primitives ────────────────────────────────────────────────────

    def _drive_straight(self, dist_m: float):
        start_x, start_y = self._x, self._y
        while not self._interrupt.is_set():
            time.sleep(0.05)
            if math.hypot(self._x - start_x, self._y - start_y) >= dist_m:
                break
            if self._lidar_front() < OBSTACLE_DIST:
                self.get_logger().warn('[DRIVE] Obstacle — stopping.')
                break
            self._vel(0.15, 0.0)
        self._stop()

    def _turn_by(self, radians: float):
        if abs(radians) < 0.01:
            return
        direction = 1.0 if radians > 0 else -1.0
        duration  = abs(radians) / SCAN_ANGULAR
        end_t     = time.time() + duration
        while not self._interrupt.is_set() and time.time() < end_t:
            time.sleep(0.05)
            self._vel(0.0, direction * SCAN_ANGULAR)
        self._stop()

    @staticmethod
    def _angle_diff(target: float, current: float) -> float:
        d = target - current
        while d >  math.pi: d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        return d

    def _is_visited(self, x: float, y: float) -> bool:
        return any(math.hypot(x - vx, y - vy) < VISITED_RADIUS
                   for vx, vy in self._visited)

    def _vel(self, linear: float, angular: float):
        t = Twist()
        t.linear.x  = float(linear)
        t.angular.z = float(angular)
        self._vel_pub.publish(t)

    def _stop(self):
        self._vel_pub.publish(Twist())

    # ── target detection ─────────────────────────────────────────────────────

    def _find_target(self):
        if not self._goal:
            return None
        if time.time() - self._det_timestamp > DETECTION_STALE_S:
            return None
        target_label  = self._goal['target']
        context_label = self._goal.get('context')
        candidates = [
            d for d in self._latest_dets
            if d['label'] == target_label
            and d.get('confidence', 0) > (0.03 if d.get('source') == 'color' else 0.20)
        ]
        if not candidates:
            return None
        if context_label:
            if (time.time() - self._context_last_seen) > CONTEXT_WINDOW_S:
                return None
        return max(candidates,
                   key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1])
                   if d.get('bbox') else 0)

    # ── RViz visualisation ───────────────────────────────────────────────────

    def _publish_viz(self):
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()

        # 1. Visited scan positions — green spheres
        for i, (vx, vy) in enumerate(self._visited):
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp    = now
            m.ns     = 'visited'
            m.id     = i
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = vx
            m.pose.position.y = vy
            m.pose.position.z = 0.15
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color = ColorRGBA(r=0.0, g=0.9, b=0.2, a=0.85)
            markers.markers.append(m)

        # 2. Open direction arrows — yellow arrows
        for j, (world_angle, avg_dist, ox, oy) in enumerate(self._open_dirs):
            arrow = Marker()
            arrow.header.frame_id = 'odom'
            arrow.header.stamp    = now
            arrow.ns     = 'open_dirs'
            arrow.id     = j
            arrow.type   = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.scale.x = 0.05   # shaft diameter
            arrow.scale.y = 0.12   # head diameter
            arrow.scale.z = 0.10   # head length
            arrow.color = ColorRGBA(r=1.0, g=0.85, b=0.0, a=0.9)
            start = Point(x=ox, y=oy, z=0.1)
            end   = Point(
                x=ox + min(avg_dist, 2.0) * math.cos(world_angle),
                y=oy + min(avg_dist, 2.0) * math.sin(world_angle),
                z=0.1)
            arrow.points = [start, end]
            markers.markers.append(arrow)

        # 3. Current robot position — red sphere
        cur = Marker()
        cur.header.frame_id = 'odom'
        cur.header.stamp    = now
        cur.ns     = 'current'
        cur.id     = 0
        cur.type   = Marker.SPHERE
        cur.action = Marker.ADD
        cur.pose.position.x = self._x
        cur.pose.position.y = self._y
        cur.pose.position.z = 0.3
        cur.pose.orientation.w = 1.0
        cur.scale.x = cur.scale.y = cur.scale.z = 0.2
        cur.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)
        markers.markers.append(cur)

        self._marker_pub.publish(markers)

        # 4. Full robot path — blue line strip
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp    = now
        path.poses = self._path_poses[-500:]   # last 500 poses
        self._path_pub.publish(path)

    def _pub_status(self, s: str):
        msg = String(); msg.data = s
        self._status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VLANavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
