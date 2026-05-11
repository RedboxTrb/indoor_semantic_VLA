#!/usr/bin/env python3
"""
Habitat→ROS2 bridge node.
Runs in 'habitat_ros' conda env (Python 3.10) with ROS2 Humble sourced.

Usage:
    conda activate habitat_ros
    source /opt/ros/humble/setup.bash
    source ~/okvis_ws/install/setup.bash
    python3 ~/habitat_ros_node.py

Publishes:
    /d435i_depth_camera/image_raw       sensor_msgs/Image  bgr8
    /d435i_depth_camera/depth/image_raw sensor_msgs/Image  32FC1
    /d435i_depth_camera/camera_info     sensor_msgs/CameraInfo
    /d435i_depth_camera/depth/camera_info
    /imu/data                           sensor_msgs/Imu   100 Hz synthetic
"""
import csv, os, threading, time, json
import numpy as np
import zmq
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time

ZMQ_PORT   = 5555
IMG_W      = 640
IMG_H      = 480
FX = FY    = 390.598938
CX         = 320.581665
CY         = 237.712845
FRAME_ID   = 'realsense_DCAM_1_optical'
IMU_HZ     = 100
G          = 9.81007   # matches okvis2.yaml
CAM_HEIGHT = 0.8       # camera height above floor [m] — matches habitat_bridge_pub.py

# 2D costmap grid: 512×512 cells at 5cm/cell = 25.6m × 25.6m
MAP_RES   = 0.05
MAP_CELLS = 512

# Habitat uses Y-up / right-hand (OpenGL).  OKVIS body frame (T_BS=I) is
# camera-aligned: X-right, Y-down, Z-forward.
# Rotation from habitat world axes to OKVIS body axes:
#   hab_x → body_x  (right stays right)
#   hab_y → -body_y  (up → -down)
#   hab_z → -body_z  (backward → -forward)
_R_HAB_TO_BODY = np.array([
    [ 1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0, -1],
], dtype=np.float64)
# Gravity in habitat world frame [0, -g, 0] (Y is up → gravity is -Y)
_G_WORLD_HAB = np.array([0.0, -G, 0.0])


def _quat_to_rot(w, x, y, z) -> np.ndarray:
    """Quaternion → 3×3 rotation matrix (R maps body→world)."""
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


class ImuState:
    """
    Synthetic IMU from Habitat ground-truth poses.

    Strategy:
    - Stationary frames  → perfect zero angular vel + gravity from current orientation
    - Moving frames      → angular velocity from rotation change only
                           (no translational acceleration — depth handles scale,
                            and Habitat's discrete teleport steps produce spurious
                            acceleration spikes if finite-differenced)
    """
    def __init__(self):
        self.t    = None
        self.quat = None   # [w,x,y,z] previous

    def _gravity_body(self, quat) -> np.ndarray:
        """Specific force = -gravity expressed in OKVIS body frame."""
        # R_quat: habitat body → habitat world
        R_quat   = _quat_to_rot(*quat)
        # R_okvis_to_world = R_quat @ R_HAB_TO_BODY^T
        R_b2w    = R_quat @ _R_HAB_TO_BODY.T
        # specific_force_world = -g_world = [0, +G, 0] (Y-up hab, upward)
        sf_world = -_G_WORLD_HAB                   # [0, +G, 0]
        return R_b2w.T @ sf_world                  # rotate to body frame

    def update(self, t: float, pos, quat, moving: bool):
        """Return (ang_vel_body [3], lin_acc_body [3])."""
        quat = np.array(quat, dtype=np.float64)

        if self.t is None or not moving:
            # First call or stationary: perfect zero motion
            self.t, self.quat = t, quat
            return np.zeros(3), self._gravity_body(quat)

        dt = max(t - self.t, 1e-6)

        # Angular velocity from quaternion change (clean, no position noise)
        pw, px, py, pz = self.quat
        nw, nx, ny, nz = quat
        # q_rel = conj(q_prev) * q_now
        cw, cx, cy, cz  = pw, -px, -py, -pz
        rw = cw*nw - cx*nx - cy*ny - cz*nz
        rx = cw*nx + cx*nw + cy*nz - cz*ny
        ry = cw*ny - cx*nz + cy*nw + cz*nx
        rz = cw*nz + cx*ny - cy*nx + cz*nw
        ang_vel_hab  = 2.0 * np.array([rx, ry, rz]) / dt
        ang_vel_body = _R_HAB_TO_BODY @ ang_vel_hab

        self.t, self.quat = t, quat
        # Linear acc = gravity only (no translational component — avoids teleport spikes)
        return ang_vel_body, self._gravity_body(quat)


class HabitatBridgeNode(Node):
    def __init__(self):
        super().__init__('habitat_bridge')

        self._pub_rgb       = self.create_publisher(Image,      '/d435i_depth_camera/image_raw',               10)
        self._pub_rgb_right = self.create_publisher(Image,      '/d435i_depth_camera/right/image_raw',         10)
        self._pub_depth     = self.create_publisher(Image,      '/d435i_depth_camera/depth/image_raw',         10)
        self._pub_ci        = self.create_publisher(CameraInfo, '/d435i_depth_camera/camera_info',             10)
        self._pub_ci_right  = self.create_publisher(CameraInfo, '/d435i_depth_camera/right/camera_info',       10)
        self._pub_dci       = self.create_publisher(CameraInfo, '/d435i_depth_camera/depth/camera_info',       10)
        self._pub_imu       = self.create_publisher(Imu,        '/imu/data',                                   10)
        self._pub_costmap   = self.create_publisher(OccupancyGrid, '/habitat/costmap_2d',                    1)
        self._pub_path2d    = self.create_publisher(Path,          '/habitat/path_2d',                       1)

        self._latest    = None
        self._lock      = threading.Lock()
        self._imu_state = ImuState()
        # Latest synthetic IMU reading (updated each new frame, held between frames)
        self._imu_ang   = np.zeros(3)
        self._imu_acc   = np.array([0.0, -G, 0.0])  # specific force for level camera: body Y-down → -G
        self._imu_lock  = threading.Lock()

        self._stop      = False
        self._zmq_thread_handle = threading.Thread(target=self._zmq_thread, daemon=True)
        self._zmq_thread_handle.start()

        # IMU runs in its own thread — rclpy's single-threaded executor can't
        # sustain 100 Hz when large images are being serialized alongside.
        self._imu_thread_handle = threading.Thread(target=self._imu_thread, daemon=True)
        self._imu_thread_handle.start()

        self.create_timer(1.0 / 30.0, self._publish_frame)
        self.create_timer(0.5,        self._publish_costmap)   # 2 Hz

        # 2D costmap state
        self._costmap_grid   = np.full((MAP_CELLS, MAP_CELLS), -1, dtype=np.int8)
        self._costmap_origin = None   # (ox, oz) world-frame corner of grid
        self._path_poses     = []
        self._latest_pose_pos  = None  # (x, y, z) from OKVIS odometry
        self._latest_pose_quat = None  # (w, x, y, z)

        # Trajectory logger
        self._traj_csv = open(os.path.expanduser('~/okvis_trajectory.csv'), 'w', newline='')
        self._traj_writer = csv.writer(self._traj_csv)
        self._traj_writer.writerow(['ros_t', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        self._traj_csv.flush()
        self._odom_sub = self.create_subscription(
            Odometry, '/okvis/okvis_odometry', self._odom_cb, 10)

        self.get_logger().info(
            f'Habitat bridge ready — ZMQ tcp://localhost:{ZMQ_PORT}'
        )
        self.get_logger().info('Trajectory logging to ~/okvis_trajectory.csv')

    # ── ZMQ receive thread ────────────────────────────────────────────────────
    def _zmq_thread(self):
        ctx  = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b'')
        sock.setsockopt(zmq.RCVTIMEO, 200)
        sock.connect(f'tcp://localhost:{ZMQ_PORT}')
        self.get_logger().info('ZMQ SUB connected')
        while not self._stop:
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                continue
            if len(parts) not in (3, 4):
                continue
            try:
                hdr       = json.loads(parts[0])
                bgr       = np.frombuffer(parts[1], dtype=np.uint8 ).reshape(IMG_H, IMG_W, 3).copy()
                bgr_right = np.frombuffer(parts[2], dtype=np.uint8 ).reshape(IMG_H, IMG_W, 3).copy() if len(parts) == 4 else None
                depth     = np.frombuffer(parts[-1], dtype=np.float32).reshape(IMG_H, IMG_W  ).copy()

                # Compute synthetic IMU if pose data present
                if 'pos' in hdr and 'quat' in hdr:
                    ang, acc = self._imu_state.update(
                        hdr['t'], hdr['pos'], hdr['quat'],
                        hdr.get('moving', False)
                    )
                    with self._imu_lock:
                        self._imu_ang = ang
                        self._imu_acc = acc

                with self._lock:
                    self._latest = (hdr['t'], bgr, bgr_right, depth)
            except Exception as exc:
                self.get_logger().warn(f'ZMQ decode error: {exc}')
        sock.close()
        ctx.term()

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _float_to_stamp(t: float) -> Time:
        s = Time()
        s.sec     = int(t)
        s.nanosec = int((t - int(t)) * 1e9)
        return s

    def _make_camera_info(self, stamp) -> CameraInfo:
        ci = CameraInfo()
        ci.header.stamp    = stamp
        ci.header.frame_id = FRAME_ID
        ci.width  = IMG_W
        ci.height = IMG_H
        ci.distortion_model = 'plumb_bob'
        ci.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        ci.k = [FX,  0.0, CX, 0.0, FY,  CY, 0.0, 0.0, 1.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.p = [FX,  0.0, CX, 0.0, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]
        return ci

    # ── OKVIS odometry subscriber ─────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        t  = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p  = msg.pose.pose.position
        v  = msg.twist.twist.linear
        row = [f'{t:.3f}', f'{p.x:.4f}', f'{p.y:.4f}', f'{p.z:.4f}',
               f'{v.x:.4f}', f'{v.y:.4f}', f'{v.z:.4f}']
        self._traj_writer.writerow(row)
        self._traj_csv.flush()
        self.get_logger().info(
            f'[odom] xyz=({p.x:.3f},{p.y:.3f},{p.z:.3f})  '
            f'vel=({v.x:.3f},{v.y:.3f},{v.z:.3f})',
            throttle_duration_sec=1.0)

        # Store latest pose for costmap projection
        q = msg.pose.pose.orientation
        self._latest_pose_pos  = (p.x, p.y, p.z)
        self._latest_pose_quat = (q.w, q.x, q.y, q.z)

        # Accumulate path for 2D trajectory display
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose   = msg.pose.pose
        self._path_poses.append(ps)

    # ── Timer callbacks ───────────────────────────────────────────────────────
    def _publish_frame(self):
        with self._lock:
            data         = self._latest
            self._latest = None
        if data is None:
            return

        t_float, bgr, bgr_right, depth = data
        stamp = self._float_to_stamp(t_float)

        def _make_rgb_msg(pixels):
            m = Image()
            m.header.stamp    = stamp
            m.header.frame_id = FRAME_ID
            m.height          = IMG_H
            m.width           = IMG_W
            m.encoding        = 'bgr8'
            m.is_bigendian    = False
            m.step            = IMG_W * 3
            m.data            = pixels.tobytes()
            return m

        self._pub_rgb.publish(_make_rgb_msg(bgr))
        if bgr_right is not None:
            self._pub_rgb_right.publish(_make_rgb_msg(bgr_right))

        dep_msg = Image()
        dep_msg.header.stamp    = stamp
        dep_msg.header.frame_id = FRAME_ID
        dep_msg.height    = IMG_H
        dep_msg.width     = IMG_W
        dep_msg.encoding  = '32FC1'
        dep_msg.is_bigendian = False
        dep_msg.step      = IMG_W * 4
        dep_msg.data      = depth.tobytes()
        self._pub_depth.publish(dep_msg)

        ci = self._make_camera_info(stamp)
        self._pub_ci.publish(ci)
        if bgr_right is not None:
            self._pub_ci_right.publish(ci)
        self._pub_dci.publish(ci)

        if self._latest_pose_pos is not None:
            self._update_costmap(depth, self._latest_pose_pos, self._latest_pose_quat)

    # ── 2D costmap ────────────────────────────────────────────────────────────
    def _update_costmap(self, depth: np.ndarray, pos, quat_wxyz):
        """Project depth frame into 2D occupancy grid using latest OKVIS pose.

        OKVIS world frame (IMU disabled, T_BS=I): X-right, Y-down, Z-forward.
        Floor plane is X-Z; Y is approximately vertical. We map world-X → col,
        world-Z → row to get a standard top-down view.
        """
        step = 8   # 8× downsample: 480→60 rows, 640→80 cols
        d    = depth[::step, ::step]
        H, W = d.shape

        u = (np.arange(W) * step).astype(np.float32)
        v = (np.arange(H) * step).astype(np.float32)
        uu, vv = np.meshgrid(u, v)

        valid = (d > 0.1) & (d < 3.5)

        X_cam = (uu - CX) * d / FX
        Y_cam = (vv - CY) * d / FY   # positive = below camera (Y-down body frame)
        Z_cam = d

        # Keep points in obstacle band 5cm–150cm above floor
        h_floor  = CAM_HEIGHT - Y_cam
        obstacle = valid & (h_floor > 0.05) & (h_floor < 1.5)

        if not np.any(obstacle):
            return

        pts = np.stack([X_cam[obstacle], Y_cam[obstacle], Z_cam[obstacle]], axis=1)

        R       = _quat_to_rot(*quat_wxyz)
        pos_arr = np.array(pos, dtype=np.float64)
        pts_w   = (R @ pts.T).T + pos_arr   # (N,3) in OKVIS world frame

        if self._costmap_origin is None:
            half = MAP_CELLS * MAP_RES / 2.0
            self._costmap_origin = (pos_arr[0] - half, pos_arr[2] - half)
        ox, oz = self._costmap_origin

        col = ((pts_w[:, 0] - ox) / MAP_RES).astype(int)
        row = ((pts_w[:, 2] - oz) / MAP_RES).astype(int)
        ok  = (col >= 0) & (col < MAP_CELLS) & (row >= 0) & (row < MAP_CELLS)
        self._costmap_grid[row[ok], col[ok]] = 100   # occupied

        # Mark robot footprint as free space
        rx = int((pos_arr[0] - ox) / MAP_RES)
        rz = int((pos_arr[2] - oz) / MAP_RES)
        r  = 3
        r0 = max(0, rz - r); r1 = min(MAP_CELLS, rz + r + 1)
        c0 = max(0, rx - r); c1 = min(MAP_CELLS, rx + r + 1)
        patch = self._costmap_grid[r0:r1, c0:c1]
        patch[patch == -1] = 0   # unknown → free within footprint

    def _publish_costmap(self):
        if self._costmap_origin is None:
            return
        stamp = self.get_clock().now().to_msg()
        ox, oz = self._costmap_origin

        og = OccupancyGrid()
        og.header.stamp    = stamp
        og.header.frame_id = 'odom'
        og.info.resolution = MAP_RES
        og.info.width      = MAP_CELLS
        og.info.height     = MAP_CELLS
        og.info.origin.position.x = float(ox)
        og.info.origin.position.y = float(oz)
        og.info.origin.position.z = 0.0
        og.data = self._costmap_grid.flatten().tolist()
        self._pub_costmap.publish(og)

        path = Path()
        path.header.stamp    = stamp
        path.header.frame_id = 'odom'
        path.poses = list(self._path_poses)
        self._pub_path2d.publish(path)

    def _imu_thread(self):
        """Dedicated 100 Hz IMU publisher thread — bypasses rclpy executor."""
        dt       = 1.0 / IMU_HZ
        tick     = 0
        t_next   = time.monotonic() + dt
        while not self._stop:
            now = time.monotonic()
            if now < t_next:
                time.sleep(t_next - now)
            t_next += dt

            with self._imu_lock:
                ang = self._imu_ang.copy()
                acc = self._imu_acc.copy()

            tick += 1
            if tick % 100 == 1:   # once per second
                self.get_logger().info(
                    f'[IMU] acc=({acc[0]:.3f},{acc[1]:.3f},{acc[2]:.3f})  '
                    f'gyr=({ang[0]:.4f},{ang[1]:.4f},{ang[2]:.4f})')

            msg = Imu()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = 'imu_link'
            msg.angular_velocity.x    = float(ang[0])
            msg.angular_velocity.y    = float(ang[1])
            msg.angular_velocity.z    = float(ang[2])
            msg.linear_acceleration.x = float(acc[0])
            msg.linear_acceleration.y = float(acc[1])
            msg.linear_acceleration.z = float(acc[2])
            msg.orientation_covariance[0] = -1.0
            self._pub_imu.publish(msg)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def destroy_node(self):
        self._stop = True
        self._zmq_thread_handle.join(timeout=2.0)
        self._imu_thread_handle.join(timeout=2.0)
        self._traj_csv.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = HabitatBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
