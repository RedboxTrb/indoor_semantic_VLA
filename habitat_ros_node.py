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
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time

ZMQ_PORT = 5555
IMG_W    = 640
IMG_H    = 480
FX = FY  = 390.598938
CX       = 320.581665
CY       = 237.712845
FRAME_ID = 'realsense_DCAM_1_optical'
IMU_HZ   = 100
G        = 9.81007   # matches okvis2.yaml

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

        self._pub_rgb   = self.create_publisher(Image,      '/d435i_depth_camera/image_raw',         10)
        self._pub_depth = self.create_publisher(Image,      '/d435i_depth_camera/depth/image_raw',   10)
        self._pub_ci    = self.create_publisher(CameraInfo, '/d435i_depth_camera/camera_info',       10)
        self._pub_dci   = self.create_publisher(CameraInfo, '/d435i_depth_camera/depth/camera_info', 10)
        self._pub_imu   = self.create_publisher(Imu,        '/imu/data',                             10)

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
            if len(parts) != 3:
                continue
            try:
                hdr   = json.loads(parts[0])
                bgr   = np.frombuffer(parts[1], dtype=np.uint8 ).reshape(IMG_H, IMG_W, 3).copy()
                depth = np.frombuffer(parts[2], dtype=np.float32).reshape(IMG_H, IMG_W   ).copy()

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
                    self._latest = (hdr['t'], bgr, depth)
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

    # ── Timer callbacks ───────────────────────────────────────────────────────
    def _publish_frame(self):
        with self._lock:
            data         = self._latest
            self._latest = None
        if data is None:
            return

        t_float, bgr, depth = data
        stamp = self._float_to_stamp(t_float)

        rgb_msg = Image()
        rgb_msg.header.stamp    = stamp
        rgb_msg.header.frame_id = FRAME_ID
        rgb_msg.height    = IMG_H
        rgb_msg.width     = IMG_W
        rgb_msg.encoding  = 'bgr8'
        rgb_msg.is_bigendian = False
        rgb_msg.step      = IMG_W * 3
        rgb_msg.data      = bgr.tobytes()
        self._pub_rgb.publish(rgb_msg)

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
        self._pub_dci.publish(ci)

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
