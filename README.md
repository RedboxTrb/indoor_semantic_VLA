# Habitat-Sim → OKVIS2-X SLAM Bridge

Live RGB-D + synthetic IMU pipeline:
**Habitat-Sim** renders a photorealistic scene → **ZMQ** → **ROS2 bridge** → **OKVIS2-X (FindAnything)** builds a 3-D semantic SLAM map viewable in RViz.

---

## Prerequisites

| Component | Version / Notes |
|---|---|
| OS | Ubuntu 22.04 |
| ROS2 | Humble |
| Conda envs | `habitat` (Python 3.9) · `habitat_ros` (Python 3.10) |
| OKVIS2-X workspace | `~/okvis_ws` — built with `colcon build --symlink-install` |
| libtorch | `~/libtorch_cu126/libtorch/lib` |
| HM3D scene | `~/habitat_data/scene_datasets/hm3d/train/00033-oPj9qMxrDEa/` |

### Install Python deps (one-time)

```bash
# Habitat publisher env
conda activate habitat
pip install pyzmq numpy-quaternion

# ROS bridge env
conda activate habitat_ros
pip install pyzmq numpy
```

### Download HM3D scene 00033 (one-time, ~620 MB)

```bash
conda activate habitat
python3 ~/habitat_slam/download_hm3d_scene.py
# Enter Matterport API Token ID and Token Secret when prompted
# (get them at: https://my.matterport.com/settings/account/devtools)
```

### Copy launch file into ROS workspace (one-time)

```bash
cp ~/habitat_slam/launch/habitat_okvis.launch.py \
   ~/okvis_ws/src/major-project/src/basebot_urdf_description/launch/

cd ~/okvis_ws
colcon build --packages-select basebot_urdf_description --symlink-install
```

---

## 4-Terminal Launch

Open **4 terminals** in order.

### Terminal 1 — Habitat publisher (`habitat` conda env)

```bash
conda activate habitat
python3 ~/habitat_slam/habitat_bridge_pub.py
```

Renders RGB + depth at 20 Hz and streams over ZMQ on port 5555.  
Wait for: `Sim ready. Publishing on tcp://*:5555`

**Controls** (press key, then Enter is NOT needed — single keypress):

| Key | Action |
|---|---|
| `w` | Move forward |
| `s` | Move backward |
| `a` | Turn left |
| `d` | Turn right |
| `r` | Reset to random navigable point |
| `q` | Quit |

### Terminal 2 — OKVIS2-X SLAM node

```bash
source /opt/ros/humble/setup.bash
source ~/okvis_ws/install/setup.bash
ros2 launch basebot_urdf_description habitat_okvis.launch.py
```

Reads config from `~/habitat_slam/habitat_okvis2.yaml`.  
Wait for: `[okvis]: Initialized!` before moving the agent.

### Terminal 3 — ROS2 bridge (`habitat_ros` conda env)

```bash
conda activate habitat_ros
source /opt/ros/humble/setup.bash
source ~/okvis_ws/install/setup.bash
python3 ~/habitat_slam/habitat_ros_node.py
```

Publishes:
- `/d435i_depth_camera/image_raw` (bgr8, 20 Hz)
- `/d435i_depth_camera/depth/image_raw` (32FC1, 20 Hz)
- `/d435i_depth_camera/camera_info` and `depth/camera_info`
- `/imu/data` (100 Hz synthetic IMU, dedicated thread)

Logs OKVIS pose to `~/okvis_trajectory.csv`.

### Terminal 4 — RViz visualisation

```bash
source /opt/ros/humble/setup.bash
source ~/okvis_ws/install/setup.bash
ros2 run rviz2 rviz2 -d \
  ~/okvis_ws/install/okvis/share/okvis/config/rviz2/rviz2_okvis2x_config.rviz
```

Expected topics visible in RViz:
- `/okvis/okvis_path` — 3-D trajectory
- `/okvis/sam_masks` — eSAM segmentation overlay
- `/okvis/language_rgb` — CLIP language features

---

## Verify IMU rate (optional sanity check)

In any sourced terminal while Terminal 3 is running:

```bash
source /opt/ros/humble/setup.bash
ros2 topic hz /imu/data
# Expected: average rate: 100.000
```

---

## File overview

| File | Purpose |
|---|---|
| `habitat_bridge_pub.py` | Habitat-Sim renderer + ZMQ PUB (Terminal 1) |
| `habitat_ros_node.py` | ROS2 bridge: ZMQ SUB → ROS topics + synthetic IMU (Terminal 3) |
| `habitat_okvis2.yaml` | OKVIS2-X sensor config (single cam0 + depth, IMU enabled) |
| `download_hm3d_scene.py` | Stream-downloads HM3D scene 00033 without the full 15 GB archive |
| `launch/habitat_okvis.launch.py` | ROS2 launch for OKVIS2-X node with topic remappings |

---

## Known config decisions

- **Single camera**: cam1 `slam_use: none` — both cam0 and cam1 subscribe to the same image topic, so activating cam1 would give OKVIS a fake stereo pair with zero disparity (landmarks at infinity). Only cam0 (with depth) is used for SLAM.
- **IMU thread**: The ROS2 bridge publishes IMU from a dedicated `threading.Thread` instead of a rclpy timer. rclpy's single-threaded executor drops to ~3 Hz when serialising large depth images alongside, which breaks OKVIS preintegration.
- **Step size**: `STEP_M = 0.07 m`, `TURN_DEG = 5°` — larger steps at 15 Hz caused RANSAC failure (feature inlier ratio < threshold) in OKVIS.
- **Gravity**: synthetic IMU outputs `[0, -9.81, 0]` m/s² (OKVIS body frame: X-right, Y-down, Z-forward).
