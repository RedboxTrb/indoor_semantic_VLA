# Semantic Mobility-Aware Navigation

A ROS 2 (Humble) system that lets a Gazebo-simulated robot find and navigate to objects described in plain English. It uses CLIP for zero-shot semantic detection of tables and bookshelves, HSV thresholding for coloured cubes, and Gemini to parse natural-language commands — all without any pre-built map.

## Setup

```bash
# Terminal 1 — Simulation + SLAM + Nav2
ros2 launch basebot_urdf_description slam_nav.launch.py

# Terminal 2 — Semantic detector
ros2 run semantic_mobility_nav detector_clip

# Terminal 3 — VLA navigator
ros2 run semantic_mobility_nav vla_navigator

# Terminal 4 — CLI (type commands here)
ros2 run semantic_mobility_nav robot_cli
```

Example commands: `go to the blue cube`, `find the table`, `stop`
