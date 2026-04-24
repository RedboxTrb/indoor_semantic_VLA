#!/usr/bin/env python3
"""
Semantic detector: HSV color detection for cubes + CLIP for semantic objects.

Cubes (red/green/blue) are detected via HSV thresholding — perfect for Gazebo's
uniform colors and far faster than CLIP. CLIP runs at reduced rate for
table and bookshelf detection.
"""
import json
import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import clip
from PIL import Image as PILImage


CLIP_CLASSES = [
    "a flat wooden dining table with four legs",
    "a tall rectangular bookshelf with multiple shelves",
]
CLIP_CLEAN_LABELS = ["table", "bookshelf"]

MOBILITY_WEIGHTS = {
    "red cube":   0.5,
    "green cube": 0.5,
    "blue cube":  0.5,
    "table":      0.1,
    "bookshelf":  0.1,
}

# HSV ranges — cubes only
COLOR_RANGES = {
    "red cube": [
        (np.array([0,   150, 80]),  np.array([8,   255, 255])),
        (np.array([172, 150, 80]),  np.array([180, 255, 255])),
    ],
    "green cube": [
        (np.array([40,  80,  70]),  np.array([90,  255, 255])),
    ],
    "blue cube": [
        (np.array([100, 80,  70]),  np.array([130, 255, 255])),
    ],
}

MIN_CUBE_AREA_PX = 200

DEFAULT_FX = 615.0
DEFAULT_FY = 615.0
DEFAULT_CX = 320.0
DEFAULT_CY = 240.0
CAMERA_HEIGHT_M = 0.87


class SemanticDetectorCLIP(Node):
    def __init__(self):
        super().__init__('semantic_detector_clip')

        self.declare_parameter('camera_topic',      '/d435i_depth_camera/image_raw')
        self.declare_parameter('depth_topic',       '/d435i_depth_camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/d435i_depth_camera/camera_info')
        self.declare_parameter('process_rate',    5.0)   # Hz — color runs every tick, CLIP every N
        self.declare_parameter('clip_every_n',    3)     # run CLIP every Nth tick
        self.declare_parameter('confidence_threshold', 0.20)

        self.bridge = CvBridge()
        self.fx, self.fy = DEFAULT_FX, DEFAULT_FY
        self.cx, self.cy = DEFAULT_CX, DEFAULT_CY
        self.camera_info_received = False

        self.get_logger().info('Loading CLIP model (ViT-B/32)...')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        # Pre-compute text features ONCE — they never change
        prompts = [f"a photo of {c}" for c in CLIP_CLASSES]
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            self._text_features = self.clip_model.encode_text(tokens).float()
            self._text_features = self._text_features / self._text_features.norm(dim=-1, keepdim=True)
        self.get_logger().info(f'CLIP text features cached | device:{self.device}')

        camera_topic    = self.get_parameter('camera_topic').value
        depth_topic     = self.get_parameter('depth_topic').value
        cam_info_topic  = self.get_parameter('camera_info_topic').value

        # Gazebo publishes sensor data with BEST_EFFORT — must match or messages are dropped
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Image,      camera_topic,   self._image_cb,   sensor_qos)
        self.create_subscription(Image,      depth_topic,    self._depth_cb,   sensor_qos)
        self.create_subscription(CameraInfo, cam_info_topic, self._cam_info_cb, sensor_qos)

        self.image_pub      = self.create_publisher(Image,  '/semantic/annotated_image', 10)
        self.detections_pub = self.create_publisher(String, '/semantic/detections',      10)

        rate = self.get_parameter('process_rate').value
        self.create_timer(1.0 / rate, self._tick)

        self.latest_image = None
        self.latest_depth = None
        self._processing  = False
        self._tick_count  = 0

        # CLIP runs in a background thread so it never blocks the ROS spin loop
        self._clip_lock      = threading.Lock()
        self._clip_ready     = threading.Event()
        self._clip_img_buf   = None
        self._clip_dep_buf   = None
        self._last_clip_detections = []

        threading.Thread(target=self._clip_worker, daemon=True).start()

        self.get_logger().info(f'Detector ready | rgb:{camera_topic}')

    def _cam_info_cb(self, msg):
        if not self.camera_info_received:
            self.fx = msg.k[0]; self.fy = msg.k[4]
            self.cx = msg.k[2]; self.cy = msg.k[5]
            self.camera_info_received = True
            self.get_logger().info(f'Intrinsics: fx={self.fx:.1f} cx={self.cx:.1f}')

    def _image_cb(self, msg):
        if not self._processing:
            self.latest_image = msg

    def _depth_cb(self, msg):
        self.latest_depth = msg

    def _tick(self):
        if self.latest_image is None or self._processing:
            return
        self._processing = True
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
            depth_image = None
            if self.latest_depth is not None:
                try:
                    depth_image = self.bridge.imgmsg_to_cv2(
                        self.latest_depth, desired_encoding='passthrough')
                except Exception:
                    pass

            color_detections = self._detect_colors(cv_image, depth_image)

            # Signal CLIP background thread every N ticks (non-blocking)
            self._tick_count += 1
            clip_every_n = self.get_parameter('clip_every_n').value
            if self._tick_count % clip_every_n == 0 and not self._clip_ready.is_set():
                with self._clip_lock:
                    self._clip_img_buf = cv_image.copy()
                    self._clip_dep_buf = depth_image.copy() if depth_image is not None else None
                self._clip_ready.set()

            with self._clip_lock:
                last_clip = list(self._last_clip_detections)

            all_detections = color_detections + last_clip
            self._publish_detections(all_detections)
            self._publish_annotated(cv_image, all_detections)

        except Exception as e:
            self.get_logger().error(f'Processing error: {e}')
        finally:
            self._processing = False

    def _clip_worker(self):
        while True:
            fired = self._clip_ready.wait(timeout=3.0)
            if not fired:
                continue  # timeout — no new frame, loop back
            self._clip_ready.clear()
            with self._clip_lock:
                img = self._clip_img_buf
                dep = self._clip_dep_buf
            if img is None:
                continue
            try:
                results = self._run_clip(img, dep)
                with self._clip_lock:
                    self._last_clip_detections = results
                # Log only when CLIP actually computes a fresh result
                for r in results:
                    dist = f"{r['depth_m']:.2f}m" if r['depth_m'] else '?'
                    self.get_logger().info(
                        f"[CLIP] {r['label']} conf={r['confidence']:.2f} dist={dist}")
            except Exception as e:
                self.get_logger().error(f'CLIP thread error: {e}')

    # ------------------------------------------------------------------
    # Color-based cube detection (fast)
    # ------------------------------------------------------------------

    def _detect_colors(self, cv_image, depth_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        h, w = cv_image.shape[:2]
        detections = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for label, ranges in COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            valid = [c for c in contours if cv2.contourArea(c) >= MIN_CUBE_AREA_PX]
            if not valid:
                continue

            for contour in [max(valid, key=cv2.contourArea)]:
                area = cv2.contourArea(contour)
                x, y, bw, bh = cv2.boundingRect(contour)
                cx_px = x + bw // 2
                cy_px = y + bh // 2
                depth_m = self._sample_depth(depth_image, cx_px, cy_px, h, w)
                pos = self._pixel_to_robot(cx_px, cy_px, depth_m) if depth_m else None

                detections.append({
                    'label':                label,
                    'confidence':           round(min(1.0, area / (w * h) * 20), 3),
                    'mobility_weight':      MOBILITY_WEIGHTS.get(label, 0.5),
                    'depth_m':              round(depth_m, 3) if depth_m else None,
                    'position_robot_frame': pos,
                    'bbox':                 [x, y, x + bw, y + bh],
                    'source':               'color',
                })

        return detections

    # ------------------------------------------------------------------
    # CLIP detection for semantic objects (slower, runs every N ticks)
    # ------------------------------------------------------------------

    GRID_ROWS = 5
    GRID_COLS = 5

    def _run_clip(self, cv_image, depth_image):
        h, w = cv_image.shape[:2]
        threshold = self.get_parameter('confidence_threshold').value
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Step 1: full-image score to decide which classes are present
        full_input = self.clip_preprocess(PILImage.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            full_feat = self.clip_model.encode_image(full_input).float()
            full_feat = full_feat / full_feat.norm(dim=-1, keepdim=True)
            full_sim = (100.0 * full_feat @ self._text_features.T).softmax(dim=-1)[0]

        active = [(i, CLIP_CLEAN_LABELS[i]) for i in range(len(CLIP_CLASSES))
                  if full_sim[i].item() >= threshold]
        if not active:
            return []

        # Step 2: 5x5 patch grid — finer localization
        rows, cols = self.GRID_ROWS, self.GRID_COLS
        ph, pw = h // rows, w // cols
        patches, patch_boxes = [], []
        for r in range(rows):
            for c in range(cols):
                x0, y0 = c * pw, r * ph
                x1, y1 = x0 + pw, y0 + ph
                crop = PILImage.fromarray(rgb[y0:y1, x0:x1])
                patches.append(self.clip_preprocess(crop))
                patch_boxes.append((x0, y0, x1, y1))

        patch_tensor = torch.stack(patches).to(self.device)
        with torch.no_grad():
            patch_feats = self.clip_model.encode_image(patch_tensor).float()
            patch_feats = patch_feats / patch_feats.norm(dim=-1, keepdim=True)
            patch_sim = (100.0 * patch_feats @ self._text_features.T).softmax(dim=-1)

        detections = []
        for cls_idx, clean_label in active:
            conf = full_sim[cls_idx].item()
            best_patch = int(patch_sim[:, cls_idx].argmax().item())
            x0, y0, x1, y1 = patch_boxes[best_patch]

            # Step 3: contour refinement inside the best patch for a tighter box
            patch_bgr = cv_image[y0:y1, x0:x1]
            tight_box = self._refine_box_with_edges(patch_bgr, x0, y0)
            if tight_box:
                bx0, by0, bx1, by1 = tight_box
            else:
                bx0, by0, bx1, by1 = x0, y0, x1, y1

            cx_px = (bx0 + bx1) // 2
            cy_px = (by0 + by1) // 2
            depth_m = self._sample_depth(depth_image, cx_px, cy_px, h, w)
            pos = self._pixel_to_robot(cx_px, cy_px, depth_m) if depth_m else None

            detections.append({
                'label':                clean_label,
                'confidence':           round(conf, 4),
                'mobility_weight':      MOBILITY_WEIGHTS.get(clean_label, 0.5),
                'depth_m':              round(depth_m, 3) if depth_m else None,
                'position_robot_frame': pos,
                'bbox':                 [bx0, by0, bx1, by1],
                'source':               'clip',
            })

        return detections

    def _refine_box_with_edges(self, patch_bgr, ox, oy):
        """Find the largest contour in the patch and return its bbox in full-image coords."""
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # Take the largest contour that covers at least 5% of the patch
        ph, pw = patch_bgr.shape[:2]
        min_area = ph * pw * 0.05
        valid = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid:
            return None
        largest = max(valid, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        # Convert to full-image coordinates
        return (ox + x, oy + y, ox + x + bw, oy + y + bh)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_depth(self, depth_image, cx_px, cy_px, h, w, patch=15):
        if depth_image is None:
            return None
        y0 = max(0, cy_px - patch); y1 = min(h, cy_px + patch)
        x0 = max(0, cx_px - patch); x1 = min(w, cx_px + patch)
        patch_data = depth_image[y0:y1, x0:x1].astype(np.float32)
        valid = patch_data[np.isfinite(patch_data) & (patch_data > 0)]
        if len(valid) == 0:
            return None
        raw = float(np.median(valid))
        # Gazebo publishes depth as float32 metres (32FC1); real D435 uses uint16 mm (16UC1)
        depth_m = raw / 1000.0 if depth_image.dtype == np.uint16 else raw
        return depth_m if 0.1 < depth_m < 10.0 else None

    def _pixel_to_robot(self, cx_px, cy_px, depth_m):
        x_cam = (cx_px - self.cx) * depth_m / self.fx
        y_cam = (cy_px - self.cy) * depth_m / self.fy
        z_cam = depth_m
        return [round(z_cam, 3), round(-x_cam, 3), round(CAMERA_HEIGHT_M - y_cam, 3)]

    def _publish_detections(self, detections):
        msg = String()
        msg.data = json.dumps({'timestamp': time.time(), 'detections': detections})
        self.detections_pub.publish(msg)
        for d in detections:
            if d.get('source') == 'clip':
                continue  # CLIP logs from worker thread when freshly computed
            dist = f"{d['depth_m']:.2f}m" if d['depth_m'] else '?'
            self.get_logger().info(
                f"[COLOR] {d['label']} conf={d['confidence']:.2f} dist={dist}")

    def _publish_annotated(self, cv_image, detections):
        annotated = cv_image.copy()
        src_color = {'color': (0, 255, 0), 'clip': (0, 165, 255)}
        for det in detections:
            color = src_color.get(det.get('source', 'clip'), (0, 165, 255))
            dist = f" @{det['depth_m']:.1f}m" if det['depth_m'] else ''
            text = f"{det['label']} {det['confidence']:.2f}{dist}"

            if 'bbox' in det:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            else:
                continue

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = SemanticDetectorCLIP()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
