#!/usr/bin/env python3
"""
Habitat-Sim ZMQ publisher.
Runs in 'habitat' conda env (Python 3.9).

Usage:
    conda activate habitat
    python3 ~/habitat_bridge_pub.py

Controls (terminal, single keypress):
    w = forward   s = backward
    a = turn left d = turn right
    r = reset     q = quit
"""
import glob, json, os, sys, time, threading
import numpy as np
import zmq
import habitat_sim

# ── Config ────────────────────────────────────────────────────────────────────
ZMQ_PORT   = 5555
IMG_W      = 640
IMG_H      = 480
CAM_HEIGHT = 0.8      # metres
BASELINE   = 0.06     # stereo baseline [m] — matches Replica stereo setup (replica-imap-stereo.py)
STEP_M     = 0.07     # 0.20 caused RANSAC failure (3 m/s @ 15 Hz > OKVIS tracking limit)
TURN_DEG   = 5.0
HFOV_DEG   = 78.7     # → focal_length ≈ 390.6 px at 640×480, matches rsD455 okvis2.yaml
PUBLISH_HZ = 20

_DATA = os.path.expanduser('~/habitat_data/scene_datasets')

# HM3D dataset-level config (provides navmesh + semantic annotations)
_HM3D_CFG = os.path.join(_DATA, 'hm3d/hm3d_annotated_basis.scene_dataset_config.json')

def _glb_in(scene_dir):
    """Return first .glb found in scene_dir, or None."""
    hits = glob.glob(os.path.join(scene_dir, '*.glb'))
    return hits[0] if hits else None

def _find_scene():
    # ── HM3D train 00033 (target scene) ──────────────────────────────────────
    preferred_dir = os.path.join(_DATA, 'hm3d/train/00033-oPj9qMxrDEa')
    glb = _glb_in(preferred_dir)
    if glb:
        cfg = _HM3D_CFG if os.path.exists(_HM3D_CFG) else None
        return cfg, glb

    # ── HM3D minival fallback ─────────────────────────────────────────────────
    for minival_dir in glob.glob(os.path.join(_DATA, 'hm3d/minival/008*')):
        glb = _glb_in(minival_dir)
        if glb:
            cfg = _HM3D_CFG if os.path.exists(_HM3D_CFG) else None
            return cfg, glb

    # ── habitat-test-scenes fallback (plain .glb, no dataset config) ─────────
    for name in ('apartment_1.glb', 'van-gogh-room.glb', 'skokloster-castle.glb'):
        path = os.path.join(_DATA, 'habitat-test-scenes', name)
        if os.path.exists(path):
            return None, path

    raise FileNotFoundError(
        'No Habitat scene data found.\n'
        'Download habitat-test-scenes (no auth):\n'
        '  python3 -m habitat_sim.utils.datasets_download '
        '--uids habitat_test_scenes --data-path ~/habitat_data\n\n'
        'Or run: python3 ~/download_hm3d_scene.py'
    )

SCENE_DATASET_CFG, SCENE_ID = _find_scene()
print(f'[cfg]   dataset : {SCENE_DATASET_CFG or "(none — plain glb)"}')
print(f'[cfg]   scene   : {SCENE_ID}')

# ── Build simulator ───────────────────────────────────────────────────────────
def build_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    if SCENE_DATASET_CFG:
        sim_cfg.scene_dataset_config_file = SCENE_DATASET_CFG
    sim_cfg.scene_id       = SCENE_ID
    sim_cfg.enable_physics = False
    sim_cfg.allow_sliding  = False

    def _cam(uuid, sensor_type, x_offset=0.0):
        s = habitat_sim.CameraSensorSpec()
        s.uuid        = uuid
        s.sensor_type = sensor_type
        s.resolution  = [IMG_H, IMG_W]
        s.position    = [x_offset, CAM_HEIGHT, 0.0]
        s.hfov        = HFOV_DEG
        return s

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        _cam('color',       habitat_sim.SensorType.COLOR),           # cam0 left
        _cam('color_right', habitat_sim.SensorType.COLOR, BASELINE), # cam1 right (6 cm)
        _cam('depth',       habitat_sim.SensorType.DEPTH),           # depth at cam0
    ]
    agent_cfg.action_space = {
        'move_forward': habitat_sim.agent.ActionSpec(
            'move_forward', habitat_sim.agent.ActuationSpec(amount=STEP_M)),
        'turn_left':  habitat_sim.agent.ActionSpec(
            'turn_left',  habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
        'turn_right': habitat_sim.agent.ActionSpec(
            'turn_right', habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
    }

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    nm = habitat_sim.NavMeshSettings()
    nm.agent_height = CAM_HEIGHT
    nm.agent_radius = 0.18
    nm.include_static_objects = True
    sim.recompute_navmesh(sim.pathfinder, nm)

    if sim.pathfinder.is_loaded:
        state = sim.agents[0].get_state()
        state.position = sim.pathfinder.get_random_navigable_point()
        sim.agents[0].set_state(state)
    else:
        print('[warn] NavMesh not loaded — agent starts at scene origin')

    return sim

# ── Keyboard (raw terminal, single keypress, non-blocking via thread) ─────────
def start_keyboard(action, quit_flag):
    import tty, termios
    def _run():
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not quit_flag[0]:
                ch = sys.stdin.read(1)
                if   ch == 'w': action[0] = 'move_forward'
                elif ch == 'a': action[0] = 'turn_left'
                elif ch == 'd': action[0] = 'turn_right'
                elif ch == 's': action[0] = 'backward'
                elif ch == 'r': action[0] = 'reset'
                elif ch == 'q': quit_flag[0] = True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    threading.Thread(target=_run, daemon=True).start()

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    import quaternion as Q

    print('Building Habitat-Sim...')
    sim = build_sim()
    print(f'Sim ready. Publishing on tcp://*:{ZMQ_PORT}')

    ctx  = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f'tcp://*:{ZMQ_PORT}')
    # Give subscribers time to connect
    time.sleep(0.5)

    action    = [None]
    quit_flag = [False]
    start_keyboard(action, quit_flag)
    print('Controls: w=fwd  a=left  d=right  s=back  r=reset  q=quit\n')

    dt       = 1.0 / PUBLISH_HZ
    frame_id = 0

    while not quit_flag[0]:
        t0  = time.time()
        act = action[0]
        action[0] = None

        if act == 'reset':
            state = sim.agents[0].get_state()
            if sim.pathfinder.is_loaded:
                state.position = sim.pathfinder.get_random_navigable_point()
            sim.agents[0].set_state(state)
            obs = sim.get_sensor_observations()
        elif act == 'backward':
            state = sim.agents[0].get_state()
            fwd   = Q.rotate_vectors(state.rotation, np.array([0.0, 0.0, -1.0]))
            fwd[1] = 0.0
            fwd   /= np.linalg.norm(fwd) + 1e-8
            new_pos = state.position - fwd * STEP_M
            if not sim.pathfinder.is_loaded or sim.pathfinder.is_navigable(new_pos):
                state.position = new_pos
                sim.agents[0].set_state(state)
            obs = sim.get_sensor_observations()
        elif act in ('move_forward', 'turn_left', 'turn_right'):
            obs = sim.step(act)
        else:
            obs = sim.get_sensor_observations()

        # RGB: RGBA → BGR
        bgr       = obs['color'][:, :, :3][:, :, ::-1].copy()        # uint8 HxWx3 left
        bgr_right = obs['color_right'][:, :, :3][:, :, ::-1].copy()  # uint8 HxWx3 right
        depth     = obs['depth'].astype(np.float32)                   # float32 HxW metres

        # Ground-truth pose for synthetic IMU
        ag    = sim.agents[0].get_state()
        pos   = ag.position.tolist()                        # [x, y, z] habitat Y-up
        rot   = ag.rotation                                 # numpy-quaternion
        quat  = [float(rot.w), float(rot.x),
                 float(rot.y), float(rot.z)]

        ts      = time.time()
        moving  = act in ('move_forward', 'turn_left', 'turn_right', 'backward')
        header  = json.dumps({'t': ts, 'w': IMG_W, 'h': IMG_H, 'fid': frame_id,
                              'pos': pos, 'quat': quat, 'moving': moving}).encode()
        sock.send_multipart([header, bgr.tobytes(), bgr_right.tobytes(), depth.tobytes()])

        frame_id += 1
        elapsed = time.time() - t0
        if frame_id % PUBLISH_HZ == 0:
            fps = 1.0 / max(elapsed, 1e-6)
            print(f'\r[Habitat publisher] frame={frame_id}  {fps:.1f} FPS  ', end='', flush=True)

        sleep_t = dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    sim.close()
    sock.close()
    ctx.term()
    print('\nDone.')

if __name__ == '__main__':
    main()
