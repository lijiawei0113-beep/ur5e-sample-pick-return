
from __future__ import annotations
import argparse
import json
import math
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
VERSION = 'HANDEYE_TOUCH_SINGLE_V3_2_TARGET_X_PLUS_2MM'
STOPPED_END_GRACE_SECONDS = 3.0
WINDOW = 'Fresh target selection / final verification'
SERIAL = 'CV2L7610016A'
WIDTH, HEIGHT, FPS = (848, 530, 15)
PATTERN = (6, 8)
SQUARE_M = 0.0255
K = np.array([[409.38232421875, 0, 421.5352783203125],
              [0, 409.3629455566406, 265.9438171386719],
              [0, 0, 1]], dtype=np.float64)
D = np.array([-1.2459391355514526, 0.8174504637718201,
              0.0002063319116132334, 6.101600592955947e-05,
              -0.1829749494791031, -1.2284886837005615,
              0.7941957116127014, -0.17399488389492035], dtype=np.float64)
CAMERA = {"serial_number": SERIAL, "width": WIDTH, "height": HEIGHT,
          "fps": FPS, "format": "YUYV", "K": K.tolist(),
          "D_opencv": D.tolist(), "D_order": "k1,k2,p1,p2,k3,k4,k5,k6",
          "source": "user-exported Orbbec factory calibration; raw color"}
STILL_SECONDS = 2.0
MAX_DRIFT_M = 0.00025
MAX_DRIFT_RAD = np.deg2rad(0.05)
MAX_LINEAR_SPEED = 0.0005
MAX_ANGULAR_SPEED = 0.003
MAX_RMSE = 0.8
RTDE_STALE_TIMEOUT_SECONDS = 5.0
H = np.array([
    [-.9997923621139053,-.01727803275277129,.010802881231019579,.010020656535130776],
    [.017295984865174023,-.9998491797343178,.0015705706463489122,.07488965600334786],
    [.010774115566534084,.0017570910066519174,.9999404137572161,.058478193092896445],
    [0,0,0,1]], dtype=float)
TIP = np.array([0.0, 0.0, 0.2456])
TARGET_BASE_X_OFFSET_M = 0.002
IP = '192.168.1.102'


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError('无效的机器人位姿')
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(pose[3:])[0]
    T[:3, 3] = pose[:3]
    return T

def pose_difference(a, b):
    shift = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    angle = float(np.linalg.norm(cv2.Rodrigues(a[:3, :3].T @ b[:3, :3])[0]))
    return (shift, angle)

class RobotMonitor:
    """独立采样，避免角点检测期间漏掉机器人运动或断流。"""

    def __init__(self, receiver):
        self.receiver = receiver
        self.history = deque(maxlen=4000)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        previous = -1.0
        last_new = time.monotonic()
        try:
            while not self.stop_event.is_set():
                if not self.receiver.isConnected():
                    raise RuntimeError('RTDE 已断开')
                t0 = float(self.receiver.getTimestamp())
                pose = np.asarray(self.receiver.getActualTCPPose(), dtype=np.float64)
                speed = np.asarray(self.receiver.getActualTCPSpeed(), dtype=np.float64)
                t1 = float(self.receiver.getTimestamp())
                now = time.monotonic()
                if (
                    t0 == t1 and t1 > previous
                    and pose.shape == (6,) and speed.shape == (6,)
                    and np.isfinite(pose).all() and np.isfinite(speed).all()
                ):
                    state = {'host_s': now, 'controller_s': t1, 'pose': pose, 'speed': speed, 'T': pose_matrix(pose)}
                    with self.lock:
                        self.history.append(state)
                    previous, last_new = (t1, now)
                if now - last_new > RTDE_STALE_TIMEOUT_SECONDS:
                    raise RuntimeError(f'超过{RTDE_STALE_TIMEOUT_SECONDS:g}秒未收到有效的新机器人数据')
                self.stop_event.wait(0.01)
        except Exception as exc:
            self.error = exc

    def snapshot(self):
        if self.error is not None:
            raise RuntimeError(f'机器人读取失败：{self.error}') from self.error
        with self.lock:
            return list(self.history)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)

def stable_window(history, start, end):
    """覆盖整个窗口，检查数据间断、速度和相对旋转，返回采样及统计。"""
    before = [i for i, s in enumerate(history) if s['host_s'] <= start]
    if not before or not history or history[-1]['host_s'] < end - 0.1:
        return None
    states = [s for s in history[before[-1]:] if s['host_s'] <= end]
    if len(states) < 3 or states[-1]['host_s'] < end - 0.1:
        return None
    if np.max(np.diff([s['host_s'] for s in states])) > 0.15:
        return None
    if np.max(np.diff([s['controller_s'] for s in states])) > 0.15:
        return None
    reference = states[0]['T']
    differences = np.array([pose_difference(reference, s['T']) for s in states])
    translation, rotation = differences.max(axis=0)
    linear = max((np.linalg.norm(s['speed'][:3]) for s in states))
    angular = max((np.linalg.norm(s['speed'][3:]) for s in states))
    if translation > MAX_DRIFT_M or rotation > MAX_DRIFT_RAD or linear > MAX_LINEAR_SPEED or (angular > MAX_ANGULAR_SPEED):
        return None
    stats = {"duration_s": states[-1]["host_s"] - states[0]["host_s"],
             "max_translation_from_start_m": float(translation),
             "max_rotation_from_start_rad": float(rotation),
             "max_linear_speed_m_s": float(linear),
             "max_angular_speed_rad_s": float(angular)}
    return (states, stats)

def show(image, message):
    view = image.copy()
    cv2.putText(view, message, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)
    cv2.putText(view, 'Q/Esc: quit', (8, HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow(WINDOW, view)
    key = cv2.waitKey(1) & 255
    return key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1

def is_duplicate(T, previous):
    return any((d < 0.01 and a < np.deg2rad(4.0) for d, a in (pose_difference(T, old) for old in previous)))

def capture_candidate(pipeline, monitor, previous, yuyv_format):
    started = last_frame = time.monotonic()
    seen = 0
    while True:
        frames = pipeline.wait_for_frames(500)
        frame = frames.get_color_frame() if frames is not None else None
        received = time.monotonic()
        history = monitor.snapshot()
        if frame is None:
            if received - last_frame > 10:
                raise RuntimeError('10秒未收到彩色帧，请关闭 Viewer 并检查相机连接')
            if cv2.waitKey(1) & 255 in (27, ord('q'), ord('Q')):
                return None
            continue
        last_frame = received
        seen += 1
        if (frame.get_width(), frame.get_height(), frame.get_format()) != (WIDTH, HEIGHT, yuyv_format):
            raise RuntimeError('彩色帧配置不是 848x530 YUYV')
        raw = np.frombuffer(frame.get_data(), dtype=np.uint8).reshape(HEIGHT, WIDTH, 2)
        bgr = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
        checked = stable_window(history, received - STILL_SECONDS, received)
        if seen < 30 or received - started < 2.0:
            message = 'Camera warming up...'
        elif checked is None:
            message = 'Hold robot STILL for 2 seconds'
        elif is_duplicate(checked[0][-1]['T'], previous):
            message = 'Pose already collected; move to a different pose'
        else:
            found, corners = cv2.findChessboardCornersSB(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), PATTERN, cv2.CALIB_CB_NORMALIZE_IMAGE)
            if found and len(corners) == 48:
                xy = corners.reshape(-1, 2)
                if xy[:, 0].min() < 10 or xy[:, 0].max() > WIDTH - 10 or xy[:, 1].min() < 10 or (xy[:, 1].max() > HEIGHT - 10):
                    message = 'Move board away from image edges'
                else:
                    end_guard = time.monotonic() + 0.25
                    while time.monotonic() < end_guard:
                        if show(bgr, 'Checking robot stability...'):
                            return None
                        time.sleep(0.015)
                    checked = stable_window(monitor.snapshot(), received - STILL_SECONDS, time.monotonic())
                    if checked is not None:
                        states, stats = checked
                        nearest = min(states, key=lambda s: abs(s['host_s'] - received))
                        return {"bgr": bgr, "corners": corners,
                                "image_received_host_s": received,
                                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                                "robot": nearest, "states": states, "stability": stats}
                    message = 'Robot moved; retrying...'
            else:
                message = 'Need all 48 inner corners visible'
        if show(bgr, message):
            return None

def recv_exact(sock, size, deadline):
    data = bytearray()
    while len(data) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Robot data timed out.')
        sock.settimeout(min(2.0, remaining))
        try:
            chunk = sock.recv(size - len(data))
        except socket.timeout:
            continue
        if not chunk:
            raise ConnectionError('Robot disconnected.')
        data.extend(chunk)
    return bytes(data)

def message(sock, deadline):
    length, kind = struct.unpack('!IB', recv_exact(sock, 5, deadline))
    if not 5 <= length <= 2000000:
        raise ValueError('Invalid robot message length.')
    return (kind, recv_exact(sock, length - 5, deadline))

def parse_state(payload):
    state, pos = ({}, 0)
    while pos < len(payload):
        if len(payload) - pos < 5:
            raise ValueError('Truncated robot state.')
        size, kind = struct.unpack_from('!IB', payload, pos)
        if size < 5 or pos + size > len(payload):
            raise ValueError('Invalid robot subpackage.')
        body = payload[pos + 5:pos + size]
        if kind == 0:
            if len(body) < 41:
                raise ValueError('Robot mode data is incomplete.')
            data = struct.unpack_from('!Q7?BBddd', body)
            names = ["timestamp", "connected", "enabled", "power", "emergency", "protective",
                     "running", "paused", "robot_mode", "control_mode", "slider", "scaling", "slider_limit"]
            state.update(dict(zip(names, data)))
        elif kind == 4:
            if len(body) < 96:
                raise ValueError('TCP-offset data is missing.')
            values = struct.unpack_from('!12d', body)
            state['pose'], state['offset'] = (list(values[:6]), list(values[6:]))
        pos += size
    return state

def verify_idle(state):
    required = {"timestamp", "connected", "enabled", "power", "emergency", "protective",
                "running", "paused", "robot_mode", "control_mode", "slider", "pose", "offset"}
    if not required <= state.keys():
        raise ValueError('Incomplete robot state. No movement allowed.')
    if len(state['pose']) != 6 or len(state['offset']) != 6:
        raise ValueError('Invalid pose/offset length.')
    if not all((math.isfinite(x) for x in state['pose'] + state['offset'] + [state['slider']])):
        raise ValueError('Invalid/non-finite robot state.')
    if not all((state[k] for k in ('connected', 'enabled', 'power'))):
        raise ValueError('Real robot must be enabled and powered.')
    if any((state[k] for k in ('emergency', 'protective', 'running', 'paused'))):
        raise ValueError('Safety stop or another program is active. No reset or preemption.')
    if state['robot_mode'] != 7 or state['control_mode'] != 0:
        raise ValueError('Robot must be idle in position control, not Freedrive/force mode.')
    if not 0.1 <= state['slider'] <= 1:
        raise ValueError('Set slider to at least 10%; commanded speeds are separately capped.')
    if max((abs(x) for x in state['offset'])) > 1e-06:
        raise ValueError('Active TCP is not ZERO. Do not change it to bypass this check.')

def read_ready(ip):
    samples = []
    with socket.create_connection((ip, 30012), timeout=5) as sock:
        deadline = time.monotonic() + 10
        while len(samples) < 10:
            kind, data = message(sock, deadline)
            if kind != 16:
                continue
            s = parse_state(data)
            if 'pose' not in s or 'timestamp' not in s:
                continue
            verify_idle(s)
            if samples and s['timestamp'] <= samples[-1]['timestamp']:
                raise ValueError('Robot stream is stale.')
            samples.append(s)
    reference = pose_matrix(samples[0]['pose'])
    for s in samples:
        d, a = pose_difference(reference, pose_matrix(s['pose']))
        if d > 0.0001 or a > math.radians(0.03):
            raise ValueError('Robot is not stationary.')
    return (samples[-1], samples)

def require_remote_control(ip):
    """Read-only Dashboard inquiry. Never switch the controller mode here."""
    with socket.create_connection((ip, 29999), timeout=4) as sock:
        sock.settimeout(4)
        with sock.makefile('r', encoding='utf-8', errors='replace', newline='\n') as stream:
            greeting = stream.readline().strip()
            if not greeting:
                raise RuntimeError('Dashboard did not send its greeting; no robot motion.')
            sock.sendall(b'is in remote control\n')
            response = stream.readline().strip().lower()
    print('Dashboard remote control:', response, flush=True)
    if response == 'false' or response.endswith(': false'):
        raise RuntimeError('Robot is in Local Control: port 30001 is read-only (C210A0). '
                           'Switch to Remote Control on the pendant, then start a NEW capture.')
    if response != 'true' and (not response.endswith(': true')):
        raise RuntimeError(f'Cannot verify Remote Control state from Dashboard: {response!r}. '
                           'No robot motion was submitted.')

def transform_ok(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('Invalid transform.')
    R = T[:3, :3]
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-10) or not np.allclose(R.T @ R, np.eye(3), atol=1e-07) or abs(np.linalg.det(R) - 1) > 1e-07:
        raise ValueError('Transform is not a proper rigid transform.')
    return T

def object_points(square):
    if not math.isfinite(square) or not 0.005 <= square <= 0.1:
        raise ValueError('Invalid square length (meters).')
    obj = np.zeros((48, 3), dtype=float)
    obj[:, :2] = np.mgrid[0:6, 0:8].T.reshape(-1, 2) * square
    return obj

def solve_board(corners, square):
    obj = object_points(square)
    corners = np.asarray(corners, dtype=float).reshape(-1, 1, 2)
    if corners.shape != (48, 1, 2) or not np.isfinite(corners).all():
        raise ValueError('Need 48 valid inner corners.')
    ok, rv, tv = cv2.solvePnP(obj, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError('Board pose solver failed.')
    T = transform_ok(pose_matrix(np.r_[tv.ravel(), rv.ravel()]))
    if np.any((obj @ T[:3, :3].T + T[:3, 3])[:, 2] <= 0):
        raise ValueError('Invalid board depth.')
    reproj = cv2.projectPoints(obj, rv, tv, K, D)[0].reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((reproj - corners.reshape(-1, 2)) ** 2, axis=1))))
    if not math.isfinite(rmse) or rmse > 0.5:
        raise ValueError(f'Board reprojection RMSE {rmse:.3f}px exceeds 0.5px; recapture.')
    return (T, rmse)

def build_plan(start_pose, T_camera_board, target_index, square):
    start = np.asarray(start_pose, dtype=float)
    if start.shape != (6,) or not np.isfinite(start).all():
        raise ValueError('Invalid capture pose.')
    if isinstance(target_index, bool) or int(target_index) != target_index or (not 0 <= target_index < 48):
        raise ValueError('Select an inner-corner index from 0 to 47.')
    B = pose_matrix(start)
    C = B @ transform_ok(H)
    W = C @ transform_ok(T_camera_board)
    point = (W @ np.r_[object_points(square)[target_index], 1])[:3]
    n = W[:3, 2].copy()
    if np.dot(n, C[:3, 3] - point) < 0:
        n = -n
    n /= np.linalg.norm(n)
    tip = start[:3] + B[:3, :3] @ TIP
    gap = float(np.dot(tip - point, n))
    tilt = math.degrees(math.acos(np.clip(np.dot(B[:3, 2], -n), -1, 1)))
    if not 0.05 <= gap <= 0.3:
        raise ValueError(f'Start tip clearance {gap * 1000:.1f}mm: need a clear 50-300mm observation pose, NOT a near-board pose.')
    if tilt > 45:
        raise ValueError(f'Tool tilt {tilt:.1f}deg: this fixed-orientation version supports <=45deg toward the board.')
    offset = B[:3, :3] @ TIP
    nominal_plane = np.r_[point - offset, start[3:]]
    plane = nominal_plane.copy()
    plane[0] += TARGET_BASE_X_OFFSET_M
    lateral = float(np.linalg.norm(plane[:2] - start[:2]))
    if lateral > 0.3:
        raise ValueError('Lateral transfer exceeds 300mm; no movement planned.')
    return {"schema":"direct_target_pose_from_fresh_view_v1","target_corner_index":int(target_index),
            "square_size_m":square,"T_tcp_camera":H.tolist(),"T_base_board":W.tolist(),
            "tip_offset_tcp_m":TIP.tolist(),"tip_offset_is_nominal":True,
            "start_pose_m_rad":start.tolist(),"target_point_base_m":point.tolist(),
            "target_base_x_offset_m":TARGET_BASE_X_OFFSET_M,
            "nominal_target_pose_m_rad":nominal_plane.tolist(),
            "commanded_tip_point_base_m":(point+np.array([TARGET_BASE_X_OFFSET_M,0.,0.])).tolist(),
            "outward_normal_base":n.tolist(),"start_tip_clearance_mm":gap*1000,
            "tool_tilt_deg":tilt,"lateral_transfer_mm":lateral*1000,
            "target_pose_m_rad":plane.tolist(),
            "orientation_policy":"hold captured orientation; no wrist rotation",
            "path_collision_checked":False,"accuracy_measured":False}

def alignment_plan(candidate, board, square):
    """Aim the optical ray through the board centre, with the tool normal to it."""
    start = np.asarray(candidate['robot']['pose'], dtype=float)
    B = pose_matrix(start)
    C = B @ H
    W = C @ transform_ok(board)
    centre = (W @ np.r_[np.array([2.5, 3.5, 0]) * square, 1])[:3]
    normal = W[:3, 2].copy()
    if np.dot(normal, C[:3, 3] - centre) < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.94:
        raise ValueError('The board plane is too tilted for this lift/level procedure.')
    z_axis = -normal
    x_axis = B[:3, 0] - np.dot(B[:3, 0], z_axis) * z_axis
    if np.linalg.norm(x_axis) < 0.1:
        raise ValueError('Cannot preserve the tool yaw while aligning to the board.')
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack((x_axis, y_axis, z_axis))
    optical = R @ H[:3, :3] @ np.array([0.0, 0.0, 1.0])
    normal_cos = -float(np.dot(optical, normal))
    if normal_cos < 0.995:
        raise ValueError('Camera/tool axes disagree: check the embedded hand-eye calibration.')
    clearance = 0.34
    cam_xyz = centre - clearance / normal_cos * optical
    view_xyz = cam_xyz - R @ H[:3, 3]
    view = np.r_[view_xyz, cv2.Rodrigues(R)[0].ravel()]
    start_camera_distance = float(np.dot(C[:3, 3] - centre, normal))
    if not 0.15 <= start_camera_distance <= 0.7:
        raise ValueError('Start camera clearance outside 150–700 mm; do not auto-align.')
    if view[2] < 0.32 or view[2] > 0.65:
        raise ValueError(f'Computed observation TCP Z {view[2] * 1000:.1f} mm outside 320–650 mm.')
    camera_T = pose_matrix(view) @ H
    corners = object_points(square)
    camera_points = (camera_T[:3, :3].T @ (W[:3, :3] @ corners.T + W[:3, 3:4] - camera_T[:3, 3:4])).T
    if np.any(camera_points[:, 2] <= 0.15):
        raise ValueError('Predicted board would lie behind or too close to camera.')
    uv, _ = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), K, D)
    uv = uv.reshape(-1, 2)
    bbox = [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]
    if bbox[0] < 25 or bbox[1] < 25 or bbox[2] > WIDTH - 25 or (bbox[3] > HEIGHT - 25):
        raise ValueError(f'Board predicted outside the new camera view: {bbox}')
    lift_z = max(float(start[2]) + 0.12, float(view[2]) + 0.12, float(centre[2]) + float(TIP[2]) + 0.18)
    if lift_z > 0.75:
        raise ValueError(f'Lift TCP Z {lift_z * 1000:.1f} mm exceeds configured limit.')
    lift = start.copy()
    lift[2] = lift_z
    turn = lift.copy()
    turn[3:] = view[3:]
    level = view.copy()
    level[2] = lift_z
    if np.linalg.norm(level[:2] - lift[:2]) > 0.35:
        raise ValueError('Alignment lateral translation exceeds 350 mm.')
    poses = [start, lift, turn, level, view]
    min_tip_gap = float('inf')
    for a, b in zip(poses[:-1], poses[1:]):
        A, B_next = (pose_matrix(a), pose_matrix(b))
        relative = cv2.Rodrigues(A[:3, :3].T @ B_next[:3, :3])[0].ravel()
        for fraction in np.linspace(0, 1, 51):
            xyz = a[:3] + fraction * (b[:3] - a[:3])
            rot = A[:3, :3] @ cv2.Rodrigues(relative * fraction)[0]
            gap = float(np.dot(xyz + rot @ TIP - centre, normal))
            min_tip_gap = min(min_tip_gap, gap)
    if min_tip_gap < 0.05:
        raise ValueError(f'Predicted probe clearance {min_tip_gap * 1000:.1f} mm < 50 mm.')
    return {'start_pose_m_rad': start.tolist(),
            'waypoints_m_rad': [p.tolist() for p in poses[1:]],
            'board_center_base_m': centre.tolist(),
            'board_normal_base': normal.tolist(),
            'camera_clearance_m': clearance,
            'minimum_predicted_probe_clearance_mm': min_tip_gap*1000,
            'predicted_board_bbox_px': bbox,
            'predicted_optical_axis_dot_board_inward': normal_cos,
            'path_collision_checked': False}

def capture(ip):
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    from rtde_receive import RTDEReceiveInterface
    before, _ = read_ready(ip)
    receiver = monitor = pipeline = None
    started = False
    try:
        receiver = RTDEReceiveInterface(ip, 125.0, ['timestamp', 'actual_TCP_pose', 'actual_TCP_speed'])
        monitor = RobotMonitor(receiver)
        monitor.thread.start()
        pipeline = Pipeline()
        serial = pipeline.get_device().get_device_info().get_serial_number()
        if serial != SERIAL:
            raise ValueError('Camera serial does not match the embedded factory intrinsics.')
        profile=pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(
            WIDTH,HEIGHT,OBFormat.YUYV,FPS)
        cfg = Config()
        cfg.enable_stream(profile)
        pipeline.start(cfg)
        started = True
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        candidate = capture_candidate(pipeline, monitor, [], OBFormat.YUYV)
        if candidate is None:
            return None
        after, _ = read_ready(ip)
        for pose in (before['pose'], after['pose']):
            d, a = pose_difference(pose_matrix(pose), candidate['robot']['T'])
            if d > 0.00015 or a > math.radians(0.04):
                raise ValueError('Robot moved during capture; restart from a stationary observation pose.')
        candidate['verified_offset'] = after['offset']
        return candidate
    finally:
        try:
            if monitor is not None:
                monitor.stop()
        finally:
            try:
                if receiver is not None:
                    receiver.disconnect()
            finally:
                if started:
                    pipeline.stop()

def capture_worker(ip, output):
    """One SDK/RTDE camera session per process; return no target on Q/Esc."""
    print('CAMERA_WORKER: opening one camera session...', flush=True)
    candidate = capture(ip)
    if candidate is None:
        return 4
    print('CAMERA_WORKER: frame captured; saving...', flush=True)
    np.savez_compressed(
        output,
        bgr=candidate['bgr'], corners=candidate['corners'],
        pose=candidate['robot']['pose'], T=candidate['robot']['T'],
        controller_s=np.array(candidate['robot']['controller_s']),
        verified_offset=np.array(candidate['verified_offset']),
        captured_at_utc=np.array(candidate['captured_at_utc']),
        stability_json=np.array(json.dumps(candidate['stability'], allow_nan=False)))
    print('CAMERA_WORKER: frame ready.', flush=True)
    return 0

def capture_isolated(ip):
    """Do not reopen Orbbec/RTDE/CV capture in the motion control process."""
    with tempfile.TemporaryDirectory(prefix='handeye_capture_') as temporary:
        output = Path(temporary) / 'frame.npz'
        print('Starting isolated camera capture...', flush=True)
        result=subprocess.run([sys.executable, str(Path(__file__).resolve()),
                               '--capture-worker', ip, str(output)], check=False)
        if result.returncode == 4:
            return None
        if result.returncode != 0:
            code = f'0x{result.returncode & 4294967295:08X}'
            raise RuntimeError(f'Camera worker exited abnormally ({code}); '
                               'robot motion will not continue. Check camera/RTDE drivers.')
        if not output.is_file():
            raise RuntimeError('Camera worker returned without a frame; no motion.')
        with np.load(output, allow_pickle=False) as data:
            candidate={'bgr':data['bgr'].copy(), 'corners':data['corners'].copy(),
                    'robot':{'pose':data['pose'].copy(), 'T':data['T'].copy(),
                             'controller_s':float(data['controller_s'])},
                    'verified_offset':data['verified_offset'].tolist(),
                    'captured_at_utc':str(data['captured_at_utc']),
                    'stability':json.loads(str(data['stability_json']))}
        print('Isolated camera frame loaded for target selection.', flush=True)
        return candidate

def select_target(candidate):
    points = candidate['corners'].reshape(-1, 2)
    selected = []

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            distance = np.linalg.norm(points - [x, y], axis=1)
            index = int(np.argmin(distance))
            if distance[index] <= 15:
                selected[:] = [index]
    cv2.setMouseCallback(WINDOW, mouse)
    try:
        while True:
            view = candidate['bgr'].copy()
            cv2.drawChessboardCorners(view, PATTERN, candidate['corners'], True)
            for i, xy in enumerate(points):
                cv2.putText(view,str(i),tuple(np.rint(xy+[3,-3]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,.3,(255,0,255),1)
            if selected:
                xy = tuple(np.rint(points[selected[0]]).astype(int))
                cv2.circle(view, xy, 11, (0, 0, 255), 2)
            cv2.putText(view,'FROZEN: click target corner; ENTER confirms; Q cancels',
                        (6,22),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,255,255),2)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(20) & 255
            if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in (10, 13) and selected:
                return (selected[0], view)
    finally:
        try:
            cv2.setMouseCallback(WINDOW, lambda *args: None)
        except cv2.error:
            pass

def controller_program(plan, run_id):
    """One straight, slow move to the computed TCP pose; contact is not inferred."""
    if not re.fullmatch('[a-f0-9]{8}', run_id):
        raise ValueError('Invalid run id.')

    def vec(v, pose=False):
        if not np.isfinite(v).all():
            raise ValueError('Non-finite program value.')
        return ('p' if pose else '') + '[' + ','.join((format(float(x), '.15g') for x in v)) + ']'
    # Keep the controller program separate from host-side monitoring.
    source = r'''def move_to_selected_target():
  def sub3(a,b):
    return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]
  end
  def dot3(a,b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
  end
  def rot_error(a,b):
    local rel=pose_trans(pose_inv(a),b)
    return norm([rel[3],rel[4],rel[5]])
  end
  def tcp_ok():
    local off=get_tcp_offset()
    return norm([off[0],off[1],off[2]])<0.000001 and norm([off[3],off[4],off[5]])<0.000001
  end
  def finish(status):
    textmsg("VIEW_@ID@|END|",status)
    halt
  end
  textmsg("VIEW_@ID@|START|","TARGET_POSE")
  expected=@START@
  goal=@GOAL@
  origin=@ORIGIN@
  n=@NORMAL@
  if not tcp_ok():
    finish("ABORT_TCP")
  end
  actual=get_actual_tcp_pose()
  if not is_steady() or norm(sub3(actual,expected))>0.00015 or rot_error(actual,expected)>0.0007:
    finish("ABORT_START_CHANGED")
  end
  q=get_actual_joint_positions()
  distance=norm(sub3(goal,expected))
  count=ceil(distance/0.002)+1
  j=1
  while j<=count:
    test_pose=interpolate_pose(expected,goal,j/count)
    tip=pose_trans(test_pose,p[0,0,@TIP_Z@,0,0,0])
    if dot3(sub3(tip,origin),n)< -0.0002:
      finish("ABORT_BELOW_NOMINAL_PLANE")
    end
    if not get_inverse_kin_has_solution(test_pose,qnear=q):
      finish("ABORT_PATH_IK")
    end
    q_next=get_inverse_kin(test_pose,qnear=q)
    if not is_within_safety_limits(q_next):
      finish("ABORT_PATH_SAFETY_LIMIT")
    end
    if sin(q_next[4])<0.0872 and sin(q_next[4])> -0.0872:
      finish("ABORT_NEAR_WRIST_SINGULARITY")
    end
    axis=0
    while axis<6:
      if norm(q_next[axis]-q[axis])>0.15:
        finish("ABORT_IK_BRANCH_JUMP")
      end
      axis=axis+1
    end
    q=q_next
    j=j+1
    sync()
  end
  if not is_steady() or not tcp_ok() or norm(sub3(get_actual_tcp_pose(),expected))>0.00015:
    finish("ABORT_CHANGED_DURING_CHECKS")
  end
  textmsg("VIEW_@ID@|STAGE|","MOVE_TO_NOMINAL_TARGET")
  movel(goal,a=0.05,v=0.010)
  actual=get_actual_tcp_pose()
  textmsg("VIEW_@ID@|STOP_POSE|",actual)
  if norm(sub3(actual,goal))>0.0003 or rot_error(actual,goal)>0.001:
    finish("ABORT_TARGET_NOT_REACHED")
  end
  finish("TARGET_POSE_REACHED_CONTACT_UNVERIFIED")
end
'''
    return (source.replace('@ID@',run_id)
            .replace('@START@',vec(plan['start_pose_m_rad'],True))
            .replace('@GOAL@',vec(plan['target_pose_m_rad'],True))
            .replace('@ORIGIN@',vec(plan['target_point_base_m']))
            .replace('@NORMAL@',vec(plan['outward_normal_base']))
            .replace('@TIP_Z@',format(float(plan['tip_offset_tcp_m'][2]),'.15g')))

def alignment_program(plan, run_id):
    if not re.fullmatch('[a-f0-9]{8}', run_id):
        raise ValueError('Invalid run id.')

    def pose_literal(values):
        return 'p[' + ','.join((format(float(v), '.15g') for v in values)) + ']'
    s = r'''def align_before_selection():
  def delta(a,b):
    return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]
  end
  def rot_error(a,b):
    local rel=pose_trans(pose_inv(a),b)
    return norm([rel[3],rel[4],rel[5]])
  end
  def tcp_ok():
    local off=get_tcp_offset()
    return norm([off[0],off[1],off[2]])<0.000001 and norm([off[3],off[4],off[5]])<0.000001
  end
  def finish(status):
    textmsg("VIEW_@ID@|END|",status)
    halt
  end
  textmsg("VIEW_@ID@|START|","ALIGN")
  expected=@START@
  goals=@GOALS@
  centre=@CENTRE@
  normal=@NORMAL@
  if not tcp_ok():
    finish("ABORT_TCP")
  end
  actual=get_actual_tcp_pose()
  if not is_steady() or norm(delta(actual,expected))>0.00015 or rot_error(actual,expected)>0.0007:
    finish("ABORT_START_CHANGED")
  end
  q=get_actual_joint_positions()
  previous=expected
  i=0
  while i<4:
    displacement=norm(delta(goals[i],previous))
    rotation=rot_error(goals[i],previous)
    count=ceil(displacement/0.003+rotation/0.03)+1
    j=1
    while j<=count:
      test_pose=interpolate_pose(previous,goals[i],j/count)
      tip=pose_trans(test_pose,p[0,0,@TIP_Z@,0,0,0])
      tip_delta=delta(tip,centre)
      if tip_delta[0]*normal[0]+tip_delta[1]*normal[1]+tip_delta[2]*normal[2]<0.05:
        finish("ABORT_PROBE_CLEARANCE")
      end
      if not get_inverse_kin_has_solution(test_pose,qnear=q):
        finish("ABORT_PATH_IK")
      end
      q_next=get_inverse_kin(test_pose,qnear=q)
      if not is_within_safety_limits(q_next):
        finish("ABORT_PATH_SAFETY_LIMIT")
      end
      if sin(q_next[4])<0.0872 and sin(q_next[4])> -0.0872:
        finish("ABORT_NEAR_WRIST_SINGULARITY")
      end
      axis=0
      while axis<6:
        if norm(q_next[axis]-q[axis])>0.15:
          finish("ABORT_IK_BRANCH_JUMP")
        end
        axis=axis+1
      end
      q=q_next
      j=j+1
      sync()
    end
    previous=goals[i]
    i=i+1
  end
  if not is_steady() or norm(delta(get_actual_tcp_pose(),expected))>0.00015:
    finish("ABORT_CHANGED_DURING_CHECKS")
  end
  i=0
  while i<4:
    if not tcp_ok():
      finish("ABORT_TCP_CHANGED")
    end
    textmsg("VIEW_@ID@|STAGE|",i)
    movel(goals[i],a=0.10,v=0.030)
    textmsg("VIEW_@ID@|STOP_POSE|",get_actual_tcp_pose())
    i=i+1
  end
  finish("ALIGN_OK")
end
'''
    return (s.replace('@ID@',run_id)
             .replace('@START@',pose_literal(plan['start_pose_m_rad']))
             .replace('@GOALS@','['+','.join(pose_literal(p) for p in plan['waypoints_m_rad'])+']')
             .replace('@CENTRE@','['+','.join(format(x,'.15g') for x in plan['board_center_base_m'])+']')
             .replace('@NORMAL@','['+','.join(format(x,'.15g') for x in plan['board_normal_base'])+']')
             .replace('@TIP_Z@',format(float(TIP[2]),'.15g')))

def submit(ip, script, run_id, folder, mode='TOUCH'):
    report={'submission_started':False,'program_sent':False,'messages':[],
            'outcome':'NOT_SUBMITTED','accuracy_measured':False,'version':VERSION,
            'mode':mode,'controller_messages':[],
            'controller_stopped_verified':False,'program_start_acknowledged':False}
    sock = None
    try:
        sock = socket.create_connection((ip, 30001), timeout=5)
        report['submission_started'] = True
        report['outcome'] = 'SUBMITTED_OUTCOME_UNCONFIRMED'
        sock.sendall(script.encode('ascii'))
        report['program_sent'] = True
        print(f'{mode} program submitted. Supervise using the pendant; Ctrl+C is NOT a safety stop.', flush=True)
        submitted_at = time.monotonic()
        deadline = submitted_at + 2100
        start_seen_at = end_seen_at = None
        stopped_before_end_at = None
        end_controller_time = None
        last_state = None
        stopped_packets = 0
        while True:
            now = time.monotonic()
            if start_seen_at is None and now - submitted_at > 5:
                raise RuntimeError('No program START acknowledgment within 5 seconds. Check controller log/Remote mode; no automatic retry.')
            if end_seen_at is not None and now - end_seen_at > 5:
                raise RuntimeError('END received but controller stop not verified within 5 seconds.')
            if stopped_before_end_at is not None and end_seen_at is None and (now - stopped_before_end_at >= STOPPED_END_GRACE_SECONDS):
                report['outcome'] = 'STOPPED_WITHOUT_END'
                raise RuntimeError('Stopped state received, but no END within 3 seconds. Inspect the controller log; no automatic retry.')
            read_deadline = min(deadline, submitted_at + 5) if start_seen_at is None else deadline
            if end_seen_at is not None:
                read_deadline = min(read_deadline, end_seen_at + 5)
            if stopped_before_end_at is not None and end_seen_at is None:
                read_deadline = min(read_deadline, stopped_before_end_at + STOPPED_END_GRACE_SECONDS)
            try:
                kind, payload = message(sock, read_deadline)
            except TimeoutError as exc:
                if stopped_before_end_at is not None and end_seen_at is None:
                    report['outcome'] = 'STOPPED_WITHOUT_END'
                    raise RuntimeError('No END before the stopped-state receive timeout. Inspect controller log and run.json; no automatic retry.') from exc
                if start_seen_at is None:
                    raise RuntimeError('No controller START after script submission. '
                                       'Check Remote Control and the pendant log; '
                                       'no movement is confirmed. No automatic retry.') from exc
                raise
            if kind == 16:
                state = parse_state(payload)
                if 'timestamp' not in state:
                    continue
                if last_state is not None and state['timestamp'] <= last_state['timestamp']:
                    raise ValueError('Stale controller state during execution monitoring.')
                last_state = state
                if state.get('emergency') or state.get('protective') or state.get('paused'):
                    raise RuntimeError('Controller reports a safety stop or pause. No reset or resume attempted.')
                if end_seen_at is not None and state['timestamp'] >= end_controller_time:
                    if state.get('running'):
                        stopped_packets = 0
                        continue
                    verify_idle(state)
                    stopped_packets += 1
                    if time.monotonic() - end_seen_at >= 1.0 and stopped_packets >= 3:
                        report['controller_stopped_verified'] = True
                        report['final_controller_state'] = state
                        report['post_end_observation_seconds'] = time.monotonic() - end_seen_at
                        report['outcome'] = report['declared_outcome']
                        break
                elif end_seen_at is None and start_seen_at is not None and (state['timestamp'] >= report['controller_start_timestamp_us']):
                    if state.get('running', True):
                        stopped_before_end_at = None
                    elif time.monotonic() - start_seen_at > 1 and stopped_before_end_at is None:
                        stopped_before_end_at = time.monotonic()
                        report['stopped_state_before_END']={'controller_timestamp_us':state['timestamp'],
                            'host_seconds_since_submit':stopped_before_end_at-submitted_at,
                            'running':state.get('running'),'paused':state.get('paused')}
                        print('Stopped state received before END; waiting up to 3 seconds for final messages (no retry)...', flush=True)
                continue
            if kind != 20 or len(payload) < 10:
                continue
            controller_time = struct.unpack_from('!Q', payload, 0)[0]
            if payload[9] == 0:
                msg = payload[10:].decode('utf-8', errors='replace').strip('\x00')
                if f'VIEW_{run_id}|' in msg:
                    print(msg, flush=True)
                    report['messages'].append(msg)
                    if f'VIEW_{run_id}|START|' in msg:
                        if start_seen_at is not None:
                            raise RuntimeError('Duplicate START acknowledgment.')
                        start_seen_at = time.monotonic()
                        report['program_start_acknowledged'] = True
                        report['controller_start_timestamp_us'] = controller_time
                    if f'VIEW_{run_id}|END|' in msg:
                        if start_seen_at is None or end_seen_at is not None:
                            raise RuntimeError('END without START, or duplicate END.')
                        report['declared_outcome'] = msg.split(f'VIEW_{run_id}|END|', 1)[1].strip()
                        report['outcome'] = 'END_RECEIVED_STOP_UNCONFIRMED'
                        end_seen_at = time.monotonic()
                        if stopped_before_end_at is not None:
                            report['END_arrived_after_stopped_state_seconds'] = end_seen_at - stopped_before_end_at
                        stopped_before_end_at = None
                        end_controller_time = controller_time
                        print('END received; watching for late errors and verifying controller stopped...', flush=True)
            elif payload[9] == 10 and len(payload) >= 18:
                line, col = struct.unpack_from('!II', payload, 10)
                error={'type':'RUNTIME_EXCEPTION','line':line,'column':col,
                       'controller_timestamp_us':controller_time,'after_END':end_seen_at is not None,
                       'message':payload[18:].decode('utf-8',errors='replace').strip('\x00')}
                report['controller_messages'].append(error)
                report['outcome'] = 'CONTROLLER_ERROR_AFTER_END' if end_seen_at is not None else 'CONTROLLER_ERROR'
                raise RuntimeError(f"Controller error {line}:{col}: {error['message']}")
            elif payload[9] == 6 and len(payload) >= 22:
                code, arg, level = struct.unpack_from('!IIi', payload, 10)
                event={'type':'CONTROLLER_CODE','code':code,'argument':arg,'level':level,
                       'controller_timestamp_us':controller_time,'raw_hex':payload.hex()}
                report['controller_messages'].append(event)
                print(f'Controller log: C{code}A{arg}; report level {level}', flush=True)
                if code == 210:
                    report['outcome'] = 'REJECTED_LOCAL_CONTROL_C210'
                    raise RuntimeError('C210A0: controller is in Local Control; port 30001 is '
                                       'read-only. Switch to Remote Control on the pendant '
                                       'and make a NEW capture. No robot START was received.')
                if level >= 2:
                    raise RuntimeError(f'Controller warning/error C{code}A{arg}; inspect pendant log.')
    except BaseException as exc:
        report['error'] = str(exc) or type(exc).__name__
        if sock is not None and report['program_start_acknowledged'] and (not report['controller_stopped_verified']):
            try:
                stop_request(sock)
                report['best_effort_stop_requested'] = True
            except OSError:
                report['best_effort_stop_requested'] = False
        if report['outcome'] == 'REJECTED_LOCAL_CONTROL_C210':
            print('Controller rejected this submission in Local Control; no START received. '
                  'Check the pendant before a new capture.',flush=True)
        else:
            print('Outcome unconfirmed. Use pendant Stop; inspect before any retry.', flush=True)
        raise
    finally:
        if sock is not None:
            sock.close()
        (folder / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print('Run record:', folder / 'run.json', flush=True)
    return report

def stop_request(sock):
    sock.sendall(b'def operator_stop_touch():\n  stopl(0.1)\nend\n')

def display_plan(data, plan):
    print(f"Target: corner #{data['target_corner_index']}; captured {data['captured_at_utc']}")
    print('Selected pixel:', np.round(data['target_pixel'], 2))
    print('Recognized point in Base [mm]:', np.round(np.array(plan['target_point_base_m']) * 1000, 3))
    print(f"Commanded Base X offset: +{plan['target_base_x_offset_m'] * 1000:.1f} mm")
    print('Commanded probe-tip point in Base [mm]:',
          np.round(np.array(plan['commanded_tip_point_base_m'])*1000,3))
    print(f"Nominal starting clearance: {plan['start_tip_clearance_mm']:.1f} mm")
    print(f"Direct travel: lateral {plan['lateral_transfer_mm']:.1f} mm; HOLD current orientation.")
    p = plan['target_pose_m_rad']
    print('Commanded target TCP [mm,rad]:', *[round(x * 1000, 3) for x in p[:3]], *[round(x, 6) for x in p[3:]])
    print('One moveL at 10 mm/s, then stay at the target pose. No contact sensing or withdrawal.')
    print('Nominal probe tip offset only; observe physical contact to assess hand-eye accuracy.')

def check_start(state, record):
    verify_idle(state)
    if state['timestamp'] < record['robot_controller_time_at_save_us']:
        raise ValueError('Controller timestamp reset; capture a new target.')
    shift, angle = pose_difference(pose_matrix(state['pose']), pose_matrix(record['start_pose_m_rad']))
    if shift > 0.00015 or angle > math.radians(0.04):
        raise ValueError(f'Robot moved after selection: {shift*1000:.3f} mm, '
                         f'{math.degrees(angle):.4f} deg. Capture again.')
    return {'position_difference_mm': shift * 1000, 'orientation_difference_deg': math.degrees(angle)}

def save_png(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f'PNG encode failed: {path}')
    path.write_bytes(encoded.tobytes())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip', default=IP, help='UR5e controller IP')
    parser.add_argument('--square-mm', type=float, default=SQUARE_M*1000,
                        help='Measured chessboard grid square, default 25.5 mm')
    parser.add_argument('--tip-z-mm', type=float, default=245.6,
                        help='Measured TCP-zero to probe tip along tool Z, default nominal 245.6 mm')
    args = parser.parse_args()
    square = args.square_mm / 1000
    object_points(square)
    if not math.isfinite(args.tip_z_mm) or not 50 <= args.tip_z_mm <= 400:
        raise ValueError('Probe tip Z must be a measured length in the 50–400 mm range.')
    TIP[2] = args.tip_z_mm / 1000
    print(VERSION)
    print('Plane capture -> auto-align -> click a target -> one direct move to its TCP pose.')
    print('Close Orbbec Viewer. Keep robot, board, camera and probe fixed; gripper open.')
    print('Active TCP must be ZERO, matching the saved hand-eye calibration.')
    print(f'Board square {args.square_mm:g} mm; TCP-zero to physical tip {args.tip_z_mm:g} mm. Verify both on the actual hardware.')
    print('This script logs the live payload/CoG; it does not compare them against old fixed values.')
    print('First frame finds the whole board plane. After automatic leveling, click a corner in the NEW frame.')
    require_remote_control(args.ip)
    preliminary = capture_isolated(args.ip)
    if preliminary is None:
        print('Cancelled before alignment; no robot motion.')
        return
    preliminary_board, preliminary_rmse = solve_board(preliminary['corners'], square)
    align = alignment_plan(preliminary, preliminary_board, square)
    cv2.destroyAllWindows()
    session = ROOT/'handeye_touch_runs'/(
        datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
    session.mkdir(parents=True, exist_ok=False)
    save_png(session / 'plane_before_alignment.png', preliminary['bgr'])
    (session/'alignment_plan.json').write_text(
        json.dumps({'plan': align, 'preliminary_board_camera': preliminary_board.tolist(),
                    'corners_px': preliminary['corners'].reshape(-1,2).tolist(),
                    'initial_rmse_px': preliminary_rmse},indent=2,allow_nan=False),
        encoding='utf-8')
    print('Plane centre Base [mm]:', np.round(np.array(align['board_center_base_m']) * 1000, 2))
    print('Plane normal Base:', np.round(align['board_normal_base'], 5))
    print('Predicted probe clearance [mm]:', round(align['minimum_predicted_probe_clearance_mm'], 1))
    print('Predicted board bbox after leveling [px]:', np.round(align['predicted_board_bbox_px'], 1))
    for name, pose in zip(('LIFT', 'ROTATE_HIGH', 'TRANSLATE_HIGH', 'OBSERVE'), align['waypoints_m_rad']):
        print(name, 'TCP [mm,rad]:', *[round(x * 1000, 2) for x in pose[:3]], *[round(x, 5) for x in pose[3:]])
    print('First motion raises the TCP, rotates the entire probe/camera assembly at height,')
    print('then translates at height and descends to the perpendicular camera view.')
    print('Automatic alignment starts now. Keep the entire arm/probe/camera/cable sweep clear.')
    cv2.destroyAllWindows()
    current, _ = read_ready(args.ip)
    start_shift, start_angle = pose_difference(preliminary['robot']['T'], pose_matrix(current['pose']))
    if start_shift > 0.00015 or start_angle > math.radians(0.04):
        raise ValueError('Robot moved after plane capture; no alignment was submitted.')
    require_remote_control(args.ip)
    align_folder = session / 'alignment'
    align_folder.mkdir()
    (align_folder/'verification.json').write_text(
        json.dumps({'start_shift_mm':start_shift*1000,
                    'start_angle_deg':math.degrees(start_angle)},indent=2),encoding='utf-8')
    align_id = uuid.uuid4().hex[:8]
    align_script = alignment_program(align, align_id)
    (align_folder / 'submitted.script').write_text(align_script, encoding='ascii')
    align_result = submit(args.ip, align_script, align_id, align_folder, mode='ALIGN')
    if align_result['outcome'] != 'ALIGN_OK':
        raise RuntimeError('Alignment did not complete. No target selection or touch will follow.')
    print('Alignment ended with controller STOPPED. Capturing NEW view to select target...')
    candidate = capture_isolated(args.ip)
    if candidate is None:
        print('Cancelled after alignment; no touch attempted.')
        return
    board, rmse = solve_board(candidate['corners'], square)
    B = candidate['robot']['T']
    W = B @ H @ board
    board_normal = W[:3, 2].copy()
    if np.dot(board_normal, (B @ H)[:3, 3] - W[:3, 3]) < 0:
        board_normal = -board_normal
    camera_cos = -float(np.dot((B @ H)[:3, 2], board_normal))
    probe_cos = -float(np.dot(B[:3, 2], board_normal))
    if min(camera_cos, probe_cos) < 0.995:
        raise ValueError(f'Actual camera/probe view is not perpendicular enough: '
                         f'camera {camera_cos:.5f}, probe {probe_cos:.5f}. No touch.')
    print(f'Fresh board normal alignment: camera cosine {camera_cos:.5f}; probe cosine {probe_cos:.5f}.')
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    selection = select_target(candidate)
    if selection is None:
        print('Cancelled at target selection; alignment completed, no touch attempted.')
        return
    index, marked = selection
    selected_state, _ = read_ready(args.ip)
    shift, angle = pose_difference(candidate['robot']['T'], pose_matrix(selected_state['pose']))
    if shift > 0.00015 or angle > math.radians(0.04):
        raise ValueError('Robot moved while selecting; restart from fresh capture.')
    plan = build_plan(candidate['robot']['pose'], board, index, square)
    plan['tip_offset_is_nominal'] = args.tip_z_mm == 245.6
    record = dict(plan)
    record.update({'version': VERSION, 'robot_ip': args.ip,
                   'captured_at_utc': candidate['captured_at_utc'],
                   'T_camera_board': board.tolist(), 'camera': CAMERA,
                   'corners_px': candidate['corners'].reshape(-1, 2).tolist(),
                   'target_pixel': candidate['corners'].reshape(-1, 2)[index].tolist(),
                   'tcp_offset_at_capture': candidate['verified_offset'],
                   'reprojection_rmse_px': rmse,
                   'robot_controller_time_at_save_us': selected_state['timestamp'],
                   'capture_stability': candidate['stability'],
                   'accuracy_measured': False})
    save_png(session / 'raw.png', candidate['bgr'])
    save_png(session / 'selected.png', marked)
    (session / 'target.json').write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    cv2.destroyAllWindows()
    print('Selected target:', session / 'selected.png')
    print(f'Board reprojection RMSE {rmse:.3f} px; this is not hand-eye accuracy.')
    display_plan(record, plan)
    print('Selected corner confirmed. Checking robot start pose before the direct move...')
    final_state, states = read_ready(args.ip)
    robot_stats = check_start(final_state, record)
    (session/'verification.json').write_text(
        json.dumps({'robot': robot_stats, 'robot_samples': states,
                    'selected_target_record': 'target.json'},
                   indent=2, allow_nan=False), encoding='utf-8')
    run_id = uuid.uuid4().hex[:8]
    script = controller_program(plan, run_id)
    touch_folder = session / 'touch'
    touch_folder.mkdir()
    (touch_folder / 'submitted.script').write_text(script, encoding='ascii')
    require_remote_control(args.ip)
    with (session / 'attempt_reserved.json').open('x', encoding='utf-8') as stream:
        json.dump({'run_id': run_id, 'reserved_at_utc': datetime.now(timezone.utc).isoformat()}, stream)
    print('Robot start verification:', robot_stats)
    outcome = submit(args.ip, script, run_id, touch_folder, mode='TARGET_POSE')
    print('Final outcome:', outcome['outcome'])
    if outcome['outcome'].startswith('ABORT_'):
        started_motion = any(('|STAGE|' in message for message in outcome['messages']))
        print('Target move may have started.' if started_motion else
              'Target move did not start; stopped during the start/path checks.')
        raise RuntimeError(f"Target move stopped: {outcome['outcome']}")
    print('TCP stopped at the calculated pose. Inspect the probe-to-corner offset manually.')
if __name__ == '__main__':
    try:
        if len(sys.argv) == 4 and sys.argv[1] == '--capture-worker':
            raise SystemExit(capture_worker(sys.argv[2], Path(sys.argv[3])))
        main()
    except KeyboardInterrupt:
        print('Interrupted. Verify the stopped state on the pendant; no automatic restart.')
        raise SystemExit(1)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        raise SystemExit(f'STOPPED: {exc}')
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
