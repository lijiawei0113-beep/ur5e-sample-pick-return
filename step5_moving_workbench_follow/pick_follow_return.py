"""Pick samples, follow the green workbench in real time, and return them."""
from __future__ import annotations
import os
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


VERSION = 'GREEN_WORKBENCH_LIVE_FOLLOW_PICK_RETURN_V2_2_TABLE_STOP'
ROOT = Path(__file__).resolve().parent
STEP4_DIR = ROOT.parent / 'step4_sample_pick_and_return'
sys.path.insert(0, str(STEP4_DIR))

import robot as motion
import vision
import pick_place_return as pick

RUN_JSON = ROOT / 'green_workbench_follow_run.json'
LATEST_VIEW = ROOT / 'green_workbench_follow_latest.png'

# Fresh RGB-D capture
WIDE_WARMUP, WIDE_FRAMES = (2, 7)
CENTER_WARMUP, CENTER_FRAMES = (3, 9)
CAPTURE_TIMEOUT_S = 15.0

# Green target
GREEN_LOWER = np.array([82, 55, 95], dtype=np.uint8)
GREEN_UPPER = np.array([100, 255, 255], dtype=np.uint8)
MIN_GREEN_AREA_PX = 70
MAX_GREEN_AREA_PX = 30000
MIN_TOP_SPAN_M = 0.012
MAX_TOP_SPAN_M = 0.1
INITIAL_LOCK_RADIUS_M = 0.08
FOLLOW_LOCK_RADIUS_M = 0.1

# Live XY follow
SIGHTLINE_OFFSET_M = 0.06
FOLLOW_PERIOD_S = 0.05
FOLLOW_COMMAND_AGE_S = 0.25
FOLLOW_FRAME_WAIT_MS = 600
FOLLOW_SPEED_M_S = 0.04
FOLLOW_ACCEL_M_S2 = 0.25
FOLLOW_STOP_ACCEL_M_S2 = 0.8
FOLLOW_KP_S_INV = 1.2
FOLLOW_ALIGNMENT_M = 0.003
STILL_CENTER_RADIUS_M = 0.002
STILL_SECONDS = 3.0
STILL_MIN_OBSERVATIONS = 8
MAX_CAPTURE_GAP_S = 0.8
FOLLOW_TIMEOUT_S = 150.0
FOLLOW_CAMERA_REOPEN_S = 3.0
FOLLOW_LOST_ABORT_S = 8.0
TABLE_QUIET_WINDOW_S = 0.65
TABLE_QUIET_RADIUS_M = 0.0015
TABLE_QUIET_TRAVEL_M = 0.0012
TABLE_RESUME_SHIFT_M = 0.004
MAX_FINAL_FOLLOW_ERROR_M = 0.03

# Preview window
DISPLAY_NAME = 'Green workbench live follow (Esc to stop)'
DISPLAY_IMAGE_WIDTH = 1280
DISPLAY_PANEL_HEIGHT = 152


def preview_max_size() -> tuple[int, int]:
    """Keep the enlarged preview inside the usable monitor area on Windows."""
    screen_w, screen_h = (1920, 1080)
    if os.name == 'nt':
        try:
            import ctypes
            screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        except (AttributeError, OSError):
            pass
    return min(DISPLAY_IMAGE_WIDTH, round(screen_w * 0.90)), max(
        300, round(screen_h * 0.88) - DISPLAY_PANEL_HEIGHT
    )


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_run(run: dict) -> None:
    pick.write_json(RUN_JSON, run)


def capture(camera: pick.PersistentRGBDCamera, receiver, label: str, warmup: int, frames: int):
    """Request a fresh synchronized snapshot while the robot is stopped."""
    old_warmup = vision.WARMUP_VALID_FRAME_COUNT
    old_frames = vision.CAPTURE_VALID_FRAME_COUNT
    old_timeout = vision.TOTAL_CAPTURE_TIMEOUT_S
    vision.WARMUP_VALID_FRAME_COUNT = warmup
    vision.CAPTURE_VALID_FRAME_COUNT = frames
    vision.TOTAL_CAPTURE_TIMEOUT_S = CAPTURE_TIMEOUT_S
    try:
        camera.open()
        return camera.capture(label, receiver, recovery_attempts=2, no_progress_timeout_s=4.0)
    finally:
        vision.WARMUP_VALID_FRAME_COUNT = old_warmup
        vision.CAPTURE_VALID_FRAME_COUNT = old_frames
        vision.TOTAL_CAPTURE_TIMEOUT_S = old_timeout


def pixels_on_top_plane(pixels_uv: np.ndarray, t_base_camera: np.ndarray, top_z: float) -> np.ndarray:
    uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(uv, vision.COLOR_K, vision.COLOR_D).reshape(-1, 2)
    rays = np.column_stack((normalized, np.ones(len(normalized))))
    rotation = np.asarray(t_base_camera[:3, :3], dtype=np.float64)
    origin = np.asarray(t_base_camera[:3, 3], dtype=np.float64)
    rays_base = (rotation @ rays.T).T
    if np.any(np.abs(rays_base[:, 2]) < 1e-05):
        raise RuntimeError('Camera rays are nearly parallel to the green top')
    lengths = (top_z - origin[2]) / rays_base[:, 2]
    if np.any(lengths <= 0.0) or np.any(lengths > 2.0):
        raise RuntimeError('Green top plane lies behind the camera')
    return origin + lengths[:, None] * rays_base


def green_center(
    color: np.ndarray,
    robot: dict,
    top_z: float,
    anchor_xy: np.ndarray,
    radius_m: float,
    previous_uv: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict]:
    blurred = cv2.GaussianBlur(color, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    valid = []
    rejected = {'area_or_border': 0, 'plane_or_span': 0, 'lock': 0}
    for contour in contours:
        area_px = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        if (
            not (MIN_GREEN_AREA_PX <= area_px <= MAX_GREEN_AREA_PX) or (
                x < 3 or y < 3 or x + w >= width - 3 or y + h >= height - 3
            )
        ):
            rejected['area_or_border'] += 1
            continue
        try:
            points = pixels_on_top_plane(contour.reshape(-1, 2), robot['T_base_camera'], top_z)[:, :2]
        except RuntimeError:
            rejected['plane_or_span'] += 1
            continue
        span = np.ptp(points, axis=0)
        if np.min(span) < MIN_TOP_SPAN_M or np.max(span) > MAX_TOP_SPAN_M:
            rejected['plane_or_span'] += 1
            continue
        metric_contour = (points * 1000.0).astype(np.float32).reshape(-1, 1, 2)
        moments = cv2.moments(metric_contour)
        if abs(moments['m00']) < 1.0:
            rejected['plane_or_span'] += 1
            continue
        center = np.array(
            [moments["m10"] / moments["m00"],
             moments["m01"] / moments["m00"]], dtype=np.float64
        ) / 1000.0
        distance = float(np.linalg.norm(center - anchor_xy))
        if distance > radius_m:
            rejected['lock'] += 1
            continue
        mm = cv2.moments(contour)
        if mm['m00'] == 0:
            continue
        uv_center = np.array([mm['m10'] / mm['m00'], mm['m01'] / mm['m00']])
        pixel_step = 0.0 if previous_uv is None else float(np.linalg.norm(uv_center - previous_uv))
        if previous_uv is not None and pixel_step > 220.0:
            rejected['lock'] += 1
            continue
        valid.append((pixel_step if previous_uv is not None else distance, -area_px, center, contour, uv_center))
    valid.sort(key=lambda item: (item[0], item[1]))
    view = color.copy()
    if valid:
        _, _, center, contour, uv_center = valid[0]
        cv2.drawContours(view, [contour], -1, (0, 255, 255), 2)
        uv_int = np.rint(uv_center).astype(int)
        cv2.drawMarker(view, tuple(uv_int), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        xy_mm = np.round(center * 1000.0, 1)
        cv2.putText(
            view, f"GREEN Base XY {xy_mm.tolist()} mm",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 0), 2,
        )
    else:
        center = None
        uv_center = None
        cv2.putText(view, 'GREEN TOP NOT FOUND - TCP HELD', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return center, {
        "contours": len(contours), "accepted": len(valid),
        "center_uv": None if uv_center is None else uv_center.tolist(),
        **rejected,
    }


def follow_preview(image: np.ndarray, phase: str, *, error_mm: float | None=None, stable_s: float=0.0, target_uv: np.ndarray | None=None) -> np.ndarray:
    """Large preview with an undistorted camera image and separate status area."""
    height, width = image.shape[:2]
    max_width, max_height = preview_max_size()
    scale = min(max_width / width, max_height / height)
    display_width = max(1, round(width * scale))
    display_height = max(1, round(height * scale))
    camera_view = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
    if target_uv is not None:
        u, v = np.rint(np.asarray(target_uv) * display_width / width).astype(int)
        cv2.drawMarker(camera_view, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 38, 2)
        cv2.circle(camera_view, (u, v), 20, (0, 255, 255), 2)
    canvas = np.full((display_height + DISPLAY_PANEL_HEIGHT, display_width, 3), (26, 28, 32), dtype=np.uint8)
    canvas[:display_height] = camera_view
    base_y = display_height
    color = {
        "FOLLOW": (0, 195, 255),
        "SETTLING": (70, 220, 110),
        "READY": (70, 220, 110),
        "LOST": (80, 90, 255),
        "CAMERA WAIT": (80, 90, 255),
    }.get(phase, (220, 220, 220))
    cv2.putText(
        canvas, "GREEN WORKBENCH  /  LIVE FOLLOW",
        (24, base_y + 40), cv2.FONT_HERSHEY_SIMPLEX,
        0.92, (242, 244, 245), 2, cv2.LINE_AA,
    )
    cv2.putText(canvas, phase, (24, base_y + 83), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    error_label = 'XY ERROR  --' if error_mm is None else f'XY ERROR  {error_mm:.1f} mm'
    cv2.putText(
        canvas, error_label, (290, base_y + 83),
        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (235, 235, 235), 2, cv2.LINE_AA,
    )
    stable_fraction = min(max(stable_s / STILL_SECONDS, 0.0), 1.0)
    cv2.putText(
        canvas, f"STABLE  {stable_s:.1f} / {STILL_SECONDS:.0f} s",
        (24, base_y + 122), cv2.FONT_HERSHEY_SIMPLEX,
        0.68, (235, 235, 235), 2, cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (290, base_y + 109), (display_width - 24, base_y + 125), (65, 68, 74), -1)
    cv2.rectangle(
        canvas, (290, base_y + 109),
        (290 + round((display_width - 314) * stable_fraction), base_y + 125),
        color, -1,
    )
    cv2.putText(
        canvas, "Esc: stop following", (display_width - 246, base_y + 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.54, (178, 184, 190), 1, cv2.LINE_AA,
    )
    return canvas


def optical_sightline_direction(robot: dict, top_z: float) -> np.ndarray:
    camera = np.asarray(robot['T_base_camera'], dtype=np.float64)
    origin = camera[:3, 3]
    direction = camera[:3, 2]
    if abs(direction[2]) < 1e-05:
        raise RuntimeError('Camera optical axis is parallel to the workbench')
    scale = (top_z - origin[2]) / direction[2]
    if scale <= 0.0:
        raise RuntimeError('Workbench is behind the camera')
    hit_xy = (origin + scale * direction)[:2]
    tcp_xy = np.asarray(pick.initial_onrobot_tcp_pose(robot)[:2], dtype=np.float64)
    sightline = hit_xy - tcp_xy
    norm = float(np.linalg.norm(sightline))
    if norm < 0.005:
        raise RuntimeError('Camera/TCP sightline offset is too small')
    return sightline / norm


def initial_offset_pose(high_pose: list[float], robot: dict, top_z: float) -> list[float]:
    """Project the calibrated eye-in-hand geometry at the planned high pose."""
    flange_to_camera = vision.inverse_transform(robot['T_base_flange']) @ robot['T_base_camera']
    high_tcp = vision.pose6_to_transform(high_pose)
    flange_to_tcp = vision.pose6_to_transform(motion.ONROBOT_SINGLE_TCP)
    high_flange = high_tcp @ vision.inverse_transform(flange_to_tcp)
    synthetic = {'T_base_camera': high_flange @ flange_to_camera, 'T_base_flange': high_flange}
    direction = optical_sightline_direction(synthetic, top_z)
    offset_pose = list(high_pose)
    offset_pose[0] -= float(direction[0] * SIGHTLINE_OFFSET_M)
    offset_pose[1] -= float(direction[1] * SIGHTLINE_OFFSET_M)
    return offset_pose


def table_is_stationary(history: deque[tuple[float, np.ndarray]], now: float) -> bool:
    """Estimate table motion in Base, independently of TCP alignment error."""
    while history and now - history[0][0] > TABLE_QUIET_WINDOW_S:
        history.popleft()
    if len(history) < 5 or history[-1][0] - history[0][0] < 0.45:
        return False
    centres = np.stack([xy for _, xy in history])
    median = np.median(centres, axis=0)
    radius = float(np.max(np.linalg.norm(centres - median, axis=1)))
    travel = float(np.linalg.norm(centres[-1] - centres[0]))
    return radius <= TABLE_QUIET_RADIUS_M and travel <= TABLE_QUIET_TRAVEL_M


# This thread owns RTDE Control throughout the follow stage.
class LiveXYServo:
    """RTDE Control has one owner; stale vision commands become a speed stop."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.velocity = np.zeros(2, dtype=np.float64)
        self.updated_at = 0.0
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(10.0):
            self.stop_event.set()
            raise RuntimeError('RTDE follow controller did not start')
        self.check()

    def command(self, xy: np.ndarray) -> None:
        with self.lock:
            self.velocity = np.asarray(xy, dtype=np.float64).reshape(2).copy()
            self.updated_at = time.monotonic()

    def hold(self) -> None:
        with self.lock:
            self.velocity[:] = 0.0
            self.updated_at = 0.0

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f'RTDE live follow stopped: {self.error}') from self.error

    def close(self) -> None:
        self.hold()
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            raise RuntimeError('RTDE thread did not exit; stop the robot on the pendant')
        self.check()

    def _run(self) -> None:
        control = None
        moving = False
        try:
            control = pick.start_rtde_control_windows()
            self.ready.set()
            while not self.stop_event.is_set():
                with self.lock:
                    velocity = self.velocity.copy()
                    updated_at = self.updated_at
                if updated_at and time.monotonic() - updated_at < FOLLOW_COMMAND_AGE_S:
                    result = control.speedL(
                        [float(velocity[0]), float(velocity[1]), 0., 0., 0., 0.],
                        FOLLOW_ACCEL_M_S2, FOLLOW_PERIOD_S,
                    )
                    if result is False:
                        raise RuntimeError('speedL returned False')
                    moving = bool(np.linalg.norm(velocity) > 1e-05)
                else:
                    if moving:
                        control.speedStop(FOLLOW_STOP_ACCEL_M_S2)
                        moving = False
                    self.stop_event.wait(0.01)
        except BaseException as error:
            self.error = error
            self.ready.set()
        finally:
            if control is not None:
                try:
                    control.speedStop(FOLLOW_STOP_ACCEL_M_S2)
                finally:
                    pick.stop_rtde_control_windows(control, strict=False)


def follow_live_workbench(receiver, initial_anchor: np.ndarray, top_z: float, high_pose: list[float], label: str) -> tuple[np.ndarray, dict]:
    """Show every RGB frame and follow in XY until the green top stops."""
    anchor = np.asarray(initial_anchor, dtype=np.float64).reshape(2)
    tcp_offset, _ = vision.read_active_tcp_offset()
    pipeline = None
    servo = LiveXYServo()
    start = time.monotonic()
    last_frame = start
    last_valid = start
    last_print = 0.0
    last_save = 0.0
    last_uv = None
    stable_since = None
    stable_reference = None
    stable_centers: deque[np.ndarray] = deque(maxlen=50)
    table_history: deque[tuple[float, np.ndarray]] = deque(maxlen=45)
    table_stop_reference = None
    resume_count = 0
    last_error_mm = None
    last_tcp_speed_mm_s = None
    receive_restarts = 0
    last_diagnostics = {}
    previous_color = None
    phase = 'FOLLOW'
    restarted = False
    preview_resized = False
    cv2.namedWindow(DISPLAY_NAME, cv2.WINDOW_NORMAL | getattr(cv2, 'WINDOW_KEEPRATIO', 0))

    def show_preview(image: np.ndarray, state: str, **status) -> np.ndarray:
        nonlocal preview_resized
        preview = follow_preview(image, state, **status)
        if not preview_resized:
            cv2.resizeWindow(DISPLAY_NAME, preview.shape[1], preview.shape[0])
            preview_resized = True
        cv2.imshow(DISPLAY_NAME, preview)
        return preview
    try:
        pipeline, align_filter, ob_format = vision.open_camera_pipeline()
        servo.start()
        while time.monotonic() - start < FOLLOW_TIMEOUT_S:
            servo.check()
            now = time.monotonic()
            try:
                frames = pipeline.wait_for_frames(FOLLOW_FRAME_WAIT_MS)
                aligned = None if frames is None else vision.aligned_frameset(align_filter, frames)
                if aligned is None:
                    raise RuntimeError('no aligned frame')
                color, _depth, _scale = vision.extract_aligned_rgbd(aligned, ob_format)
            except RuntimeError:
                servo.hold()
                stable_since = None
                stable_centers.clear()
                table_history.clear()
                if previous_color is not None:
                    show_preview(previous_color, 'CAMERA WAIT')
                if cv2.waitKey(1) & 255 == 27:
                    raise RuntimeError('Esc pressed: follow stopped')
                if now - last_frame > FOLLOW_CAMERA_REOPEN_S and (not restarted):
                    pipeline.stop()
                    pipeline, align_filter, ob_format = vision.open_camera_pipeline()
                    restarted = True
                    last_frame = time.monotonic()
                elif now - last_frame > FOLLOW_LOST_ABORT_S:
                    raise RuntimeError('Camera stream stopped; sample held high')
                continue
            now = time.monotonic()
            if now - last_frame > MAX_CAPTURE_GAP_S:
                stable_since = None
                stable_centers.clear()
                table_history.clear()
            last_frame = now
            restarted = False
            previous_color = color
            try:
                robot = receiver.snapshot(vision.robot_snapshot, tcp_offset, reconnect_on_stale=False)
            except Exception as error:
                servo.hold()
                stable_since = None
                stable_centers.clear()
                table_history.clear()
                print(f'{label}: RTDE Receive interrupted; holding XY: {error}', flush=True)
                servo.close()
                if receive_restarts >= 1:
                    raise RuntimeError('RTDE Receive failed again during follow') from error
                receiver.snapshot(vision.robot_snapshot, tcp_offset, reconnect_on_stale=True)
                receive_restarts += 1
                servo = LiveXYServo()
                servo.start()
                phase = 'FOLLOW'
                last_frame = time.monotonic()
                continue
            tcp = np.asarray(robot['actual_tcp_pose_m_rad'], dtype=np.float64)
            tcp_speed = np.asarray(robot['actual_tcp_speed_m_rad_s'][:2])
            if abs(float(tcp[2]) - float(high_pose[2])) > 0.005:
                servo.hold()
                raise RuntimeError('TCP left the fixed high follow plane')
            green_xy, diagnostics = green_center(
                color, robot, top_z, anchor,
                INITIAL_LOCK_RADIUS_M if last_uv is None else FOLLOW_LOCK_RADIUS_M,
                last_uv,
            )
            last_diagnostics = diagnostics
            view = color.copy()
            display_error_mm = None
            display_stable_s = 0.0
            display_uv = None
            if green_xy is None:
                servo.hold()
                stable_since = None
                stable_centers.clear()
                table_history.clear()
                phase = 'LOST'
                if now - last_valid > FOLLOW_LOST_ABORT_S:
                    raise RuntimeError('Green top lost; sample held high')
            else:
                last_valid = now
                if phase == 'LOST':
                    phase = 'FOLLOW'
                last_uv = np.asarray(diagnostics['center_uv'], dtype=float)
                anchor = green_xy
                display_uv = last_uv
                relative_xy = green_xy - tcp[:2]
                sightline = optical_sightline_direction(robot, top_z)
                error_xy = relative_xy - sightline * SIGHTLINE_OFFSET_M
                error = float(np.linalg.norm(error_xy))
                tcp_speed_norm = float(np.linalg.norm(tcp_speed))
                last_error_mm = error * 1000.0
                last_tcp_speed_mm_s = tcp_speed_norm * 1000.0
                table_history.append((now, green_xy.copy()))
                if phase == 'SETTLING':
                    servo.hold()
                    if table_stop_reference is not None and np.linalg.norm(green_xy - table_stop_reference) > TABLE_RESUME_SHIFT_M:
                        resume_count += 1
                    else:
                        resume_count = 0
                    if resume_count >= 2:
                        phase = 'FOLLOW'
                        stable_since = None
                        stable_centers.clear()
                        table_history.clear()
                        table_stop_reference = None
                        resume_count = 0
                    elif tcp_speed_norm < 0.002:
                        if (
                            stable_reference is None or stable_since is None
                                or np.linalg.norm(green_xy - stable_reference)
                                > STILL_CENTER_RADIUS_M
                        ):
                            stable_since = now
                            stable_reference = green_xy.copy()
                            stable_centers.clear()
                        stable_centers.append(green_xy.copy())
                    else:
                        stable_since = None
                        stable_centers.clear()
                if phase == 'FOLLOW':
                    stable_since = None
                    stable_centers.clear()
                    if table_is_stationary(table_history, now):
                        servo.hold()
                        phase = 'SETTLING'
                        stable_reference = None
                        table_stop_reference = green_xy.copy()
                        resume_count = 0
                        print(f"{label}: green top stopped; continuous XY speed "
                              f"held, remaining alignment {error*1000:.1f} mm",
                              flush=True)
                    elif error <= FOLLOW_ALIGNMENT_M:
                        servo.hold()
                    else:
                        velocity = error_xy * FOLLOW_KP_S_INV
                        magnitude = float(np.linalg.norm(velocity))
                        if magnitude > FOLLOW_SPEED_M_S:
                            velocity *= FOLLOW_SPEED_M_S / magnitude
                        servo.command(velocity)
                elapsed = 0.0 if stable_since is None else now - stable_since
                display_error_mm = error * 1000.0
                display_stable_s = elapsed
                if phase == 'SETTLING' and elapsed >= STILL_SECONDS and (len(stable_centers) >= STILL_MIN_OBSERVATIONS):
                    if error > MAX_FINAL_FOLLOW_ERROR_M:
                        raise RuntimeError(
                            f"Workbench stopped but TCP remains {error*1000:.1f} "
                            "mm from the planned follow offset; placement blocked"
                        )
                    final_xy = np.median(np.stack(stable_centers), axis=0)
                    servo.hold()
                    display_view = show_preview(
                        view, "READY", error_mm=display_error_mm,
                        stable_s=display_stable_s, target_uv=display_uv,
                    )
                    cv2.waitKey(1)
                    cv2.imwrite(str(LATEST_VIEW), display_view)
                    return final_xy, {
                        "method": "live_xy_servo_then_stationary_green_area_centroid",
                        "stable_seconds": elapsed,
                        "stable_observations": len(stable_centers),
                        "last_detection": diagnostics,
                        "final_center_base_xy_m": final_xy.tolist(),
                    }
            display_view = show_preview(
                view, phase, error_mm=display_error_mm,
                stable_s=display_stable_s, target_uv=display_uv,
            )
            if now - last_save > 0.5:
                cv2.imwrite(str(LATEST_VIEW), display_view)
                last_save = now
            if now - last_print >= 1.0:
                print(f"{label}: {phase}; target {diagnostics['accepted']}, "
                      f"XY error {last_error_mm if last_error_mm is not None else float('nan'):.1f} mm; "
                      f"TCP speed {last_tcp_speed_mm_s if last_tcp_speed_mm_s is not None else float('nan'):.1f} mm/s; "
                      f"stable {0 if stable_since is None else now-stable_since:.1f}/3 s",
                      flush=True)
                last_print = now
            if cv2.waitKey(1) & 255 == 27:
                raise RuntimeError('Esc pressed: follow stopped')
        raise TimeoutError('Green workbench did not settle; sample held high')
    finally:
        try:
            servo.close()
        finally:
            if pipeline is not None:
                pipeline.stop()
            cv2.destroyWindow(DISPLAY_NAME)


def move_checked(control, pose: list[float], name: str, *, vertical=False) -> None:
    if hasattr(control, 'isPoseWithinSafetyLimits') and (not control.isPoseWithinSafetyLimits(pose)):
        raise RuntimeError(f'Controller rejected {name}')
    speed = motion.VERTICAL_SPEED_M_S if vertical else motion.TRANSFER_SPEED_M_S
    accel = motion.VERTICAL_ACCEL_M_S2 if vertical else motion.TRANSFER_ACCEL_M_S2
    motion.move_l(control, pose, speed, accel, name)


def main() -> int:
    print(VERSION, flush=True)
    print(
        "One pass: tray discovery -> grasp -> live green top follow -> "
        "3-second stop -> hold closed at top -> return to original tray point.",
        flush=True,
    )
    # Keep this stage's coordinates and logs in its own folder.
    pick.RUN_JSON = RUN_JSON
    pick.WORKBENCH_JSON = ROOT / 'workbench_pose.json'
    pick.RECOVERY_JSON = ROOT / 'rtde_recovery_checkpoint.json'
    pick.OVERVIEW_IMAGE = ROOT / 'all_samples_result.png'
    run = {'version': VERSION, 'started_at_utc': timestamp(), 'samples': []}
    receiver = None
    control = None
    camera = pick.PersistentRGBDCamera()
    try:
        place, _place_info = pick.load_calibrated_workbench()
        top_z = float(place[2])
        anchor = np.asarray(place[:2], dtype=np.float64)
        receiver = pick.PersistentRTDEReceive(
            motion.ROBOT_IP, frequency_hz=125.0,
            connect_attempts=pick.RTDE_RECONNECT_ATTEMPTS,
            freshness_timeout_s=0.60,
        )
        run['stage'] = 'oblique_tray'
        home_robot = pick.read_stationary_robot_snapshot(receiver)
        color, depth, _meta, oblique_robot = capture(camera, receiver, 'new_oblique_tray', WIDE_WARMUP, WIDE_FRAMES)
        tray, _ = pick.locate_tray_from_oblique(color, depth, oblique_robot)
        run['stage'] = 'vertical_overview'
        camera.close(announce=False)
        pick.verticalize_camera(home_robot, tray, run)
        overview_robot = pick.read_stationary_robot_snapshot(receiver)
        overview_expected = np.asarray(run['vertical_overview']['pose_base_tcp_m_rad'][:3], dtype=np.float64)
        overview_actual = np.asarray(pick.initial_onrobot_tcp_pose(overview_robot)[:3], dtype=np.float64)
        if np.linalg.norm(overview_actual - overview_expected) > 0.003:
            raise RuntimeError('Robot has not reached the planned tray overview')
        run['stage'] = 'tray_samples'
        wide_color, wide_depth, wide_meta, wide_robot = capture(
            camera, receiver, "new_tray_wide", WIDE_WARMUP, WIDE_FRAMES
        )
        strict, planes, _ = vision.coarse_depth_candidates(wide_color, wide_depth, wide_robot)
        candidates, _, _ = pick.recover_missed_coarse_candidates(wide_depth, wide_robot, strict, planes)
        vertical_tray = pick.confirm_tray_from_vertical_capture(wide_color, wide_depth, wide_robot, tray)
        candidates, _ = pick.filter_candidates_to_tray(candidates, tray, vertical_tray)
        if not candidates:
            raise RuntimeError('No tray sample detected')
        print(f'Tray samples to process: {len(candidates)}', flush=True)
        camera.close(announce=False)
        pick.prepare_rg2_command_program()
        pick.run_loaded_rg2_command('OPEN', receiver)
        control = pick.start_rtde_control_windows(post_rg2=True)
        current_pose = pick.initial_onrobot_tcp_pose(wide_robot)
        for order, coarse in enumerate(candidates, start=1):
            label = f'SAMPLE_{order}'
            run['stage'] = f'{label}:center_camera'
            _, view_pose = pick.desired_batch_view_pose(coarse, wide_robot)
            current_pose = pick.move_to_centered_view(control, current_pose, view_pose, label)
            pick.stop_rtde_control_windows(control)
            control = None
            centered_robot = pick.read_stationary_robot_snapshot(receiver)
            centered_actual = np.asarray(pick.initial_onrobot_tcp_pose(centered_robot)[:3], dtype=np.float64)
            if np.linalg.norm(centered_actual - np.asarray(view_pose[:3])) > 0.003:
                raise RuntimeError(f'{label}: camera did not reach centered pose')
            run['stage'] = f'{label}:refine'
            close_color, close_depth, _close_meta, close_robot = capture(
                camera, receiver, f"{label}_close",
                CENTER_WARMUP, CENTER_FRAMES,
            )
            refined, _view = pick.refine_selected_top(close_color, close_depth, close_robot, coarse, planes)
            camera.close(announce=False)
            search_place = [float(anchor[0]), float(anchor[1]), top_z]
            sample = pick.build_sample_motion(
                order, coarse, refined, view_pose, search_place,
                {"source_file": "workbench_pose.json"},
            )
            run["samples"].append({
                "order": order,
                "refined_center_base_m": sample["refined_center_base_m"],
                "status": "refined",
            })
            save_run(run)
            control = pick.start_rtde_control_windows()
            pick.preflight_sample_poses(control, sample)
            poses = sample['poses_base_tcp_m_rad']
            run['stage'] = f'{label}:pick'
            move_checked(control, poses['high_pick'], f'{label}_HIGH_PICK')
            move_checked(control, poses['grasp_pick'], f'{label}_GRASP', vertical=True)
            control = pick.gripper_action_windows(control, receiver, 'CLOSE')
            move_checked(control, poses['high_pick'], f'{label}_LIFT', vertical=True)
            move_checked(control, poses['transit_low'], f'{label}_TRANSIT_LOW')
            move_checked(control, poses['transit_high'], f'{label}_TRANSIT_HIGH', vertical=True)
            move_checked(control, poses['high_place'], f'{label}_HIGH_PLACE')
            offset_high = initial_offset_pose(poses['high_place'], close_robot, top_z)
            move_checked(control, offset_high, f'{label}_FOLLOW_VIEW_HIGH')
            pick.stop_rtde_control_windows(control)
            control = None
            run['stage'] = f'{label}:green_follow'
            final_xy, follow_details = follow_live_workbench(receiver, anchor, top_z, offset_high, label)
            anchor = final_xy.copy()
            run['samples'][-1]['follow'] = follow_details
            save_run(run)
            final_high = list(poses['high_place'])
            final_high[:2] = final_xy.tolist()
            final_place = list(poses['grasp_place'])
            final_place[:2] = final_xy.tolist()
            run['stage'] = f'{label}:hold_and_return'
            control = pick.start_rtde_control_windows()
            move_checked(control, final_high, f'{label}_CENTER_HIGH')
            move_checked(control, final_place, f'{label}_HOLD_AT_TOP', vertical=True)
            time.sleep(float(motion.WAIT_AT_PLACE_S))
            move_checked(control, final_high, f'{label}_LIFT_FROM_TOP', vertical=True)
            move_checked(control, poses['transit_high'], f'{label}_RETURN_HIGH')
            move_checked(control, poses['transit_low'], f'{label}_RETURN_LOW', vertical=True)
            move_checked(control, poses['high_pick'], f'{label}_RETURN_TRAY_HIGH')
            move_checked(control, poses['grasp_pick'], f'{label}_RETURN_ORIGINAL', vertical=True)
            control = pick.gripper_action_windows(control, receiver, 'OPEN')
            move_checked(control, poses['high_pick'], f'{label}_LIFT_AFTER_OPEN', vertical=True)
            current_pose = list(poses['high_pick'])
            run['samples'][-1]['status'] = 'returned_to_original_tray_point'
            save_run(run)
        run['stage'] = 'return_home'
        motion.move_j(control, home_robot['actual_q_rad'], 'FINAL_HOME_Q')
        run['stage'] = 'completed'
        print('COMPLETE: samples returned to their original tray points.', flush=True)
        return 0
    except BaseException as error:
        run['stage_error'] = f'{type(error).__name__}: {error}'
        print(
            f"STOPPED at {run.get('stage')}: {run['stage_error']}; "
            "no automatic movement recovery.", file=sys.stderr, flush=True,
        )
        traceback.print_exc()
        return 1
    finally:
        camera.close()
        pick.stop_rtde_control_windows(control, strict=False)
        if receiver is not None:
            receiver.close()
        run['finished_at_utc'] = timestamp()
        save_run(run)


if __name__ == '__main__':
    sys.exit(main())
