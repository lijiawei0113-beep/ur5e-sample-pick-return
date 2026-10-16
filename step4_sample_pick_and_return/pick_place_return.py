from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import gc
import json
import math
import os
import sys
import threading
import time
import traceback
import warnings

import cv2
import numpy as np

from robot import PersistentRTDEReceive
import robot
import vision


VERSION = 'STEP6_PICK_HOLD_RETURN_ALL_VERTICAL_TRAY_ROI_V19'
ROOT = Path(__file__).resolve().parent
OVERVIEW_IMAGE = ROOT / 'all_samples_result.png'
RUN_JSON = ROOT / 'all_samples_result.json'
RECOVERY_JSON = ROOT / 'rtde_recovery_checkpoint.json'
WORKBENCH_JSON = ROOT / 'workbench_pose.json'
RECOVERY_SCHEMA = 'ur5e_empty_gripper_checkpoint_v1'
RECOVERY_CLEARANCE_Z_M = 0.25
JSON_REPLACE_RETRY_DELAYS_S = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)

# Gripper commands
RG2_COMMAND_PROGRAM = '/programs/rg2_command.urp'
RG2_COMMAND_REGISTER = 18
RG2_COMMAND_IDLE = 0
RG2_COMMAND_OPEN = 1
RG2_COMMAND_CLOSE = 2
RG2_ACK_REGISTER = 18
RG2_ACK_OPEN_ACTIVE = 101
RG2_ACK_OPEN_DONE = 1
RG2_ACK_CLOSE_ACTIVE = 102
RG2_ACK_CLOSE_DONE = 2
RG2_REGISTER_WRITE_ATTEMPTS = 3
RG2_PROGRAM_TIMEOUT_S = 20.0
RG2_PROGRAM_START_TIMEOUT_S = 5.0
RG2_ACK_POLL_S = 0.05
RG2_ACK_AFTER_STOP_GRACE_S = 1.0
RG2_LOAD_SETTLE_S = 1.0
RG2_POST_PROGRAM_SETTLE_S = 1.5
RTDE_RECONNECT_ATTEMPTS = 4
RTDE_RECONNECT_BASE_DELAY_S = 1.0
POST_RG2_RTDE_RECONNECT_DELAYS_S = (0.0, 2.0, 3.0, 5.0, 8.0, 12.0)

# Capture and recovery
CAMERA_RECOVERY_ATTEMPTS = 2
CAMERA_RECOVERY_DELAY_S = 2.0
CENTER_CAMERA_RECOVERY_ATTEMPTS = 3
CENTER_CAMERA_NO_PROGRESS_TIMEOUT_S = 4.0
CAMERA_STATIONARY_TIMEOUT_S = 5.0
CAMERA_STATIONARY_HOLD_S = 0.3
CAMERA_STATIONARY_POLL_S = 0.05
CAMERA_MAX_TCP_LINEAR_SPEED_M_S = 0.002
CAMERA_MAX_TCP_ANGULAR_SPEED_RAD_S = 0.02

# Workbench and transfer geometry
WORKBENCH_NOMINAL_TOP_SIDE_M = 0.03
CYLINDER_NOMINAL_DIAMETER_M = 0.025
MIN_CYLINDER_EDGE_MARGIN_M = 0.0015
WORKBENCH_VERTICAL_TRANSFER_CLEARANCE_M = 0.07
WORKBENCH_MIN_LIFT_CLEARANCE_M = 0.03
WORKBENCH_HIGH_Z_SEARCH_STEP_M = 0.005
WORKBENCH_FREE_YAW_DELTAS_DEG = (0.0, 90.0, -90.0, 180.0)
WORKBENCH_TRANSIT_FRACTIONS = (0.5, 0.4, 0.6, 0.3, 0.7)
WORKBENCH_TRANSIT_MIN_ENDPOINT_DISTANCE_M = 0.1
MIN_BATCH_CANDIDATES = 1
BATCH_MAX_VIEW_TRANSLATION_M = 0.45
VERTICALIZE_MIN_TCP_Z_M = 0.25
MIN_VERTICAL_OVERVIEW_TCP_Z_M = 0.18
OVERVIEW_CAMERA_IMAGE_DOWN_OFFSET_M = 0.06
MIN_VERTICAL_CAMERA_CLEARANCE_M = 0.3
MAX_VERTICAL_CAMERA_CLEARANCE_M = 0.6
TRAY_IMAGE_MARGIN_PX = 25.0
TRAY_FOV_METRIC_MARGIN_M = 0.01
TRAY_FOV_CLEARANCE_SCALE = 1.05
VERTICALIZE_SPEED_M_S = 0.16
VERTICALIZE_ACCEL_M_S2 = 0.45
MIN_DOWNWARD_OPTICAL_AXIS_DOT = 0.995
MIN_DOWNWARD_TCP_AXIS_DOT = 0.99
MAX_TRAY_CENTERING_TRANSLATION_M = 0.55

# Tray detection
TRAY_PLANE_TOLERANCE_M = 0.005
TRAY_CLOSE_KERNEL_PX = 11
TRAY_MIN_PLANE_PIXELS = 1500
TRAY_SHORT_SIDE_RANGE_M = (0.1, 0.45)
TRAY_LONG_SIDE_RANGE_M = (0.25, 0.7)
TRAY_CANDIDATE_MARGIN_M = 0.003
RELAXED_TRAY_CANDIDATE_MIN_INSIDE_M = 0.0
VERTICAL_TRAY_CANDIDATE_MIN_INSIDE_M = 0.005
VERTICAL_TRAY_MAX_LONG_SIDE_DIFFERENCE_M = 0.08
VERTICAL_TRAY_MAX_SHORT_SIDE_DIFFERENCE_M = 0.06
RELAXED_COARSE_MIN_HEIGHT_M = 0.018
RELAXED_COARSE_MAX_HEIGHT_M = 0.085
RELAXED_COARSE_MIN_POINTS = 40
RELAXED_COARSE_MAX_POINTS = 10000
RELAXED_COARSE_MIN_EXTENT_M = 0.005
RELAXED_COARSE_MAX_EXTENT_M = 0.11
RELAXED_COARSE_DUPLICATE_DISTANCE_M = 0.04
COARSE_CIRCLE_LOCAL_RADIUS_M = 0.05
COARSE_CIRCLE_TOP_BAND_M = 0.006
COARSE_CIRCLE_MIN_POINTS = 20
COARSE_CIRCLE_DIAMETER_RANGE_M = (0.01, 0.05)
COARSE_CIRCLE_MAX_ASPECT = 2.5
COARSE_CIRCLE_MIN_CONVEX_HULL_CIRCULARITY = 0.45

# Sample refinement
CENTER_WARMUP_VALID_FRAMES = 2
CENTER_CAPTURE_VALID_FRAMES = 7
CENTER_CAPTURE_TIMEOUT_S = 8.0
GLOBAL_WARMUP_VALID_FRAMES = 2
GLOBAL_CAPTURE_VALID_FRAMES = 9
REFINE_LOCAL_RADIUS_M = 0.05
REFINE_TOP_BAND_M = 0.007
REFINE_MIN_TOP_POINTS = 30
REFINE_MIN_HEIGHT_ABOVE_TRAY_M = 0.02
REFINE_MAX_SHIFT_FROM_COARSE_M = 0.05
REFINE_MAX_CENTER_ESTIMATOR_DISAGREEMENT_M = 0.005
REFINE_COMPONENT_CLOSE_KERNEL_PX = 3
REFINE_COMPONENT_MIN_SEED_POINTS = 12
REFINE_FRAGMENT_MERGE_DISTANCE_M = 0.018
REFINE_CIRCLE_RADIUS_RANGE_M = (0.008, 0.022)
REFINE_CIRCLE_MIN_HULL_POINTS = 8
REFINE_CIRCLE_MIN_COVERAGE_DEG = 210.0
REFINE_CIRCLE_MAX_RMS_RESIDUAL_M = 0.0015
robot.JOINT_SPEED_RAD_S = 0.7
robot.JOINT_ACCEL_RAD_S2 = 1.0
robot.TRANSFER_SPEED_M_S = 0.3
robot.TRANSFER_ACCEL_M_S2 = 0.65
robot.VERTICAL_SPEED_M_S = 0.08
robot.VERTICAL_ACCEL_M_S2 = 0.25
vision.VIEW_SPEED_M_S = 0.2
vision.VIEW_ACCEL_M_S2 = 0.5


def _is_transient_camera_timeout(error: Exception) -> bool:
    message = str(error).lower()
    return isinstance(error, TimeoutError) or any(
        marker in message
        for marker in (
            "warming up synchronized rgb-d frames",
            "valid aligned frames",
            "wait_for_frames",
            "frame data timed out",
            "camera reader error",
            "camera reader thread stopped",
        )
    )


class PersistentRGBDCamera:
    """One pipeline plus one continuous frame-consumer thread."""

    def __init__(self) -> None:
        self.pipeline = None
        self.align_filter = None
        self.ob_format = None
        self.generation = 0
        self.frame_sequence = 0
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._reader_thread = None
        self._reader_error = None
        self._request_active = False
        self._request_warmup_remaining = 0
        self._request_capture_count = 0
        self._request_records = []
        self._request_start_sequence = 0

    def open(self) -> None:
        if self.pipeline is not None:
            return
        pipeline, align_filter, ob_format = vision.open_camera_pipeline()
        self.pipeline = pipeline
        self.align_filter = align_filter
        self.ob_format = ob_format
        self.generation += 1
        self._reader_error = None
        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, name='gemini305_rgbd_reader', daemon=True)
        self._reader_thread.start()
        print(
            f"Persistent RGB-D camera reader ready; generation "
            f"{self.generation}; frames are consumed continuously.",
            flush=True,
        )

    def close(self, *, announce: bool=True) -> None:
        pipeline = self.pipeline
        thread = self._reader_thread
        self._stop_event.set()
        with self._condition:
            self._request_active = False
            self._request_records = []
            self._condition.notify_all()
        if thread is not None:
            thread.join(timeout=vision.FRAME_WAIT_MS / 1000.0 + 0.8)
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self.pipeline = None
        self.align_filter = None
        self.ob_format = None
        self._reader_thread = None
        if pipeline is not None and announce:
            print('Persistent RGB-D camera reader closed at program exit.', flush=True)

    def _reader_loop(self) -> None:
        """Own all wait_for_frames calls, including while the robot moves."""
        while not self._stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(vision.FRAME_WAIT_MS)
            except RuntimeError as error:
                if self._stop_event.is_set():
                    break
                if 'timed out' in str(error).lower():
                    continue
                with self._condition:
                    self._reader_error = error
                    self._condition.notify_all()
                continue
            except Exception as error:
                if self._stop_event.is_set():
                    break
                with self._condition:
                    self._reader_error = error
                    self._condition.notify_all()
                continue
            if frames is None:
                continue
            with self._condition:
                self.frame_sequence += 1
                request_active = self._request_active
            if not request_active:
                continue
            try:
                aligned = vision.aligned_frameset(self.align_filter, frames)
                if aligned is None:
                    continue
                color, depth_mm, scale = vision.extract_aligned_rgbd(aligned, self.ob_format)
            except RuntimeError:
                continue
            with self._condition:
                if not self._request_active:
                    continue
                if self._request_warmup_remaining > 0:
                    self._request_warmup_remaining -= 1
                else:
                    self._request_records.append((color.copy(), depth_mm.copy(), float(scale)))
                self._condition.notify_all()

    def _collect_fresh_rgbd(self, *, no_progress_timeout_s: float | None=None) -> tuple[np.ndarray, np.ndarray, dict]:
        warmup_count = int(vision.WARMUP_VALID_FRAME_COUNT)
        capture_count = int(vision.CAPTURE_VALID_FRAME_COUNT)
        timeout_s = float(vision.TOTAL_CAPTURE_TIMEOUT_S)
        deadline = time.monotonic() + timeout_s
        with self._condition:
            if self._request_active:
                raise RuntimeError('Concurrent RGB-D capture request is not allowed')
            self._reader_error = None
            self._request_active = True
            self._request_warmup_remaining = warmup_count
            self._request_capture_count = capture_count
            self._request_records = []
            self._request_start_sequence = self.frame_sequence
            progress_marker = (0, 0)
            last_progress_at = time.monotonic()
            while len(self._request_records) < capture_count:
                if self._reader_error is not None:
                    error = self._reader_error
                    self._request_active = False
                    self._request_records = []
                    raise RuntimeError(f'Camera reader error: {error}')
                if self._reader_thread is None or not self._reader_thread.is_alive():
                    self._request_active = False
                    self._request_records = []
                    raise RuntimeError('Camera reader thread stopped unexpectedly')
                now = time.monotonic()
                warmup_completed = warmup_count - self._request_warmup_remaining
                current_marker = (warmup_completed, len(self._request_records))
                if current_marker != progress_marker:
                    progress_marker = current_marker
                    last_progress_at = now
                if no_progress_timeout_s is not None and now - last_progress_at >= no_progress_timeout_s:
                    received = len(self._request_records)
                    raw_frames = self.frame_sequence - self._request_start_sequence
                    self._request_active = False
                    self._request_records = []
                    raise TimeoutError(
                        f"No aligned-frame progress for "
                        f"{no_progress_timeout_s:.1f} s: raw frames="
                        f"{raw_frames}, warmup={warmup_completed}/"
                        f"{warmup_count}, captured={received}/"
                        f"{capture_count}"
                    )
                remaining = deadline - now
                if remaining <= 0.0:
                    received = len(self._request_records)
                    self._request_active = False
                    self._request_records = []
                    raise TimeoutError(
                        f"Only received {received}/{capture_count} fresh "
                        "aligned frames from the continuous reader"
                    )
                wait_s = min(0.5, remaining)
                if no_progress_timeout_s is not None:
                    wait_s = min(wait_s, max(0.01, no_progress_timeout_s - (now - last_progress_at)))
                self._condition.wait(timeout=wait_s)
            records = list(self._request_records[:capture_count])
            start_sequence = self._request_start_sequence
            end_sequence = self.frame_sequence
            self._request_active = False
            self._request_records = []
        colors = [item[0] for item in records]
        depths = [item[1] for item in records]
        scales = [item[2] for item in records]
        depth_stack = np.stack(depths, axis=0)
        valid_count = np.count_nonzero(depth_stack > 0.0, axis=0)
        with warnings.catch_warnings(), np.errstate(invalid='ignore'):
            warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
            depth_stack[depth_stack <= 0.0] = np.nan
            median_depth = np.nanmedian(depth_stack, axis=0).astype(np.float32)
        median_depth[valid_count < max(3, capture_count // 3)] = 0.0
        median_depth[~np.isfinite(median_depth)] = 0.0
        middle_color = colors[len(colors) // 2]
        positive_observations = valid_count[valid_count > 0]
        info = {
            "warmup_valid_frames": warmup_count,
            "captured_valid_frames": capture_count,
            "depth_scale_median": float(np.median(scales)),
            "valid_depth_fraction": float(np.mean(median_depth > 0.0)),
            "median_valid_observations_per_pixel": float(
                np.median(positive_observations)
                if positive_observations.size
                else 0.0
            ),
            "reader_sequence_start": int(start_sequence),
            "reader_sequence_end": int(end_sequence),
        }
        return (middle_color, median_depth, info)

    @staticmethod
    def _snapshot(receiver, tcp_offset: np.ndarray) -> dict:
        if hasattr(receiver, 'snapshot'):
            return receiver.snapshot(vision.robot_snapshot, tcp_offset, reconnect_on_stale=True)
        return vision.robot_snapshot(receiver, tcp_offset)

    def _wait_for_stationary_robot(self, label: str, receiver: PersistentRTDEReceive) -> tuple[np.ndarray, dict, dict, dict]:
        """Require a continuous stationary interval instead of a fixed sleep."""
        tcp_offset, tcp_spread = vision.read_active_tcp_offset()
        started = time.monotonic()
        deadline = started + CAMERA_STATIONARY_TIMEOUT_S
        stable_since = None
        last_linear = float('inf')
        last_angular = float('inf')
        while True:
            robot = self._snapshot(receiver, tcp_offset)
            speed = np.asarray(robot['actual_tcp_speed_m_rad_s'], dtype=np.float64)
            last_linear = float(np.linalg.norm(speed[:3]))
            last_angular = float(np.linalg.norm(speed[3:]))
            now = time.monotonic()
            stationary = (
                last_linear <= CAMERA_MAX_TCP_LINEAR_SPEED_M_S
                and last_angular <= CAMERA_MAX_TCP_ANGULAR_SPEED_RAD_S
            )
            if stationary:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= CAMERA_STATIONARY_HOLD_S:
                    diagnostics = {
                        "stationary_wait_s": float(now - started),
                        "stationary_hold_s": float(now - stable_since),
                        "tcp_linear_speed_m_s": last_linear,
                        "tcp_angular_speed_rad_s": last_angular,
                    }
                    return (tcp_offset, tcp_spread, robot, diagnostics)
            else:
                stable_since = None
            if now >= deadline:
                raise RuntimeError(
                    f"Robot did not settle before {label} capture within "
                    f"{CAMERA_STATIONARY_TIMEOUT_S:.1f} s; TCP speed is "
                    f"{last_linear*1000.0:.2f} mm/s and "
                    f"{last_angular:.4f} rad/s"
                )
            time.sleep(CAMERA_STATIONARY_POLL_S)

    def _capture_once(self, label: str, receiver: PersistentRTDEReceive, *, no_progress_timeout_s: float | None=None):
        if self.pipeline is None:
            self.open()
        tcp_start, spread_start, start_robot, stationary_info = self._wait_for_stationary_robot(label, receiver)
        color, depth_mm, capture_info = self._collect_fresh_rgbd(no_progress_timeout_s=no_progress_timeout_s)
        tcp_end, spread_end = vision.read_active_tcp_offset()
        end_robot = self._snapshot(receiver, tcp_end)
        if np.max(np.abs(tcp_end - tcp_start)) > 1e-07:
            raise RuntimeError(f'Active TCP changed during {label} capture')
        position_change_mm = float(
            np.linalg.norm(
                start_robot["T_base_flange"][:3, 3]
                - end_robot["T_base_flange"][:3, 3]
            )
            * 1000.0
        )
        orientation_change_deg = vision.rotation_difference_deg(
            start_robot["T_base_flange"], end_robot["T_base_flange"]
        )
        if position_change_mm > 0.5 or orientation_change_deg > 0.15:
            raise RuntimeError(
                f"Robot moved during {label} capture: "
                f"{position_change_mm:.3f} mm, "
                f"{orientation_change_deg:.4f} deg"
            )
        capture_info.update(
            {
                "label": label,
                "robot_position_change_mm": position_change_mm,
                "robot_orientation_change_deg": orientation_change_deg,
                "tcp_start_read_spread": spread_start,
                "tcp_end_read_spread": spread_end,
                "persistent_camera_generation": self.generation,
                "continuous_reader": True,
                "stationary_gate": stationary_info,
            }
        )
        return (color, depth_mm, capture_info, start_robot)

    def capture(
        self, label: str, receiver: PersistentRTDEReceive, *,
        recovery_attempts: int = CAMERA_RECOVERY_ATTEMPTS,
        no_progress_timeout_s: float | None = None,
    ):
        if recovery_attempts < 1:
            raise ValueError('recovery_attempts must be at least 1')
        last_error = None
        for attempt in range(1, recovery_attempts + 1):
            try:
                result = self._capture_once(label, receiver, no_progress_timeout_s=no_progress_timeout_s)
                if attempt > 1:
                    print(f'CAMERA RECOVERED: {label} succeeded after {attempt - 1} exceptional pipeline restart(s).', flush=True)
                return result
            except Exception as error:
                if not _is_transient_camera_timeout(error):
                    raise
                last_error = error
                if attempt >= recovery_attempts:
                    break
                print(
                    f"WARNING: continuous camera reader failed during "
                    f"{label}: {error}; restarting pipeline "
                    f"{attempt}/{recovery_attempts - 1} in "
                    f"{CAMERA_RECOVERY_DELAY_S:.1f} s.",
                    flush=True,
                )
                self.close(announce=False)
                gc.collect()
                time.sleep(CAMERA_RECOVERY_DELAY_S)
        raise TimeoutError(f'{label} failed after continuous-reader recovery: {last_error}')


def set_rg2_command_register(command: int) -> None:
    """Write one RG2 command while no RTDE control session is active."""
    from rtde_io import RTDEIOInterface
    last_error = None
    for attempt in range(1, RG2_REGISTER_WRITE_ATTEMPTS + 1):
        io = None
        try:
            io = RTDEIOInterface(robot.ROBOT_IP, False, False)
            if not io.setInputIntRegister(RG2_COMMAND_REGISTER, int(command)):
                raise RuntimeError(f'Controller rejected input integer register {RG2_COMMAND_REGISTER}={command}')
            return
        except Exception as error:
            last_error = error
            if attempt < RG2_REGISTER_WRITE_ATTEMPTS:
                time.sleep(0.5 * attempt)
        finally:
            if io is not None:
                try:
                    io.disconnect()
                except Exception:
                    pass
    raise RuntimeError(
        f"Could not write RG2 command register after "
        f"{RG2_REGISTER_WRITE_ATTEMPTS} attempts: {last_error}"
    )


def wait_dashboard_stopped(timeout_s: float, context: str) -> str:
    """Wait until Dashboard reports STOPPED, using a fresh short connection."""
    deadline = time.monotonic() + timeout_s
    last_state = 'unavailable'
    last_error = None
    while time.monotonic() < deadline:
        sock = stream = None
        try:
            sock, stream, _ = robot.dashboard_session()
            stream.write(b'programState\n')
            last_state = stream.readline().decode('utf-8', errors='replace').strip()
            if last_state.upper().startswith('STOPPED'):
                return last_state
        except Exception as error:
            last_error = error
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        time.sleep(0.25)
    detail = last_state if last_error is None else f'{last_state}; {last_error}'
    raise TimeoutError(f'Dashboard did not reach STOPPED during {context}: {detail}')


def prepare_rg2_command_program() -> None:
    """Load the single RG2 command program exactly once for this run."""
    sock = stream = None
    try:
        sock, stream, banner = robot.dashboard_session()
        print(banner, flush=True)
        remote = robot.dashboard_command(stream, 'is in remote control')
        if remote.strip().lower() != 'true':
            raise RuntimeError('Robot is not in Remote Control mode')
        state = robot.dashboard_command(stream, 'programState')
        if not state.upper().startswith('STOPPED'):
            raise RuntimeError(f'Another program is active before RG2 setup: {state}')
        loaded = robot.dashboard_command(stream, f'load {RG2_COMMAND_PROGRAM}')
        if 'loading program' not in loaded.lower():
            raise RuntimeError(f'RG2 command program did not load: {loaded}')
    finally:
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
    time.sleep(RG2_LOAD_SETTLE_S)
    wait_dashboard_stopped(RG2_PROGRAM_TIMEOUT_S, 'initial RG2 program load')
    set_rg2_command_register(RG2_COMMAND_IDLE)
    print(f'RG2 command program loaded once: {RG2_COMMAND_PROGRAM}', flush=True)


def run_loaded_rg2_command(action: str, receiver: PersistentRTDEReceive) -> None:
    """Run one RG2 request and prove it completed through the robot ACK."""
    commands = {
        "OPEN": (
            RG2_COMMAND_OPEN,
            RG2_ACK_OPEN_ACTIVE,
            RG2_ACK_OPEN_DONE,
        ),
        "CLOSE": (
            RG2_COMMAND_CLOSE,
            RG2_ACK_CLOSE_ACTIVE,
            RG2_ACK_CLOSE_DONE,
        ),
    }
    action_upper = action.upper()
    if action_upper not in commands:
        raise ValueError(f'Unknown RG2 action: {action}')
    if not hasattr(receiver, 'read_output_int_register'):
        raise RuntimeError(
            "robot.py does not provide "
            "read_output_int_register(); use the V8 file supplied with this "
            "pick_place_return.py"
        )
    command, active_ack, done_ack = commands[action_upper]
    previous_ack = receiver.read_output_int_register(RG2_ACK_REGISTER, reconnect_on_stale=True)
    set_rg2_command_register(command)
    print(
        f"RG2 {action_upper}: input register {RG2_COMMAND_REGISTER}={command}; "
        f"expect output ACK {active_ack}->{done_ack}; play preloaded "
        f"{RG2_COMMAND_PROGRAM}",
        flush=True,
    )
    sock = stream = None
    program_started = False
    try:
        sock, stream, banner = robot.dashboard_session()
        print(banner, flush=True)
        remote = robot.dashboard_command(stream, 'is in remote control')
        if remote.strip().lower() != 'true':
            raise RuntimeError('Robot is not in Remote Control mode')
        state = robot.dashboard_command(stream, 'programState')
        if not state.upper().startswith('STOPPED'):
            raise RuntimeError(f'Program was not stopped before RG2 {action_upper}: {state}')
        selected = robot.dashboard_command(stream, 'get loaded program')
        if 'rg2_command.urp' not in selected.lower():
            raise RuntimeError(f'The preloaded program changed; refusing to play an unknown program: {selected}')
        played = robot.dashboard_command(stream, 'play')
        if 'starting program' not in played.lower():
            raise RuntimeError(f'RG2 command program did not start: {played}')
        program_started = True
        start_deadline = time.monotonic() + RG2_PROGRAM_START_TIMEOUT_S
        execution_seen = False
        last_state = state
        last_ack = previous_ack
        while time.monotonic() < start_deadline:
            last_state = robot.dashboard_command(stream, 'programState')
            last_ack = receiver.read_output_int_register(RG2_ACK_REGISTER, reconnect_on_stale=True)
            state_playing = last_state.upper().startswith('PLAYING')
            active_transition = last_ack == active_ack
            completed_transition = last_ack == done_ack and previous_ack != done_ack
            if state_playing or active_transition or completed_transition:
                execution_seen = True
                break
            time.sleep(RG2_ACK_POLL_S)
        if not execution_seen:
            raise TimeoutError(
                f"RG2 {action_upper} never proved program execution; "
                f"last programState={last_state!r}, output ACK={last_ack}, "
                f"previous ACK={previous_ack}"
            )
        print(f'RG2 {action_upper} execution verified: programState={last_state}; ACK={last_ack}', flush=True)
        completion_deadline = time.monotonic() + RG2_PROGRAM_TIMEOUT_S
        stopped_without_ack_at = None
        while time.monotonic() < completion_deadline:
            last_ack = receiver.read_output_int_register(RG2_ACK_REGISTER, reconnect_on_stale=True)
            last_state = robot.dashboard_command(stream, 'programState')
            if last_ack == done_ack:
                break
            if last_state.upper().startswith('STOPPED'):
                if stopped_without_ack_at is None:
                    stopped_without_ack_at = time.monotonic()
                elif time.monotonic() - stopped_without_ack_at >= RG2_ACK_AFTER_STOP_GRACE_S:
                    raise RuntimeError(f'RG2 {action_upper} stopped without completion ACK {done_ack}; final ACK={last_ack}')
            else:
                stopped_without_ack_at = None
            time.sleep(RG2_ACK_POLL_S)
        else:
            raise TimeoutError(
                f"RG2 {action_upper} did not produce completion ACK "
                f"{done_ack} within {RG2_PROGRAM_TIMEOUT_S:.1f} s; "
                f"last ACK={last_ack}, programState={last_state}"
            )
        stop_deadline = time.monotonic() + RG2_PROGRAM_TIMEOUT_S
        while time.monotonic() < stop_deadline:
            last_state = robot.dashboard_command(stream, 'programState')
            if last_state.upper().startswith('STOPPED'):
                break
            time.sleep(RG2_ACK_POLL_S)
        else:
            raise TimeoutError(
                f"RG2 {action_upper} produced ACK {done_ack} but the "
                f"PolyScope program did not stop; last state={last_state}"
            )
        final_ack = receiver.read_output_int_register(RG2_ACK_REGISTER, reconnect_on_stale=True)
        if final_ack != done_ack:
            raise RuntimeError(f'RG2 {action_upper} ACK changed after STOPPED: expected {done_ack}, received {final_ack}')
        print(f'Dashboard: programState -> {last_state}', flush=True)
        print(f'RG2 {action_upper} completed and ACK {final_ack} verified', flush=True)
    except BaseException:
        if program_started:
            stop_sock = stop_stream = None
            stop_sent = False
            try:
                if stream is not None:
                    try:
                        robot.dashboard_command(stream, 'stop')
                        stop_sent = True
                    except Exception:
                        stop_sent = False
                if not stop_sent:
                    stop_sock, stop_stream, _ = robot.dashboard_session()
                    robot.dashboard_command(stop_stream, 'stop')
                    stop_sent = True
            except Exception:
                pass
            finally:
                if stop_stream is not None:
                    try:
                        stop_stream.close()
                    except Exception:
                        pass
                if stop_sock is not None:
                    try:
                        stop_sock.close()
                    except Exception:
                        pass
        raise
    finally:
        try:
            set_rg2_command_register(RG2_COMMAND_IDLE)
        except Exception as clear_error:
            print(f'WARNING: could not clear RG2 command register: {clear_error}', file=sys.stderr, flush=True)
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
    time.sleep(RG2_POST_PROGRAM_SETTLE_S)


def start_rtde_control_windows(*, post_rg2: bool=False):
    """Create a fresh RTDE controller, retrying connection setup only.

    The default path is intentionally unchanged.  The longer schedule is used
    only after a completed RG2 program, and therefore cannot delay or alter a
    normal robot move.
    """
    last_error = None
    if post_rg2:
        delays = POST_RG2_RTDE_RECONNECT_DELAYS_S
    else:
        delays = tuple(
            0.0 if attempt == 0 else RTDE_RECONNECT_BASE_DELAY_S * attempt
            for attempt in range(RTDE_RECONNECT_ATTEMPTS)
        )
    attempt_count = len(delays)
    for attempt_index, delay_s in enumerate(delays):
        attempt = attempt_index + 1
        if delay_s > 0.0:
            if post_rg2:
                print(
                    f"RTDE control is still unavailable after RG2; waiting "
                    f"{delay_s:.1f} s before attempt {attempt}/{attempt_count}.",
                    flush=True,
                )
            time.sleep(delay_s)
        try:
            control = robot.start_rtde_control()
            if hasattr(control, 'isConnected') and (not control.isConnected()):
                raise RuntimeError('new RTDE control interface is not connected')
            if attempt > 1:
                print(f'RTDE control connected on attempt {attempt}', flush=True)
            return control
        except Exception as error:
            last_error = error
            print(
                f"RTDE control connection attempt {attempt}/"
                f"{attempt_count} failed: {error}",
                file=sys.stderr,
                flush=True,
            )
    raise RuntimeError(
        f"Could not create a fresh RTDE control interface after "
        f"{attempt_count} attempts: {last_error}"
    )


def stop_rtde_control_windows(control, *, strict: bool=True) -> None:
    """Deliberately end RTDE before any PolyScope RG2 program is played."""
    if control is None:
        return
    robot.stop_rtde_control(control)
    try:
        wait_dashboard_stopped(RG2_PROGRAM_TIMEOUT_S, 'RTDE shutdown')
    except Exception as error:
        if strict:
            raise
        print(f'WARNING: final RTDE shutdown could not be confirmed: {error}', file=sys.stderr, flush=True)


def gripper_action_windows(control, receiver: PersistentRTDEReceive, action: str, *, on_command_confirmed=None):
    """Run one RG2 transition, commit its ACK, then rebuild RTDE Control.

    ``on_command_confirmed`` runs only after ``run_loaded_rg2_command`` has
    verified the action-specific completion ACK and STOPPED state, but before
    a fresh RTDEControlInterface is constructed.  This keeps the physical
    gripper state accurate if the subsequent network reconnection fails.
    """
    stop_rtde_control_windows(control)
    control = None
    run_loaded_rg2_command(action, receiver)
    if on_command_confirmed is not None:
        on_command_confirmed()
    return start_rtde_control_windows(post_rg2=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    """Atomically replace JSON, retrying only a transient Windows file lock."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.{time.monotonic_ns()}.tmp')
    last_error = None
    try:
        temporary.write_text(payload, encoding='utf-8')
        for attempt, delay_s in enumerate(JSON_REPLACE_RETRY_DELAYS_S, start=1):
            if delay_s > 0.0:
                time.sleep(delay_s)
            try:
                os.replace(temporary, path)
                if attempt > 1:
                    print(f'JSON CHECKPOINT RECOVERED: {path.name} was replaced after {attempt} attempts.', flush=True)
                return
            except PermissionError as error:
                last_error = error
                if attempt == 1:
                    print(f'WARNING: Windows temporarily locked {path.name}; retrying the atomic checkpoint replace.', flush=True)
        raise PermissionError(
            f"Windows kept {path.name} locked for "
            f"{sum(JSON_REPLACE_RETRY_DELAYS_S):.2f} s; atomic checkpoint "
            "replacement was not completed"
        ) from last_error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def update_recovery_checkpoint(state: dict, **changes) -> None:
    """Persist only application state; native RTDE objects are never reused."""
    state.update(changes)
    state['updated_at_utc'] = utc_now()
    write_json(RECOVERY_JSON, state)


def recover_empty_robot_to_saved_home(home_q_rad) -> None:
    """One-shot recovery used only by a new child when no sample is held.

    The interrupted RTDEControlInterface is deliberately not reused.  A fresh
    Receive and Control pair reads the real stopped pose, raises vertically to
    clearance without descending, and then returns to the saved Home joints.
    """
    home_q = [float(value) for value in home_q_rad]
    if len(home_q) != 6 or not np.all(np.isfinite(home_q)):
        raise ValueError('Recovery Home joint vector must contain six finite values')
    receiver = None
    control = None
    try:
        robot.verify_remote_and_stopped()
        receiver = PersistentRTDEReceive(
            robot.ROBOT_IP,
            frequency_hz=125.0,
            connect_attempts=RTDE_RECONNECT_ATTEMPTS,
            freshness_timeout_s=0.60,
        )
        control = start_rtde_control_windows()
        actual = receiver.read_basic_state(reconnect_on_stale=True)
        actual_pose = [float(value) for value in actual['pose']]
        clearance_pose = list(actual_pose)
        clearance_pose[2] = max(float(actual_pose[2]), RECOVERY_CLEARANCE_Z_M, float(robot.MIN_TRANSFER_Z_M))
        print('EMPTY RECOVERY: fresh RTDE connections verified; sample_held=false.', flush=True)
        if clearance_pose[2] > actual_pose[2] + 0.001:
            robot.move_l(
                control,
                clearance_pose,
                robot.VERTICAL_SPEED_M_S,
                robot.VERTICAL_ACCEL_M_S2,
                "EMPTY_RECOVERY_VERTICAL_CLEARANCE",
            )
        robot.move_j(control, home_q, 'EMPTY_RECOVERY_SAVED_HOME_Q')
        print(
            "EMPTY RECOVERY COMPLETE: robot reached the saved Home joints; "
            "the interrupted cycle will be rediscovered from the beginning.",
            flush=True,
        )
    finally:
        stop_rtde_control_windows(control, strict=False)
        if receiver is not None:
            receiver.close()


def load_calibrated_workbench() -> tuple[list[float], dict]:
    """Load the saved refined top centre without re-detecting the workbench."""
    if not WORKBENCH_JSON.is_file():
        raise FileNotFoundError(f'Missing {WORKBENCH_JSON.name}; run calibrate_workbench.py first')
    record = json.loads(WORKBENCH_JSON.read_text(encoding='utf-8-sig'))
    center = np.asarray(record.get('center_base_m'), dtype=np.float64).reshape(-1)
    refined = record.get('refined_top', {})
    if center.size != 3 or not np.all(np.isfinite(center)):
        raise RuntimeError('workbench_pose.json has no finite 3-D center_base_m')
    normal = np.asarray(record.get('normal_base', [0.0, 0.0, 1.0]), dtype=np.float64)
    sides_mm = np.asarray(refined.get('top_size_mm', [30.0, 30.0]), dtype=np.float64)
    place = center.astype(float).tolist()
    diagnostics = {
        "method": "fixed RGB-D calibrated workbench top centre",
        "source_file": WORKBENCH_JSON.name,
        "calibration_status": record.get("status"),
        "support_point_base_m": place,
        "top_size_mm": sides_mm.astype(float).tolist(),
        "nominal_top_side_mm": WORKBENCH_NOMINAL_TOP_SIDE_M * 1000.0,
        "plane_rms_mm": refined.get("plane_rms_mm"),
        "top_tilt_deg": refined.get("top_tilt_deg"),
        "center_estimator_disagreement_mm": refined.get(
            "center_estimator_disagreement_mm"
        ),
        "coarse_to_refined_xy_shift_mm": record.get(
            "coarse_to_refined_xy_shift_mm"
        ),
        "estimated_height_above_table_mm": record.get(
            "estimated_height_above_table_mm"
        ),
        "normal_base": normal.astype(float).tolist(),
        "runtime_redetection_performed": False,
    }
    return (place, diagnostics)


def project_base_point_to_color(point_base_m: list[float], robot: dict, image_shape: tuple[int, ...]) -> list[int] | None:
    """Project the saved Base point only for the diagnostic image."""
    point = np.append(np.asarray(point_base_m, dtype=np.float64), 1.0)
    point_camera = vision.inverse_transform(robot['T_base_camera']) @ point
    if not np.all(np.isfinite(point_camera)) or point_camera[2] <= 0.0:
        return None
    pixels, _ = cv2.projectPoints(
        point_camera[:3].reshape(1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        vision.COLOR_K,
        vision.COLOR_D,
    )
    u, v = pixels.reshape(2)
    height, width = image_shape[:2]
    if not (0.0 <= u < width and 0.0 <= v < height):
        return None
    return [int(round(float(u))), int(round(float(v)))]


def locate_tray_from_oblique(color: np.ndarray, depth_mm: np.ndarray, robot: dict) -> tuple[dict, dict]:
    """Locate the large tray plane; never infer a sample from this view."""
    x_base, y_base, z_base, _, _ = vision.depth_to_base_maps(depth_mm, robot['T_base_camera'])
    workspace = vision.workspace_mask(x_base, y_base, z_base)
    planes = vision.infer_scene_planes(z_base, workspace)
    tray_z = float(planes['tray_upper_z_m'])
    raw_plane = workspace & (np.abs(z_base - tray_z) <= TRAY_PLANE_TOLERANCE_M)
    mask = raw_plane.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TRAY_CLOSE_KERNEL_PX, TRAY_CLOSE_KERNEL_PX))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    possible = []
    for label in range(1, count):
        connected = labels == label
        valid = connected & raw_plane
        pixel_count = int(np.count_nonzero(valid))
        if pixel_count < TRAY_MIN_PLANE_PIXELS:
            continue
        points_mm = np.column_stack((x_base[valid], y_base[valid])).astype(np.float32) * 1000.0
        if not np.all(np.isfinite(points_mm)):
            continue
        rectangle = cv2.minAreaRect(points_mm.reshape(-1, 1, 2))
        side_a = float(rectangle[1][0]) / 1000.0
        side_b = float(rectangle[1][1]) / 1000.0
        short_side, long_side = sorted((side_a, side_b))
        if (
            not (
            TRAY_SHORT_SIDE_RANGE_M[0]
            <= short_side
            <= TRAY_SHORT_SIDE_RANGE_M[1]
            and TRAY_LONG_SIDE_RANGE_M[0]
            <= long_side
            <= TRAY_LONG_SIDE_RANGE_M[1]
                    )
        ):
            continue
        box_m = cv2.boxPoints(rectangle).astype(np.float64) / 1000.0
        edge_vectors = np.roll(box_m, -1, axis=0) - box_m
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        long_edge = edge_vectors[int(np.argmax(edge_lengths))]
        long_axis_xy = long_edge / np.linalg.norm(long_edge)
        bgr = color[valid]
        possible.append(
            {
                "center_base_xy_m": [
                    float(rectangle[0][0]) / 1000.0,
                    float(rectangle[0][1]) / 1000.0,
                ],
                "rectangle_base_xy_m": box_m.tolist(),
                "long_axis_base_xy": long_axis_xy.astype(float).tolist(),
                "short_side_m": short_side,
                "long_side_m": long_side,
                "tray_upper_z_m": tray_z,
                "plane_pixel_count": pixel_count,
                "pixel_bbox_xywh": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                "mean_bgr": np.mean(bgr, axis=0).astype(float).tolist(),
                "selection_score": float(short_side * long_side * pixel_count),
            }
        )
    if not possible:
        raise RuntimeError(
            "The oblique RGB-D view did not contain one complete tray-sized plane. "
            "No sample was inferred and no robot motion was sent"
        )
    possible.sort(key=lambda item: float(item['selection_score']), reverse=True)
    tray = possible[0]
    tray['other_tray_sized_planes_rejected'] = len(possible) - 1
    return (tray, planes)


def confirm_tray_from_vertical_capture(color: np.ndarray, depth_mm: np.ndarray, robot: dict, oblique_tray: dict) -> dict:
    """Re-measure the tray in the current vertical candidate frame.

    The same depth-plane implementation is valid from the vertical view, where
    the complete tray is visible with less perspective compression.  The
    earlier oblique rectangle remains an independent identity/sanity check; a
    large disagreement blocks all candidate motion instead of weakening the
    ROI.  This function changes no sample segmentation or robot pose.
    """
    vertical_tray, _ = locate_tray_from_oblique(color, depth_mm, robot)
    old_center = np.asarray(oblique_tray['center_base_xy_m'], dtype=np.float64)
    new_center = np.asarray(vertical_tray['center_base_xy_m'], dtype=np.float64)
    center_difference = float(np.linalg.norm(new_center - old_center))
    long_difference = abs(float(vertical_tray['long_side_m']) - float(oblique_tray['long_side_m']))
    short_difference = abs(float(vertical_tray['short_side_m']) - float(oblique_tray['short_side_m']))
    diagnostics = {
        "method": "same_vertical_rgbd_frame_horizontal_tray_plane",
        "oblique_to_vertical_center_difference_mm": (
            center_difference * 1000.0
        ),
        "oblique_to_vertical_long_side_difference_mm": (
            long_difference * 1000.0
        ),
        "oblique_to_vertical_short_side_difference_mm": (
            short_difference * 1000.0
        ),
        "required_candidate_inside_margin_mm": (
            VERTICAL_TRAY_CANDIDATE_MIN_INSIDE_M * 1000.0
        ),
    }
    vertical_tray['vertical_confirmation'] = diagnostics
    if long_difference > VERTICAL_TRAY_MAX_LONG_SIDE_DIFFERENCE_M:
        raise RuntimeError(
            "Current vertical tray long side disagrees with the oblique tray "
            f"by {long_difference * 1000.0:.1f} mm; all sample candidates are "
            "blocked before motion"
        )
    if short_difference > VERTICAL_TRAY_MAX_SHORT_SIDE_DIFFERENCE_M:
        raise RuntimeError(
            "Current vertical tray short side disagrees with the oblique tray "
            f"by {short_difference * 1000.0:.1f} mm; all sample candidates are "
            "blocked before motion"
        )
    center_mm = (new_center * 1000.0).round(1).tolist()
    print(
        "VERTICAL TRAY ROI VERIFIED: current-frame centre Base XY "
        f"{center_mm} mm; size "
        f"{vertical_tray['long_side_m'] * 1000.0:.1f} x "
        f"{vertical_tray['short_side_m'] * 1000.0:.1f} mm; candidate centre "
        f"must be >= {VERTICAL_TRAY_CANDIDATE_MIN_INSIDE_M * 1000.0:.1f} "
        "mm inside.",
        flush=True,
    )
    return vertical_tray


def filter_candidates_to_tray(candidates: list[dict], tray: dict, vertical_tray: dict | None=None) -> tuple[list[dict], int]:
    """Require candidates inside both oblique and current vertical tray ROIs."""
    contour_mm = (np.asarray(tray['rectangle_base_xy_m'], dtype=np.float32) * 1000.0).reshape(-1, 1, 2)
    vertical_contour_mm = None
    if vertical_tray is not None:
        vertical_contour_mm = (
            np.asarray(
                vertical_tray["rectangle_base_xy_m"], dtype=np.float32
            )
            * 1000.0
        ).reshape(-1, 1, 2)
    accepted = []
    for item in candidates:
        center_mm = tuple((np.asarray(item['coarse_center_base_m'][:2], dtype=np.float64) * 1000.0).astype(float))
        signed_distance_mm = float(cv2.pointPolygonTest(contour_mm, center_mm, True))
        item = dict(item)
        item['tray_boundary_signed_distance_mm'] = signed_distance_mm
        is_relaxed_recovery = item.get('coarse_detection_method') == 'relaxed_fragment_recovery'
        minimum_signed_distance_mm = (
            RELAXED_TRAY_CANDIDATE_MIN_INSIDE_M * 1000.0
            if is_relaxed_recovery
            else -TRAY_CANDIDATE_MARGIN_M * 1000.0
        )
        item['tray_boundary_required_distance_mm'] = minimum_signed_distance_mm
        passes_oblique_boundary = signed_distance_mm >= minimum_signed_distance_mm
        passes_vertical_boundary = True
        if vertical_contour_mm is not None:
            vertical_signed_distance_mm = float(cv2.pointPolygonTest(vertical_contour_mm, center_mm, True))
            vertical_required_distance_mm = VERTICAL_TRAY_CANDIDATE_MIN_INSIDE_M * 1000.0
            item['vertical_tray_boundary_signed_distance_mm'] = vertical_signed_distance_mm
            item['vertical_tray_boundary_required_distance_mm'] = vertical_required_distance_mm
            passes_vertical_boundary = vertical_signed_distance_mm >= vertical_required_distance_mm
        if passes_oblique_boundary and passes_vertical_boundary:
            accepted.append(item)
    accepted.sort(key=lambda item: float(item['principal_point_distance_px']))
    for candidate_id, item in enumerate(accepted, start=1):
        item['candidate_id'] = candidate_id
    return (accepted, len(candidates) - len(accepted))


def build_wide_record(wide_capture: dict, initial_robot: dict, coarse_candidates: list[dict], planes: dict) -> dict:
    """Build only the fields needed by the tested fixed-place selector."""
    support_z = float(planes['sample_support_z_m'])
    rough_objects = []
    for item in coarse_candidates:
        x, y, _ = item['coarse_center_base_m']
        top_z = float(item['coarse_top_z_base_m'])
        rough_objects.append(
            {
                "candidate_id": int(item["candidate_id"]),
                "center_base_m": [float(x), float(y), (support_z + top_z) / 2.0],
                "support_z_base_m": support_z,
            }
        )
    return {
        "schema": "ur5e_gemini305_rgbd_cylinder_detection_v1",
        "created_at_utc": utc_now(),
        "camera": {
            "K": vision.COLOR_K.tolist(),
            "D_opencv_k1_k2_p1_p2_k3_k4_k5_k6": vision.COLOR_D.tolist(),
            "capture": wide_capture,
        },
        "robot_at_capture": vision.json_ready_snapshot(initial_robot),
        "horizontal_surface_peaks": planes["peaks"],
        "cylinder_candidates": rough_objects,
    }


def discover_global_samples_for_cycle(
    cycle_index: int, camera, receiver, home_robot: dict, tray: dict,
    place: list[float], run: dict, recovery_state: dict,
) -> tuple[list[dict], dict, dict, dict]:
    """Reuse the fixed overview TCP, then rediscover every current sample."""
    run['stage'] = f'cycle_{cycle_index}:fixed_global_overview'
    update_recovery_checkpoint(
        recovery_state,
        active_cycle=cycle_index,
        active_sample=None,
        entered_stage=run["stage"],
        sample_held=False,
        gripper_expected="OPEN",
        gripper_transition_in_progress=False,
        automatic_recovery_allowed=True,
    )
    print(f'[CYCLE {cycle_index}] Move from Home to the fixed global overview TCP.', flush=True)
    overview = run.get('vertical_overview')
    if not isinstance(overview, dict):
        raise RuntimeError('The cycle-1 fixed global overview pose is unavailable')
    lift_pose = overview['lift_pose_base_tcp_m_rad']
    level_pose = overview['level_pose_base_tcp_m_rad']
    overview_pose = overview['pose_base_tcp_m_rad']
    overview_control = None
    try:
        overview_control = start_rtde_control_windows()
        current_pose = initial_onrobot_tcp_pose(home_robot)
        if (
            np.linalg.norm(
            np.asarray(lift_pose[:3], dtype=np.float64)
            - np.asarray(current_pose[:3], dtype=np.float64)
                    ) > 0.001
        ):
            update_recovery_checkpoint(
                recovery_state,
                entered_stage=f"cycle_{cycle_index}:global_lift",
                motion_target_base_tcp_m_rad=[float(value) for value in lift_pose],
            )
            robot.move_l(
                overview_control,
                lift_pose,
                VERTICALIZE_SPEED_M_S,
                VERTICALIZE_ACCEL_M_S2,
                f"CYCLE_{cycle_index}_GLOBAL_LIFT",
            )
            update_recovery_checkpoint(recovery_state, completed_stage=f'cycle_{cycle_index}:global_lift')
        update_recovery_checkpoint(
            recovery_state,
            entered_stage=f"cycle_{cycle_index}:global_level",
            motion_target_base_tcp_m_rad=[float(value) for value in level_pose],
        )
        robot.move_l(
            overview_control,
            level_pose,
            VERTICALIZE_SPEED_M_S,
            VERTICALIZE_ACCEL_M_S2,
            f"CYCLE_{cycle_index}_GLOBAL_LEVEL",
        )
        update_recovery_checkpoint(
            recovery_state,
            completed_stage=f"cycle_{cycle_index}:global_level",
            entered_stage=f"cycle_{cycle_index}:global_overview",
            motion_target_base_tcp_m_rad=[float(value) for value in overview_pose],
        )
        robot.move_l(
            overview_control,
            overview_pose,
            VERTICALIZE_SPEED_M_S,
            VERTICALIZE_ACCEL_M_S2,
            f"CYCLE_{cycle_index}_GLOBAL_OVERVIEW",
        )
        update_recovery_checkpoint(recovery_state, completed_stage=f'cycle_{cycle_index}:global_overview')
    finally:
        stop_rtde_control_windows(overview_control)
    run['stage'] = f'cycle_{cycle_index}:global_wide_capture'
    update_recovery_checkpoint(recovery_state, entered_stage=run['stage'], motion_target_base_tcp_m_rad=None)
    print(f'[CYCLE {cycle_index}] Global RGB-D sample discovery...', flush=True)
    wide_color, wide_depth, wide_capture, initial_robot = capture_global_wide_fast(
        f"cycle_{cycle_index}_batch_wide",
        receiver,
        camera,
    )
    update_recovery_checkpoint(recovery_state, completed_stage=run['stage'])
    optical_axis = np.asarray(initial_robot['T_base_camera'][:3, 2], dtype=np.float64)
    vertical_dot = float(np.dot(optical_axis, [0.0, 0.0, -1.0]))
    onrobot_transform = initial_robot['T_base_flange'] @ vision.pose6_to_transform(robot.ONROBOT_SINGLE_TCP)
    tcp_vertical_dot = float(np.dot(onrobot_transform[:3, 2], [0.0, 0.0, -1.0]))
    if vertical_dot < MIN_DOWNWARD_OPTICAL_AXIS_DOT:
        raise RuntimeError(
            f"Cycle {cycle_index}: camera is not vertical after leveling: "
            f"optical-axis dot={vertical_dot:.6f}"
        )
    if tcp_vertical_dot < MIN_DOWNWARD_TCP_AXIS_DOT:
        raise RuntimeError(
            f"Cycle {cycle_index}: RG2 TCP is not vertical after leveling: "
            f"TCP-axis dot={tcp_vertical_dot:.6f}"
        )
    strict_candidates, planes, _ = vision.coarse_depth_candidates(wide_color, wide_depth, initial_robot)
    candidates, recovered_before_tray_filter, circle_rejected_count = (
        recover_missed_coarse_candidates(
            wide_depth,
            initial_robot,
            strict_candidates,
            planes,
        )
    )
    vertical_tray = confirm_tray_from_vertical_capture(wide_color, wide_depth, initial_robot, tray)
    candidates, outside_tray_rejected_count = filter_candidates_to_tray(candidates, vertical_tray, vertical_tray)
    recovered_count = sum(
        item.get("coarse_detection_method") == "relaxed_fragment_recovery"
        for item in candidates
    )
    if len(candidates) < MIN_BATCH_CANDIDATES:
        raise RuntimeError(
            f"Cycle {cycle_index}: global overview found no valid sample; "
            "the cycle is blocked before any grasp"
        )
    if len(candidates) > 20:
        raise RuntimeError(f'Cycle {cycle_index}: implausible coarse candidate count {len(candidates)}')
    wide_record = build_wide_record(wide_capture, initial_robot, candidates, planes)
    place_pixel = project_base_point_to_color(place, initial_robot, wide_color.shape)
    write_overview(wide_color, candidates, place_pixel)
    current = robot.read_current_robot(wide_record, receiver)
    robot.validate_start(wide_record, current)
    diagnostics = {
        "capture": wide_capture,
        "candidate_count": len(candidates),
        "strict_candidate_count": len(strict_candidates),
        "relaxed_recovered_candidate_count": recovered_count,
        "relaxed_recovered_before_tray_filter": recovered_before_tray_filter,
        "circle_rejected_candidate_count": circle_rejected_count,
        "outside_tray_rejected_candidate_count": outside_tray_rejected_count,
        "vertical_confirmed_tray": vertical_tray,
        "camera_optical_axis_dot_base_minus_z": vertical_dot,
        "onrobot_tcp_axis_dot_base_minus_z": tcp_vertical_dot,
    }
    print(
        f"CYCLE {cycle_index}: globally detected {len(candidates)} sample(s) "
        f"({len(candidates) - recovered_count} strict + "
        f"{recovered_count} recovered).",
        flush=True,
    )
    for item in candidates:
        xyz = (np.asarray(item['coarse_center_base_m'], dtype=np.float64) * 1000.0).round(1).tolist()
        print(f"  cycle {cycle_index} order {item['candidate_id']}: coarse Base {xyz} mm", flush=True)
    return (candidates, planes, initial_robot, diagnostics)


def coarse_top_circle_metrics(center_xy: np.ndarray, top_z: float, x_base: np.ndarray, y_base: np.ndarray, z_base: np.ndarray) -> tuple[bool, dict]:
    """Loose circle check on a horizontal 3-D top slice, not image ellipse."""
    local = (
        np.isfinite(x_base) & np.isfinite(y_base) & np.isfinite(z_base)
        & (np.hypot(x_base - center_xy[0], y_base - center_xy[1])
           <= COARSE_CIRCLE_LOCAL_RADIUS_M)
        & (np.abs(z_base - float(top_z)) <= COARSE_CIRCLE_TOP_BAND_M)
    )
    points = np.column_stack((x_base[local], y_base[local]))
    if points.shape[0] < COARSE_CIRCLE_MIN_POINTS:
        return (False, {'passed': False, 'reason': 'too_few_top_points', 'top_point_count': int(points.shape[0])})
    median = np.median(points, axis=0)
    radial = np.linalg.norm(points - median, axis=1)
    points = points[radial <= min(0.035, float(np.percentile(radial, 98)))]
    if points.shape[0] < COARSE_CIRCLE_MIN_POINTS:
        return False, {
            "passed": False,
            "reason": "too_few_trimmed_top_points",
            "top_point_count": int(points.shape[0]),
        }
    centered = points - np.median(points, axis=0)
    covariance = np.cov(centered, rowvar=False)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return (False, {'passed': False, 'reason': 'invalid_top_covariance'})
    _, vectors = np.linalg.eigh(covariance)
    projected = centered @ vectors
    bounds = np.percentile(projected, [2, 98], axis=0)
    extents = bounds[1] - bounds[0]
    if not np.all(np.isfinite(extents)) or float(np.min(extents)) <= 0.0:
        return (False, {'passed': False, 'reason': 'invalid_top_extents'})
    diameter = float(np.mean(extents))
    aspect = float(np.max(extents) / np.min(extents))
    contour = (points * 1000.0).astype(np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))
    circularity = 4.0 * math.pi * hull_area / (hull_perimeter * hull_perimeter) if hull_perimeter > 0.0 else 0.0
    passed = bool(
        COARSE_CIRCLE_DIAMETER_RANGE_M[0]
        <= diameter
        <= COARSE_CIRCLE_DIAMETER_RANGE_M[1]
        and aspect <= COARSE_CIRCLE_MAX_ASPECT
        and circularity >= COARSE_CIRCLE_MIN_CONVEX_HULL_CIRCULARITY
    )
    return passed, {
        "passed": passed,
        "reason": "ok" if passed else "circle_geometry_out_of_range",
        "top_point_count": int(points.shape[0]),
        "diameter_estimate_mm": diameter * 1000.0,
        "axis_aspect_ratio": aspect,
        "convex_hull_circularity": circularity,
    }


def recover_missed_coarse_candidates(depth_mm: np.ndarray, robot: dict, strict_candidates: list[dict], planes: dict) -> tuple[list[dict], int, int]:
    """Recover fragmented edge samples without changing grasp precision."""
    x_base, y_base, z_base, _, _ = vision.depth_to_base_maps(depth_mm, robot['T_base_camera'])
    finite = np.isfinite(x_base) & np.isfinite(y_base) & np.isfinite(z_base)
    workspace = (
        finite
        & (x_base >= vision.BASE_X_RANGE_M[0])
        & (x_base <= vision.BASE_X_RANGE_M[1])
        & (y_base >= vision.BASE_Y_RANGE_M[0])
        & (y_base <= vision.BASE_Y_RANGE_M[1])
        & (z_base >= vision.BASE_Z_RANGE_M[0])
        & (z_base <= vision.BASE_Z_RANGE_M[1])
    )
    tray_z = float(planes['tray_upper_z_m'])
    mask = (
        workspace
        & (z_base >= tray_z + RELAXED_COARSE_MIN_HEIGHT_M)
        & (z_base <= tray_z + RELAXED_COARSE_MAX_HEIGHT_M)
    ).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    circle_rejected = 0
    for strict in strict_candidates:
        item = dict(strict)
        center_xy = np.asarray(item['coarse_center_base_m'][:2], dtype=np.float64)
        passed, metrics = coarse_top_circle_metrics(
            center_xy,
            float(item["coarse_top_z_base_m"]),
            x_base,
            y_base,
            z_base,
        )
        item['coarse_circle_check'] = metrics
        if passed:
            candidates.append(item)
        else:
            circle_rejected += 1
    recovered = 0
    for label in range(1, count):
        morphology_area = int(stats[label, cv2.CC_STAT_AREA])
        if not RELAXED_COARSE_MIN_POINTS <= morphology_area <= RELAXED_COARSE_MAX_POINTS:
            continue
        component = labels == label
        valid_component = component & finite
        rows, cols = np.nonzero(valid_component)
        valid_area = int(rows.size)
        if not RELAXED_COARSE_MIN_POINTS <= valid_area <= RELAXED_COARSE_MAX_POINTS:
            continue
        points = np.column_stack((x_base[valid_component], y_base[valid_component]))
        center_xy = np.median(points, axis=0)
        centered = points - center_xy
        covariance = np.cov(centered, rowvar=False)
        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            continue
        _, vectors = np.linalg.eigh(covariance)
        projected = centered @ vectors
        bounds = np.percentile(projected, [5, 95], axis=0)
        extents = bounds[1] - bounds[0]
        if (
            not np.all(np.isfinite(extents))
            or float(np.min(extents)) < RELAXED_COARSE_MIN_EXTENT_M
            or float(np.max(extents)) > RELAXED_COARSE_MAX_EXTENT_M
        ):
            continue
        top_z = float(np.percentile(z_base[valid_component], 90))
        circle_passed, circle_metrics = coarse_top_circle_metrics(center_xy, top_z, x_base, y_base, z_base)
        if not circle_passed:
            circle_rejected += 1
            continue
        if (
            any(
            np.linalg.norm(
                center_xy
                - np.asarray(old["coarse_center_base_m"][:2], dtype=np.float64)
            )
            <= RELAXED_COARSE_DUPLICATE_DISTANCE_M
            for old in candidates
                    )
        ):
            continue
        pixel = [float(np.mean(cols)), float(np.mean(rows))]
        principal_distance = float(math.hypot(pixel[0] - vision.COLOR_K[0, 2], pixel[1] - vision.COLOR_K[1, 2]))
        candidates.append(
            {
                "coarse_center_base_m": [
                    float(center_xy[0]),
                    float(center_xy[1]),
                    float(np.median(z_base[valid_component])),
                ],
                "coarse_top_z_base_m": top_z,
                "pixel_centroid": pixel,
                "bbox_xywh_px": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                "component_area_px": valid_area,
                "morphology_component_area_px": morphology_area,
                "top_patch_axis_extents_mm": (extents * 1000.0).tolist(),
                "principal_point_distance_px": principal_distance,
                "coarse_detection_method": "relaxed_fragment_recovery",
                "coarse_circle_check": circle_metrics,
            }
        )
        recovered += 1
    candidates.sort(key=lambda item: float(item['principal_point_distance_px']))
    for candidate_id, item in enumerate(candidates, start=1):
        item['candidate_id'] = candidate_id
        item.setdefault('coarse_detection_method', 'strict_depth_component')
    return (candidates, recovered, circle_rejected)


def desired_batch_view_pose(coarse: dict, initial_robot: dict) -> tuple[np.ndarray, list[float]]:
    """Use the wider, scene-appropriate observation envelope for ALL mode."""
    old_limit = vision.MAX_VIEW_TRANSLATION_M
    vision.MAX_VIEW_TRANSLATION_M = BATCH_MAX_VIEW_TRANSLATION_M
    try:
        return vision.desired_centered_camera_pose(coarse, initial_robot)
    finally:
        vision.MAX_VIEW_TRANSLATION_M = old_limit


def write_overview(color: np.ndarray, candidates: list[dict], place_pixel: list[int] | None) -> None:
    image = vision.make_coarse_view(color, candidates)
    for item in candidates:
        x, y, _, _ = item['bbox_xywh_px']
        method = 'R' if item.get('coarse_detection_method') == 'relaxed_fragment_recovery' else 'S'
        circularity = float(item.get('coarse_circle_check', {}).get('convex_hull_circularity', 0.0))
        cv2.putText(
            image,
            f"{method} circle={circularity:.2f}",
            (int(x), min(image.shape[0] - 8, int(y) + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0) if method == "S" else (0, 180, 255),
            1,
        )
    if place_pixel is not None:
        col, row = (int(place_pixel[0]), int(place_pixel[1]))
        cv2.drawMarker(image, (col, row), (255, 0, 255), cv2.MARKER_CROSS, 32, 3)
        cv2.circle(image, (col, row), 23, (255, 0, 255), 2)
        label_origin = (max(5, col - 155), max(45, row - 18))
        label = 'WORKBENCH PLACE'
    else:
        label_origin = (8, image.shape[0] - 14)
        label = 'WORKBENCH PLACE OUTSIDE THIS IMAGE'
    cv2.putText(image, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
    vision.save_png(OVERVIEW_IMAGE, image)


def initial_onrobot_tcp_pose(initial_robot: dict) -> list[float]:
    transform = initial_robot['T_base_flange'] @ vision.pose6_to_transform(robot.ONROBOT_SINGLE_TCP)
    return vision.transform_to_pose6(transform)


def read_stationary_robot_snapshot(receiver: PersistentRTDEReceive) -> dict:
    """Read the true pre-run pose through the process-wide Receive owner."""
    tcp_offset, tcp_spread = vision.read_active_tcp_offset()
    snapshot = receiver.snapshot(vision.robot_snapshot, tcp_offset, reconnect_on_stale=True)
    speed = np.asarray(snapshot['actual_tcp_speed_m_rad_s'], dtype=np.float64)
    if np.linalg.norm(speed[:3]) > 0.002 or np.linalg.norm(speed[3:]) > 0.02:
        raise RuntimeError('Robot is moving before camera verticalization')
    snapshot['tcp_read_spread'] = float(tcp_spread)
    return snapshot


def vertical_camera_rotation(current_rotation: np.ndarray, image_x_axis_base_xy: list[float] | np.ndarray | None=None) -> np.ndarray:
    """Point optical +Z down and align image X with the tray long axis."""
    current_rotation = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
    z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    current_x = current_rotation[:, 0].copy()
    current_x[2] = 0.0
    if image_x_axis_base_xy is None:
        x_axis = current_x
    else:
        hint = np.asarray(image_x_axis_base_xy, dtype=np.float64).reshape(2)
        x_axis = np.array([hint[0], hint[1], 0.0], dtype=np.float64)
        if np.linalg.norm(current_x) >= 1e-06 and np.dot(x_axis, current_x) < 0.0:
            x_axis = -x_axis
    if np.linalg.norm(x_axis) < 1e-06:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if (
        not np.all(np.isfinite(rotation))
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6
        or np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 1e-6
    ):
        raise RuntimeError('Could not construct a valid vertical camera rotation')
    return rotation


def required_tray_camera_clearance(tray: dict) -> float:
    """Choose height so the aligned tray fits the calibrated color image."""
    width = float(vision.COLOR_WIDTH)
    height = float(vision.COLOR_HEIGHT)
    fx = float(vision.COLOR_K[0, 0])
    fy = float(vision.COLOR_K[1, 1])
    cx = float(vision.COLOR_K[0, 2])
    cy = float(vision.COLOR_K[1, 2])
    usable_x = min(cx, width - 1.0 - cx) - TRAY_IMAGE_MARGIN_PX
    usable_top = cy - TRAY_IMAGE_MARGIN_PX
    usable_bottom = height - 1.0 - cy - TRAY_IMAGE_MARGIN_PX
    if min(usable_x, usable_top, usable_bottom) <= 10.0:
        raise RuntimeError('Configured tray image margin leaves no usable field of view')
    half_long = float(tray['long_side_m']) / 2.0 + TRAY_FOV_METRIC_MARGIN_M
    half_short = float(tray['short_side_m']) / 2.0 + TRAY_FOV_METRIC_MARGIN_M
    upper_extent = half_short + OVERVIEW_CAMERA_IMAGE_DOWN_OFFSET_M
    lower_extent = max(0.0, half_short - OVERVIEW_CAMERA_IMAGE_DOWN_OFFSET_M)
    clearance = max(
        MIN_VERTICAL_CAMERA_CLEARANCE_M,
        half_long * fx / usable_x,
        upper_extent * fy / usable_top,
        lower_extent * fy / usable_bottom,
    ) * TRAY_FOV_CLEARANCE_SCALE
    if clearance > MAX_VERTICAL_CAMERA_CLEARANCE_M:
        raise RuntimeError(
            f"Tray needs {clearance*1000:.1f} mm camera clearance to fit; "
            f"limit is {MAX_VERTICAL_CAMERA_CLEARANCE_M*1000:.0f} mm"
        )
    return float(clearance)


def predicted_tray_bbox_uv(tray: dict, t_base_camera: np.ndarray) -> list[float]:
    """Project the four metric tray corners and enforce an image margin."""
    box_xy = np.asarray(tray['rectangle_base_xy_m'], dtype=np.float64).reshape(4, 2)
    corners_base = np.column_stack((box_xy, np.full(4, float(tray['tray_upper_z_m']), dtype=np.float64)))
    rotation = np.asarray(t_base_camera[:3, :3], dtype=np.float64)
    center = np.asarray(t_base_camera[:3, 3], dtype=np.float64)
    corners_camera = (rotation.T @ (corners_base - center).T).T
    if np.any(corners_camera[:, 2] <= 0.0):
        raise RuntimeError('A tray corner would be behind the vertical camera')
    u = float(vision.COLOR_K[0, 0]) * corners_camera[:, 0] / corners_camera[:, 2] + float(vision.COLOR_K[0, 2])
    v = float(vision.COLOR_K[1, 1]) * corners_camera[:, 1] / corners_camera[:, 2] + float(vision.COLOR_K[1, 2])
    bbox = [float(np.min(u)), float(np.min(v)), float(np.max(u)), float(np.max(v))]
    width = float(vision.COLOR_WIDTH)
    height = float(vision.COLOR_HEIGHT)
    if (
        bbox[0] < TRAY_IMAGE_MARGIN_PX
        or bbox[1] < TRAY_IMAGE_MARGIN_PX
        or bbox[2] > width - 1.0 - TRAY_IMAGE_MARGIN_PX
        or bbox[3] > height - 1.0 - TRAY_IMAGE_MARGIN_PX
    ):
        raise RuntimeError(
            "Predicted vertical view does not contain the complete tray: "
            f"bbox={[round(value, 1) for value in bbox]} px"
        )
    return bbox


def compute_vertical_overview_pose(home_robot: dict, tray: dict) -> tuple[list[float], list[float], list[float], np.ndarray]:
    """Build full 6-D tray-centred, tray-aligned vertical overview poses."""
    if MIN_VERTICAL_OVERVIEW_TCP_Z_M < vision.MIN_SAFE_TCP_Z_M:
        raise RuntimeError('Configured vertical overview TCP height is below safe minimum')
    home_tcp_pose = initial_onrobot_tcp_pose(home_robot)
    lift_pose = list(home_tcp_pose)
    lift_delta = max(0.0, VERTICALIZE_MIN_TCP_Z_M - float(lift_pose[2]))
    lift_pose[2] += lift_delta
    overview_camera = np.eye(4, dtype=np.float64)
    overview_camera[:3, :3] = vertical_camera_rotation(
        home_robot["T_base_camera"][:3, :3],
        tray["long_axis_base_xy"],
    )
    tray_xy = np.asarray(tray['center_base_xy_m'], dtype=np.float64)
    camera_clearance = required_tray_camera_clearance(tray)
    overview_camera[:3, 3] = np.array(
        [
            float(tray_xy[0]),
            float(tray_xy[1]),
            float(tray["tray_upper_z_m"]) + camera_clearance,
        ],
        dtype=np.float64,
    ) + overview_camera[:3, 1] * OVERVIEW_CAMERA_IMAGE_DOWN_OFFSET_M
    centering_translation = float(
        np.linalg.norm(
            overview_camera[:2, 3]
            - np.asarray(home_robot["T_base_camera"][:2, 3], dtype=np.float64)
        )
    )
    if centering_translation > MAX_TRAY_CENTERING_TRANSLATION_M:
        raise RuntimeError(
            f"Tray-centered camera translation is {centering_translation*1000:.1f} mm; "
            f"limit is {MAX_TRAY_CENTERING_TRANSLATION_M*1000:.0f} mm"
        )
    flange = overview_camera @ vision.inverse_transform(vision.T_FLANGE_CAMERA)
    overview_tcp = flange @ vision.pose6_to_transform(robot.ONROBOT_SINGLE_TCP)
    overview_pose = vision.transform_to_pose6(overview_tcp)
    overview_raise = max(0.0, MIN_VERTICAL_OVERVIEW_TCP_Z_M - float(overview_pose[2]))
    if overview_raise > 0.0:
        overview_camera[2, 3] += overview_raise
        overview_tcp[2, 3] += overview_raise
        overview_pose = vision.transform_to_pose6(overview_tcp)
    predicted_bbox = predicted_tray_bbox_uv(tray, overview_camera)
    actual_clearance = float(overview_camera[2, 3] - float(tray['tray_upper_z_m']))
    level_camera = overview_camera.copy()
    level_raise = max(0.0, float(lift_pose[2]) - float(overview_pose[2]))
    level_camera[2, 3] += level_raise
    level_flange = level_camera @ vision.inverse_transform(vision.T_FLANGE_CAMERA)
    level_tcp = level_flange @ vision.pose6_to_transform(robot.ONROBOT_SINGLE_TCP)
    level_pose = vision.transform_to_pose6(level_tcp)
    tray["vertical_view_plan"] = {
        "camera_clearance_above_tray_m": actual_clearance,
        "predicted_tray_bbox_uv": predicted_bbox,
        "image_x_axis_base_xy": overview_camera[:2, 0].astype(float).tolist(),
        "image_y_axis_base_xy": overview_camera[:2, 1].astype(float).tolist(),
    }
    camera_dot = float(np.dot(overview_camera[:3, 2], [0.0, 0.0, -1.0]))
    tcp_dot = float(np.dot(overview_tcp[:3, 2], [0.0, 0.0, -1.0]))
    if camera_dot < MIN_DOWNWARD_OPTICAL_AXIS_DOT:
        raise RuntimeError(f'Vertical camera construction failed: dot={camera_dot:.6f}')
    if tcp_dot < MIN_DOWNWARD_TCP_AXIS_DOT:
        raise RuntimeError(f'RG2 TCP would not be vertical after leveling: dot={tcp_dot:.6f}')
    if not np.all(np.isfinite(level_pose)) or not np.all(np.isfinite(overview_pose)):
        raise RuntimeError('Computed vertical overview TCP pose is not finite')
    return (lift_pose, level_pose, overview_pose, overview_camera)


def verticalize_camera(home_robot: dict, tray: dict, run: dict) -> np.ndarray:
    """Lift first, then level the camera; always stop this RTDE session."""
    lift_pose, level_pose, overview_pose, desired_camera = compute_vertical_overview_pose(home_robot, tray)
    run["vertical_overview"] = {
        "lift_pose_base_tcp_m_rad": lift_pose,
        "level_pose_base_tcp_m_rad": level_pose,
        "pose_base_tcp_m_rad": overview_pose,
        "T_base_camera": desired_camera.tolist(),
        "camera_image_down_offset_m": OVERVIEW_CAMERA_IMAGE_DOWN_OFFSET_M,
        "camera_optical_axis_dot_base_minus_z": float(
            np.dot(desired_camera[:3, 2], [0.0, 0.0, -1.0])
        ),
    }
    write_json(RUN_JSON, run)
    robot.verify_remote_and_stopped()
    view_plan = tray['vertical_view_plan']
    print(
        "Planned full 6-D overview: "
        f"camera clearance {view_plan['camera_clearance_above_tray_m']*1000:.1f} mm; "
        "predicted tray bbox "
        f"{[round(value, 1) for value in view_plan['predicted_tray_bbox_uv']]} px.",
        flush=True,
    )
    print(
        "First motion will lift, translate and rotate the complete TCP pose. "
        "Keep the complete rotation sweep clear and pendant Stop reachable.",
        flush=True,
    )
    for remaining in range(5, 0, -1):
        print(f'  {remaining}', flush=True)
        time.sleep(1.0)
    control = None
    try:
        control = start_rtde_control_windows()
        current_pose = initial_onrobot_tcp_pose(home_robot)
        if np.linalg.norm(np.asarray(lift_pose[:3]) - np.asarray(current_pose[:3])) > 0.001:
            robot.move_l(control, lift_pose, VERTICALIZE_SPEED_M_S, VERTICALIZE_ACCEL_M_S2, 'VERTICALIZE_LIFT')
        robot.move_l(control, level_pose, VERTICALIZE_SPEED_M_S, VERTICALIZE_ACCEL_M_S2, 'VERTICALIZE_CAMERA')
        robot.move_l(
            control,
            overview_pose,
            VERTICALIZE_SPEED_M_S,
            VERTICALIZE_ACCEL_M_S2,
            "VERTICAL_OVERVIEW_HEIGHT",
        )
    finally:
        stop_rtde_control_windows(control)
    return desired_camera


def move_to_centered_view(control, current_pose: list[float], desired_pose: list[float], label: str):
    """Move at observation height; never return through the initial pose."""
    transition = list(current_pose)
    transition[2] = max(float(transition[2]), vision.MIN_SAFE_TCP_Z_M)
    if np.linalg.norm(np.asarray(transition[:3]) - np.asarray(current_pose[:3])) > 0.001:
        robot.move_l(control, transition, vision.VIEW_SPEED_M_S, vision.VIEW_ACCEL_M_S2, f'{label}_LIFT_FOR_VIEW')
    robot.move_l(control, desired_pose, vision.VIEW_SPEED_M_S, vision.VIEW_ACCEL_M_S2, f'{label}_CENTER_CAMERA')
    return desired_pose


def capture_centered_fast(label: str, receiver: PersistentRTDEReceive, camera: PersistentRGBDCamera):
    """Short synchronized capture used only after a sample is centred."""
    detector = vision
    old_warmup = detector.WARMUP_VALID_FRAME_COUNT
    old_capture = detector.CAPTURE_VALID_FRAME_COUNT
    old_timeout = getattr(detector, 'TOTAL_CAPTURE_TIMEOUT_S', None)
    detector.WARMUP_VALID_FRAME_COUNT = CENTER_WARMUP_VALID_FRAMES
    detector.CAPTURE_VALID_FRAME_COUNT = CENTER_CAPTURE_VALID_FRAMES
    if old_timeout is not None:
        detector.TOTAL_CAPTURE_TIMEOUT_S = CENTER_CAPTURE_TIMEOUT_S
    try:
        return camera.capture(
            label,
            receiver,
            recovery_attempts=CENTER_CAMERA_RECOVERY_ATTEMPTS,
            no_progress_timeout_s=CENTER_CAMERA_NO_PROGRESS_TIMEOUT_S,
        )
    finally:
        detector.WARMUP_VALID_FRAME_COUNT = old_warmup
        detector.CAPTURE_VALID_FRAME_COUNT = old_capture
        if old_timeout is not None:
            detector.TOTAL_CAPTURE_TIMEOUT_S = old_timeout


def capture_global_wide_fast(label: str, receiver: PersistentRTDEReceive, camera: PersistentRGBDCamera):
    """Use a short fresh-frame median only for full-tray sample discovery."""
    detector = vision
    old_warmup = detector.WARMUP_VALID_FRAME_COUNT
    old_capture = detector.CAPTURE_VALID_FRAME_COUNT
    detector.WARMUP_VALID_FRAME_COUNT = GLOBAL_WARMUP_VALID_FRAMES
    detector.CAPTURE_VALID_FRAME_COUNT = GLOBAL_CAPTURE_VALID_FRAMES
    try:
        return camera.capture(label, receiver)
    finally:
        detector.WARMUP_VALID_FRAME_COUNT = old_warmup
        detector.CAPTURE_VALID_FRAME_COUNT = old_capture


def convex_hull_xy(points: np.ndarray) -> np.ndarray:
    """Return a deterministic 2-D convex hull without adding a dependency."""
    unique = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if unique.shape[0] < 3:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def cross(origin, first, second) -> float:
        first_delta = first - origin
        second_delta = second - origin
        return float(first_delta[0] * second_delta[1] - first_delta[1] * second_delta[0])
    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def uniformly_sample_closed_polyline(vertices: np.ndarray, spacing_m: float=0.0005) -> np.ndarray:
    """Sample a hull approximately uniformly so dense corners get no bias."""
    vertices = np.asarray(vertices, dtype=np.float64)
    samples = []
    for index in range(vertices.shape[0]):
        start = vertices[index]
        end = vertices[(index + 1) % vertices.shape[0]]
        length = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(length / spacing_m)))
        for fraction in np.arange(count, dtype=np.float64) / count:
            samples.append(start + fraction * (end - start))
    return np.asarray(samples, dtype=np.float64)


def robust_circle_from_top_points(points: np.ndarray) -> dict:
    """Fit a sub-millimetre Base-XY circle to a possibly noisy top outline.

    Circle quality controls whether this centre is used; it is deliberately
    not an object-recognition gate. A failed fit falls back to the independent
    robust-bounds centre in ``refine_selected_top``.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] >= 80:
        seed_center = np.median(points, axis=0)
        seed_radius = np.linalg.norm(points - seed_center, axis=1)
        edge_limit = float(np.percentile(seed_radius, 97.5))
        circle_points = points[seed_radius <= edge_limit]
    else:
        circle_points = points
    hull = convex_hull_xy(circle_points)
    result = {
        "valid": False,
        "center_xy_m": [float("nan"), float("nan")],
        "radius_m": float("nan"),
        "rms_residual_m": float("inf"),
        "angular_coverage_deg": 0.0,
        "hull_points": int(hull.shape[0]),
    }
    if hull.shape[0] < REFINE_CIRCLE_MIN_HULL_POINTS:
        return result
    boundary = uniformly_sample_closed_polyline(hull)
    if boundary.shape[0] < 12:
        return result
    origin = np.mean(boundary, axis=0)
    local = boundary - origin
    design = np.column_stack((2.0 * local[:, 0], 2.0 * local[:, 1], np.ones(boundary.shape[0])))
    target = np.sum(local * local, axis=1)
    solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        return result
    center = origin + solution[:2]
    radius_squared = float(solution[2] + solution[0] * solution[0] + solution[1] * solution[1])
    if radius_squared <= 0.0 or not np.isfinite(radius_squared):
        return result
    radius = math.sqrt(radius_squared)
    for _ in range(8):
        delta_xy = center - boundary
        distances = np.linalg.norm(delta_xy, axis=1)
        usable = distances > 1e-09
        if np.count_nonzero(usable) < 8:
            return result
        residual = distances[usable] - radius
        median_residual = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median_residual)))
        huber = max(0.00015, 2.5 * 1.4826 * mad)
        absolute = np.abs(residual)
        weights = np.ones_like(absolute)
        outside = absolute > huber
        weights[outside] = huber / absolute[outside]
        jacobian = np.column_stack(
            (
                delta_xy[usable, 0] / distances[usable],
                delta_xy[usable, 1] / distances[usable],
                -np.ones(np.count_nonzero(usable)),
            )
        )
        root_weight = np.sqrt(weights)
        step, _, _, _ = np.linalg.lstsq(jacobian * root_weight[:, None], -residual * root_weight, rcond=None)
        center += step[:2]
        radius += float(step[2])
        if float(np.linalg.norm(step)) < 1e-08:
            break
    distances = np.linalg.norm(boundary - center, axis=1)
    residual = distances - radius
    median_residual = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median_residual)))
    inlier_limit = max(0.00075, 3.5 * 1.4826 * mad)
    inliers = np.abs(residual - median_residual) <= inlier_limit
    if np.count_nonzero(inliers) < 8:
        return result
    rms = float(np.sqrt(np.mean(residual[inliers] ** 2)))
    angles = np.sort(
        np.mod(
            np.arctan2(
                boundary[inliers, 1] - center[1],
                boundary[inliers, 0] - center[0],
            ),
            2.0 * math.pi,
        )
    )
    gaps = np.diff(np.concatenate((angles, angles[:1] + 2.0 * math.pi)))
    coverage_deg = math.degrees(2.0 * math.pi - float(np.max(gaps)))
    valid = bool(
        np.all(np.isfinite(center))
        and np.isfinite(radius)
        and REFINE_CIRCLE_RADIUS_RANGE_M[0]
        <= radius
        <= REFINE_CIRCLE_RADIUS_RANGE_M[1]
        and rms <= REFINE_CIRCLE_MAX_RMS_RESIDUAL_M
        and coverage_deg >= REFINE_CIRCLE_MIN_COVERAGE_DEG
    )
    result.update(
        {
            "valid": valid,
            "center_xy_m": center.astype(float).tolist(),
            "radius_m": float(radius),
            "rms_residual_m": rms,
            "angular_coverage_deg": coverage_deg,
        }
    )
    return result


def isolate_selected_top_component(top_mask: np.ndarray, x_base: np.ndarray, y_base: np.ndarray, coarse_xy: np.ndarray) -> tuple[np.ndarray, dict]:
    """Keep only the centred selected top; do not require it to be circular."""
    mask_u8 = top_mask.astype(np.uint8) * 255
    kernel = np.ones((REFINE_COMPONENT_CLOSE_KERNEL_PX, REFINE_COMPONENT_CLOSE_KERNEL_PX), dtype=np.uint8)
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, labels, _, _ = cv2.connectedComponentsWithStats(closed, 8)
    principal = np.array([vision.COLOR_K[0, 2], vision.COLOR_K[1, 2]], dtype=np.float64)
    candidates = []
    for label in range(1, count):
        valid = (labels == label) & top_mask & np.isfinite(x_base) & np.isfinite(y_base)
        rows, cols = np.nonzero(valid)
        if rows.size < REFINE_COMPONENT_MIN_SEED_POINTS:
            continue
        points = np.column_stack((x_base[valid], y_base[valid]))
        center_xy = np.median(points, axis=0)
        pixel = np.array([np.mean(cols), np.mean(rows)], dtype=np.float64)
        coarse_distance = float(np.linalg.norm(center_xy - coarse_xy))
        pixel_distance = float(np.linalg.norm(pixel - principal))
        score = coarse_distance / 0.03 + pixel_distance / 150.0
        score += 1.5 / math.sqrt(float(rows.size))
        candidates.append(
            {
                "label": label,
                "points": points,
                "center_xy": center_xy,
                "score": score,
                "valid_points": int(rows.size),
            }
        )
    if not candidates:
        raise RuntimeError('selected top has no usable connected depth component')
    seed = min(candidates, key=lambda item: item['score'])
    merged_labels = [
        item["label"]
        for item in candidates
        if float(np.linalg.norm(item["center_xy"] - seed["center_xy"]))
        <= REFINE_FRAGMENT_MERGE_DISTANCE_M
    ]
    selected = top_mask & np.isin(labels, merged_labels)
    if np.count_nonzero(selected) < REFINE_MIN_TOP_POINTS:
        selected = top_mask & (labels == seed['label'])
    diagnostics = {
        "connected_component_count": len(candidates),
        "selected_seed_points": seed["valid_points"],
        "merged_fragment_count": len(merged_labels),
    }
    return (selected, diagnostics)


def refine_selected_top(color: np.ndarray, depth_mm: np.ndarray, robot: dict, coarse: dict, planes: dict) -> tuple[dict, np.ndarray]:
    """Measure the top centre of the one already-selected local sample.

    Coarse XY is used only as a navigation and identity prior. The returned
    grasp XY comes from close-view depth points expressed in Base coordinates.
    """
    x_base, y_base, z_base, _, _ = vision.depth_to_base_maps(depth_mm, robot['T_base_camera'])
    finite = np.isfinite(x_base) & np.isfinite(y_base) & np.isfinite(z_base)
    coarse_xy = np.asarray(coarse['coarse_center_base_m'][:2], dtype=np.float64)
    support_z = float(planes['sample_support_z_m'])
    tray_z = float(planes['tray_upper_z_m'])
    local_distance = np.hypot(x_base - coarse_xy[0], y_base - coarse_xy[1])
    local_high = (
        finite
        & (local_distance <= REFINE_LOCAL_RADIUS_M)
        & (z_base >= tray_z + REFINE_MIN_HEIGHT_ABOVE_TRAY_M)
        & (z_base <= support_z + 0.090)
    )
    z_values = z_base[local_high]
    if z_values.size < REFINE_MIN_TOP_POINTS:
        raise RuntimeError(f'selected top has only {z_values.size} valid elevated depth points')
    z_low = float(np.min(z_values))
    z_high = float(np.max(z_values))
    bins = np.arange(z_low - 0.0005, z_high + 0.0015, 0.001)
    if bins.size < 3:
        bins = np.array([z_low - 0.001, z_low, z_low + 0.0015])
    histogram, edges = np.histogram(z_values, bins=bins)
    peak_index = int(np.argmax(histogram))
    peak_z = float((edges[peak_index] + edges[peak_index + 1]) / 2.0)
    top_mask = local_high & (np.abs(z_base - peak_z) <= REFINE_TOP_BAND_M)
    selected_mask, component_diagnostics = isolate_selected_top_component(top_mask, x_base, y_base, coarse_xy)
    rows, cols = np.nonzero(selected_mask)
    valid_area = int(rows.size)
    if valid_area < REFINE_MIN_TOP_POINTS:
        raise RuntimeError(f'selected stable top has only {valid_area} points')
    points = np.column_stack((x_base[selected_mask], y_base[selected_mask]))
    median_center = np.median(points, axis=0)
    radial = np.linalg.norm(points - median_center, axis=1)
    radial_limit = min(0.025, float(np.percentile(radial, 99.5)))
    trimmed = points[radial <= radial_limit]
    if trimmed.shape[0] < REFINE_MIN_TOP_POINTS:
        trimmed = points
    median_center = np.median(trimmed, axis=0)
    centered = trimmed - median_center
    covariance = np.cov(centered, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    projected = centered @ vectors
    bounds = np.percentile(projected, [2, 98], axis=0)
    extents = bounds[1] - bounds[0]
    if not np.all(np.isfinite(extents)) or float(np.min(extents)) <= 0.0:
        raise RuntimeError('selected top has invalid Base-XY extents')
    bounds_center = median_center + (bounds[0] + bounds[1]) / 2.0 @ vectors.T
    circle = robust_circle_from_top_points(trimmed)
    if circle['valid']:
        center_xy = np.asarray(circle['center_xy_m'], dtype=np.float64)
        refinement_method = 'robust_3d_top_circle'
        disagreement = float(np.linalg.norm(center_xy - bounds_center))
        diameter = 2.0 * float(circle['radius_m'])
    else:
        center_xy = bounds_center
        refinement_method = 'robust_bounds_circle_fallback'
        disagreement = float(np.linalg.norm(bounds_center - median_center))
        diameter = float(np.mean(extents))
    if disagreement > REFINE_MAX_CENTER_ESTIMATOR_DISAGREEMENT_M:
        raise RuntimeError(f'top-center estimators disagree by {disagreement * 1000.0:.1f} mm')
    shift = float(np.linalg.norm(center_xy - coarse_xy))
    if shift > REFINE_MAX_SHIFT_FROM_COARSE_M:
        raise RuntimeError(f'refined center moved {shift * 1000.0:.1f} mm from selected target')
    top_z = float(np.median(z_base[selected_mask]))
    height = top_z - support_z
    aspect = float(np.max(extents) / np.min(extents))
    reference_xy = np.mean(points, axis=0)
    pixel_design = np.column_stack((points - reference_xy, np.ones(points.shape[0], dtype=np.float64)))
    pixel_targets = np.column_stack((cols, rows)).astype(np.float64)
    pixel_coefficients, _, pixel_rank, _ = np.linalg.lstsq(pixel_design, pixel_targets, rcond=None)
    if pixel_rank >= 3:
        pixel = np.append(center_xy - reference_xy, 1.0) @ pixel_coefficients
    else:
        pixel = np.array([float(np.median(cols)), float(np.median(rows))])
    if not np.all(np.isfinite(pixel)):
        pixel = np.array([float(np.median(cols)), float(np.median(rows))])
    left, right = (int(np.min(cols)), int(np.max(cols)))
    top, bottom = (int(np.min(rows)), int(np.max(rows)))
    view = color.copy()
    cv2.rectangle(view, (left, top), (right, bottom), (0, 255, 0), 3)
    center_pixel = (int(round(pixel[0])), int(round(pixel[1])))
    cv2.drawMarker(view, center_pixel, (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
    cv2.rectangle(view, (0, 0), (view.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(
        view,
        f"SELECTED TOP | {refinement_method} | check={disagreement*1000:.1f} mm",
        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
    )
    circle_center = np.asarray(circle['center_xy_m'], dtype=np.float64)
    circle_diagnostics = {
        "used": bool(circle["valid"]),
        "center_xy_base_m": (
            circle_center.astype(float).tolist()
            if np.all(np.isfinite(circle_center))
            else None
        ),
        "radius_mm": (
            float(circle["radius_m"]) * 1000.0
            if np.isfinite(circle["radius_m"])
            else None
        ),
        "rms_residual_mm": (
            float(circle["rms_residual_m"]) * 1000.0
            if np.isfinite(circle["rms_residual_m"])
            else None
        ),
        "angular_coverage_deg": float(circle["angular_coverage_deg"]),
        "hull_points": int(circle["hull_points"]),
    }
    result = {
        "refinement_method": refinement_method,
        "refined_center_xy_base_m": center_xy.astype(float).tolist(),
        "refined_top_center_base_m": [
            float(center_xy[0]), float(center_xy[1]), top_z
        ],
        "refined_center_base_m": [
            float(center_xy[0]), float(center_xy[1]),
            float(support_z + height / 2.0),
        ],
        "support_z_base_m": support_z,
        "height_estimate_mm": height * 1000.0,
        "diameter_estimate_mm": diameter * 1000.0,
        "top_patch_axis_extents_mm": (extents * 1000.0).astype(float).tolist(),
        "top_patch_aspect_ratio": aspect,
        "center_shift_from_coarse_mm": shift * 1000.0,
        "center_estimator_disagreement_mm": disagreement * 1000.0,
        "circle_center_fit": circle_diagnostics,
        "top_component_selection": component_diagnostics,
        "refined_pixel_centroid": pixel.astype(float).tolist(),
        "refined_component_area_px": valid_area,
        "refined_bbox_xywh_px": [
            left, top, right - left + 1, bottom - top + 1
        ],
    }
    return (result, view)


def build_sample_motion(order: int, coarse: dict, refined: dict, desired_view_pose: list[float], place: list[float], place_diagnostics: dict) -> dict:
    center = np.asarray(refined['refined_center_base_m'], dtype=np.float64)
    if not np.all(np.isfinite(center)):
        raise RuntimeError(f'Sample {order} refined center is not finite')
    diameter = float(refined['diameter_estimate_mm'])
    height = float(refined['height_estimate_mm'])
    edge_margin_m = (WORKBENCH_NOMINAL_TOP_SIDE_M - CYLINDER_NOMINAL_DIAMETER_M) / 2.0
    if edge_margin_m < MIN_CYLINDER_EDGE_MARGIN_M:
        raise RuntimeError(
            f"Sample {order} diameter {diameter:.1f} mm leaves only "
            f"{edge_margin_m*1000.0:.1f} mm per side on the nominal 30 mm top"
        )
    support_z = float(refined['support_z_base_m'])
    place_z = float(place[2])
    grasp_pick_z = support_z + robot.GRASP_HEIGHT_ABOVE_SUPPORT_M - 0.005
    grasp_place_z = place_z + robot.GRASP_HEIGHT_ABOVE_SUPPORT_M - 0.005
    top_z = float(refined['refined_top_center_base_m'][2])
    tray_high_z = max(vision.MIN_SAFE_TCP_Z_M, robot.MIN_TRANSFER_Z_M, top_z + robot.TRANSFER_CLEARANCE_M)
    workbench_high_z = max(
        tray_high_z,
        place_z + height / 1000.0 + 0.040,
        grasp_pick_z + WORKBENCH_VERTICAL_TRANSFER_CLEARANCE_M,
        grasp_place_z + WORKBENCH_VERTICAL_TRANSFER_CLEARANCE_M,
    )
    pick_xy = np.asarray(center[:2], dtype=np.float64)
    workbench_xy = np.asarray(place[:2], dtype=np.float64)
    route_delta_xy = workbench_xy - pick_xy
    route_length = float(np.linalg.norm(route_delta_xy))
    if route_length <= 2.0 * WORKBENCH_TRANSIT_MIN_ENDPOINT_DISTANCE_M:
        raise RuntimeError('Pick and workbench are too close to form a separate safe transit point')
    valid_transit_fractions = [
        float(fraction) for fraction in WORKBENCH_TRANSIT_FRACTIONS
        if route_length * float(fraction)
        >= WORKBENCH_TRANSIT_MIN_ENDPOINT_DISTANCE_M
        and route_length * (1.0 - float(fraction))
        >= WORKBENCH_TRANSIT_MIN_ENDPOINT_DISTANCE_M
    ]
    if not valid_transit_fractions:
        raise RuntimeError('No adaptive transit point has sufficient distance from both endpoints')
    transit_fraction = valid_transit_fractions[0]
    transit_xy = pick_xy + transit_fraction * route_delta_xy
    orientation = [float(value) for value in desired_view_pose[3:6]]
    return {
        "order": order,
        "coarse_candidate_id": int(coarse["candidate_id"]),
        "coarse_center_base_m": coarse["coarse_center_base_m"],
        "refined_center_base_m": center.astype(float).tolist(),
        "refined_top_center_base_m": refined["refined_top_center_base_m"],
        "support_z_base_m": support_z,
        "diameter_estimate_mm": diameter,
        "height_estimate_mm": height,
        "coarse_to_refined_shift_mm": float(
            refined["center_shift_from_coarse_mm"]
        ),
        "center_estimator_disagreement_mm": float(
            refined["center_estimator_disagreement_mm"]
        ),
        "workbench_place_center_base_m": [float(value) for value in place],
        "nominal_workbench_edge_margin_mm": edge_margin_m * 1000.0,
        "workbench_place_source": place_diagnostics.get("source_file"),
        "workbench_dwell_mode": "gripper_closed_sample_held_at_place_pose",
        "workbench_dwell_seconds": float(robot.WAIT_AT_PLACE_S),
        "motion_strategy": (
            "low_lift_to_adaptive_segment_transit_then_high_translate_and_descend"
        ),
        "central_transit_xy_base_m": transit_xy.astype(float).tolist(),
        "transit_fraction_pick_to_workbench": transit_fraction,
        "pick_to_workbench_xy_distance_mm": route_length * 1000.0,
        "poses_base_tcp_m_rad": {
            "high_pick": [
                float(center[0]), float(center[1]), tray_high_z, *orientation
            ],
            "grasp_pick": [
                float(center[0]), float(center[1]), grasp_pick_z, *orientation
            ],
            "transit_low": [
                float(transit_xy[0]), float(transit_xy[1]), tray_high_z,
                *orientation,
            ],
            "transit_high": [
                float(transit_xy[0]), float(transit_xy[1]), workbench_high_z,
                *orientation,
            ],
            "high_place": [
                float(place[0]), float(place[1]), workbench_high_z, *orientation
            ],
            "grasp_place": [
                float(place[0]), float(place[1]), grasp_place_z, *orientation
            ],
        },
    }


def rotate_vertical_pose_yaw(pose: list[float], delta_deg: float) -> list[float]:
    """Rotate a vertical cylindrical grasp about Base Z without changing XYZ."""
    transform = vision.pose6_to_transform(pose)
    angle = math.radians(delta_deg)
    c, s = (math.cos(angle), math.sin(angle))
    rotation_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    transform[:3, :3] = rotation_z @ transform[:3, :3]
    result = vision.transform_to_pose6(transform)
    return [float(value) for value in result]


def preflight_sample_poses(control, sample: dict) -> None:
    """Find a reachable workbench lift height/yaw before picking the sample."""
    if not hasattr(control, 'isPoseWithinSafetyLimits'):
        return
    poses = sample['poses_base_tcp_m_rad']
    preferred_high_z = float(poses['high_place'][2])
    minimum_high_z = float(poses['grasp_place'][2]) + WORKBENCH_MIN_LIFT_CLEARANCE_M
    if preferred_high_z < minimum_high_z:
        preferred_high_z = minimum_high_z
    steps = int(math.ceil((preferred_high_z - minimum_high_z) / WORKBENCH_HIGH_Z_SEARCH_STEP_M))
    candidate_heights = [preferred_high_z - index * WORKBENCH_HIGH_Z_SEARCH_STEP_M for index in range(steps + 1)]
    if not candidate_heights or candidate_heights[-1] > minimum_high_z + 1e-09:
        candidate_heights.append(minimum_high_z)
    candidate_heights[-1] = max(candidate_heights[-1], minimum_high_z)
    original_pose = list(poses['high_pick'])
    pose_names = tuple(poses.keys())
    rejected_first = None
    pick_xy = np.asarray(poses['high_pick'][:2], dtype=np.float64)
    workbench_xy = np.asarray(poses['high_place'][:2], dtype=np.float64)
    route_delta_xy = workbench_xy - pick_xy
    route_length = float(np.linalg.norm(route_delta_xy))
    transit_fractions = [
        float(fraction) for fraction in WORKBENCH_TRANSIT_FRACTIONS
        if route_length * float(fraction)
        >= WORKBENCH_TRANSIT_MIN_ENDPOINT_DISTANCE_M
        and route_length * (1.0 - float(fraction))
        >= WORKBENCH_TRANSIT_MIN_ENDPOINT_DISTANCE_M
    ]
    if not transit_fractions:
        raise RuntimeError('No adaptive transit point is sufficiently separated from both endpoints')
    for transit_fraction in transit_fractions:
        transit_xy = pick_xy + transit_fraction * route_delta_xy
        for yaw_delta_deg in WORKBENCH_FREE_YAW_DELTAS_DEG:
            rotated_reference = rotate_vertical_pose_yaw(original_pose, yaw_delta_deg)
            orientation = rotated_reference[3:6]
            for high_z in candidate_heights:
                trial = {}
                for name in pose_names:
                    trial_pose = list(poses[name])
                    trial_pose[3:6] = orientation
                    if name in {'transit_low', 'transit_high'}:
                        trial_pose[0] = float(transit_xy[0])
                        trial_pose[1] = float(transit_xy[1])
                    if name in {'transit_high', 'high_place'}:
                        trial_pose[2] = float(high_z)
                    trial[name] = trial_pose
                rejected_names = [name for name in pose_names if not control.isPoseWithinSafetyLimits(trial[name])]
                if not rejected_names:
                    poses.update(trial)
                    selected_clearance = float(high_z) - float(poses['grasp_place'][2])
                    sample['central_transit_xy_base_m'] = transit_xy.astype(float).tolist()
                    sample['transit_fraction_pick_to_workbench'] = float(transit_fraction)
                    sample["controller_preflight"] = {
                        "all_destinations_accepted": True,
                        "selected_workbench_high_z_m": float(high_z),
                        "selected_lift_clearance_mm": selected_clearance * 1000.0,
                        "selected_free_yaw_delta_deg": float(yaw_delta_deg),
                        "selected_transit_fraction_pick_to_workbench": (
                            float(transit_fraction)
                        ),
                    }
                    print(
                        f"SAMPLE_{sample['order']} PRECHECK: all poses accepted; "
                        f"workbench lift {selected_clearance*1000.0:.1f} mm; "
                        f"transit {transit_fraction:.2f} pick->workbench; "
                        f"free yaw {yaw_delta_deg:+.0f} deg.",
                        flush=True,
                    )
                    return
                if rejected_first is None:
                    rejected_first = ', '.join(rejected_names)
    raise RuntimeError(
        "No controller-approved workbench pose set was found before picking; "
        f"first rejected pose was {rejected_first}. Tried lift clearances from "
        f"{(preferred_high_z-float(poses['grasp_place'][2]))*1000.0:.1f} to "
        f"{WORKBENCH_MIN_LIFT_CLEARANCE_M*1000.0:.1f} mm, "
        f"{len(transit_fractions)} adaptive transit points and four vertical yaws"
    )


def perform_pick_place_return(control_ref: list, receiver: PersistentRTDEReceive, sample: dict, run: dict, recovery_state: dict, cycle_index: int=1):
    """Hold the sample at the workbench, return it, then end open and high."""
    control = control_ref[0]
    poses = sample['poses_base_tcp_m_rad']
    tag = f"CYCLE_{cycle_index}_SAMPLE_{sample['order']}"

    def stage(name: str) -> None:
        run['stage'] = f'{tag}:{name}'
        write_json(RUN_JSON, run)
        update_recovery_checkpoint(
            recovery_state,
            active_cycle=cycle_index,
            active_sample=int(sample["order"]),
            entered_stage=run["stage"],
        )

    def completed(name: str) -> None:
        update_recovery_checkpoint(recovery_state, completed_stage=f'{tag}:{name}')
    stage('preflight_all_destinations')
    preflight_sample_poses(control, sample)
    stage('high_pick')
    robot.move_l(control, poses["high_pick"], robot.TRANSFER_SPEED_M_S,
                robot.TRANSFER_ACCEL_M_S2, f"{tag}_HIGH_PICK")
    completed('high_pick')
    stage('grasp_pick')
    robot.move_l(control, poses["grasp_pick"], robot.VERTICAL_SPEED_M_S,
                robot.VERTICAL_ACCEL_M_S2, f"{tag}_GRASP_PICK")
    completed('grasp_pick')
    stage('close_pick')
    update_recovery_checkpoint(
        recovery_state,
        sample_held=None,
        gripper_expected="UNKNOWN_DURING_CLOSE",
        gripper_transition_in_progress=True,
        automatic_recovery_allowed=False,
    )

    def close_confirmed() -> None:
        update_recovery_checkpoint(
            recovery_state,
            sample_held=True,
            gripper_expected="CLOSED",
            gripper_transition_in_progress=False,
            automatic_recovery_allowed=False,
            completed_stage=f"{tag}:close_pick",
        )
    control = gripper_action_windows(control, receiver, 'CLOSE', on_command_confirmed=close_confirmed)
    control_ref[0] = control
    stage('lift_pick')
    robot.move_l(control, poses["high_pick"], robot.VERTICAL_SPEED_M_S,
                robot.VERTICAL_ACCEL_M_S2, f"{tag}_LIFT_PICK")
    completed('lift_pick')
    stage('central_transit_low_outbound')
    robot.move_l(control, poses["transit_low"], robot.TRANSFER_SPEED_M_S,
                robot.TRANSFER_ACCEL_M_S2, f"{tag}_CENTRAL_TRANSIT_LOW")
    completed('central_transit_low_outbound')
    stage('central_transit_high_outbound')
    robot.move_l(control, poses["transit_high"], robot.VERTICAL_SPEED_M_S,
                robot.VERTICAL_ACCEL_M_S2, f"{tag}_CENTRAL_TRANSIT_HIGH")
    completed('central_transit_high_outbound')
    stage('high_place')
    robot.move_l(control, poses["high_place"], robot.TRANSFER_SPEED_M_S,
                robot.TRANSFER_ACCEL_M_S2, f"{tag}_HIGH_PLACE")
    completed('high_place')
    stage('place')
    robot.move_l(control, poses["grasp_place"], robot.VERTICAL_SPEED_M_S,
                robot.VERTICAL_ACCEL_M_S2, f"{tag}_PLACE")
    completed('place')
    stage('hold_closed_at_place')
    print(
        f"{tag}: RG2 remains CLOSED; hold sample at workbench centre for "
        f"{robot.WAIT_AT_PLACE_S:.0f} seconds",
        flush=True,
    )
    time.sleep(robot.WAIT_AT_PLACE_S)
    completed('hold_closed_at_place')
    stage('lift_held_sample_after_wait')
    robot.move_l(control, poses["high_place"], robot.VERTICAL_SPEED_M_S,
                robot.VERTICAL_ACCEL_M_S2, f"{tag}_LIFT_HELD_AFTER_WAIT")
    completed('lift_held_sample_after_wait')
    stage('central_transit_high_return')
    robot.move_l(control, poses["transit_high"], robot.TRANSFER_SPEED_M_S,
                robot.TRANSFER_ACCEL_M_S2, f"{tag}_CENTRAL_TRANSIT_HIGH_RETURN")
    completed('central_transit_high_return')
    stage('central_transit_low_return')
    robot.move_l(control, poses["transit_low"], robot.VERTICAL_SPEED_M_S,
                robot.VERTICAL_ACCEL_M_S2, f"{tag}_CENTRAL_TRANSIT_LOW_RETURN")
    completed('central_transit_low_return')
    stage('high_original')
    robot.move_l(control, poses["high_pick"], robot.TRANSFER_SPEED_M_S,
                robot.TRANSFER_ACCEL_M_S2, f"{tag}_HIGH_ORIGINAL")
    completed('high_original')
    stage('return_original')
    robot.move_l(control, poses["grasp_pick"], robot.VERTICAL_SPEED_M_S,
                robot.VERTICAL_ACCEL_M_S2, f"{tag}_RETURN_ORIGINAL")
    completed('return_original')
    stage('release_original')
    update_recovery_checkpoint(
        recovery_state,
        sample_held=None,
        gripper_expected="UNKNOWN_DURING_OPEN",
        gripper_transition_in_progress=True,
        automatic_recovery_allowed=False,
    )

    def open_confirmed() -> None:
        update_recovery_checkpoint(
            recovery_state,
            sample_held=False,
            gripper_expected="OPEN",
            gripper_transition_in_progress=False,
            automatic_recovery_allowed=True,
            completed_stage=f"{tag}:release_original",
        )
    control = gripper_action_windows(control, receiver, 'OPEN', on_command_confirmed=open_confirmed)
    control_ref[0] = control
    stage('lift_original')
    robot.move_l(control, poses["high_pick"], robot.VERTICAL_SPEED_M_S,
                robot.VERTICAL_ACCEL_M_S2, f"{tag}_LIFT_ORIGINAL")
    completed('lift_original')
    return list(poses['high_pick'])


def notify_loop_reporter(loop_reporter, event: str, **payload) -> None:
    """Send timing events without allowing a display failure to move the robot."""
    if loop_reporter is None:
        return
    try:
        loop_reporter(event, dict(payload))
    except Exception as error:
        print(f'WARNING: loop timing display failed during {event}: {error}', file=sys.stderr, flush=True)


def main(total_cycles: int | None=None, loop_reporter=None, cycle_number_offset: int=0) -> int:
    """Run forever by default; a positive total_cycles keeps finite test mode."""
    cycle_number_offset = int(cycle_number_offset)
    if cycle_number_offset < 0:
        raise ValueError('cycle_number_offset cannot be negative')
    if total_cycles is not None:
        if isinstance(total_cycles, bool):
            raise ValueError('total_cycles must be None or a positive integer')
        total_cycles = int(total_cycles)
        if total_cycles < 1:
            raise ValueError('total_cycles must be at least 1')
        if cycle_number_offset >= total_cycles:
            raise ValueError('cycle_number_offset must be smaller than total_cycles')
    total_cycles_label = 'infinite' if total_cycles is None else str(total_cycles)
    first_cycle_number = cycle_number_offset + 1
    program_started_monotonic = time.monotonic()
    active_cycle = first_cycle_number
    cycle_started_monotonic = program_started_monotonic
    print(VERSION, flush=True)
    print(f'One continuous run: {total_cycles_label} complete sample-loop(s).', flush=True)
    print(
        "At the workbench the RG2 stays closed: hold the sample at the "
        "placement pose, wait, then return it without releasing/regrasping.",
        flush=True,
    )
    print(
        "The first pass in this worker establishes the fixed tray overview TCP. "
        "Every cycle performs "
        "a new full-tray RGB-D discovery, then a fresh centred refinement for "
        "every sample before grasping.",
        flush=True,
    )
    notify_loop_reporter(loop_reporter, 'run_start', total_cycles=total_cycles, elapsed_s=0.0)
    notify_loop_reporter(
        loop_reporter,
        "cycle_start",
        cycle=first_cycle_number,
        total_cycles=total_cycles,
        vision_enabled=True,
        elapsed_s=0.0,
    )
    run = {
        "version": VERSION,
        "started_at_utc": utc_now(),
        "cycle_mode": "infinite" if total_cycles is None else "finite",
        "requested_total_cycles": total_cycles,
        "completed_cycles": cycle_number_offset,
        "active_cycle": first_cycle_number,
        "status": "starting",
        "stage": "read_home_pose",
        "samples": [],
        "cycles": [],
    }
    control = None
    receiver = None
    camera = None
    recovery_state = None
    try:
        receiver = PersistentRTDEReceive(
            robot.ROBOT_IP,
            frequency_hz=125.0,
            connect_attempts=RTDE_RECONNECT_ATTEMPTS,
            freshness_timeout_s=0.60,
        )
        camera = PersistentRGBDCamera()
        camera.open()
        run['stage'] = 'load_workbench_center'
        place, place_diagnostics = load_calibrated_workbench()
        place_mm = (np.asarray(place, dtype=np.float64) * 1000.0).round(3).tolist()
        print(
            f"Fixed workbench top centre Base {place_mm} mm "
            f"(read directly from {WORKBENCH_JSON.name}).",
            flush=True,
        )
        print(
            "The script will not re-detect this centre. Keep the workbench fixed "
            "and its top empty at the start of the run.",
            flush=True,
        )
        run['stage'] = 'read_home_pose'
        print('[0/5] Read the stationary pre-run pose...', flush=True)
        home_robot = read_stationary_robot_snapshot(receiver)
        run['home_robot'] = vision.json_ready_snapshot(home_robot)
        recovery_state = {
            "schema": RECOVERY_SCHEMA,
            "version": VERSION,
            "active_cycle": first_cycle_number,
            "active_sample": None,
            "completed_cycles": cycle_number_offset,
            "entered_stage": "read_home_pose",
            "completed_stage": "read_home_pose",
            "sample_held": False,
            "gripper_expected": "OPEN",
            "gripper_transition_in_progress": False,
            "automatic_recovery_allowed": True,
            "home_q_rad": [
                float(value) for value in home_robot["actual_q_rad"]
            ],
            "home_tcp_pose_base_m_rad": [
                float(value) for value in initial_onrobot_tcp_pose(home_robot)
            ],
            "cycle_number_offset": cycle_number_offset,
            "worker_started_at_utc": run["started_at_utc"],
        }
        update_recovery_checkpoint(recovery_state)
        run['stage'] = 'oblique_tray_capture'
        update_recovery_checkpoint(recovery_state, entered_stage=run['stage'])
        print('[1/5] Oblique RGB-D: locate the tray only (no sample inference)...', flush=True)
        oblique_color, oblique_depth, oblique_capture, oblique_robot = camera.capture('oblique_tray_only', receiver)
        update_recovery_checkpoint(recovery_state, completed_stage=run['stage'])
        home_joint_change = float(
            np.max(
                np.abs(
                    np.asarray(oblique_robot["actual_q_rad"], dtype=np.float64)
                    - np.asarray(home_robot["actual_q_rad"], dtype=np.float64)
                )
            )
        )
        if home_joint_change > 0.01:
            raise RuntimeError('Robot joints changed before tray localization; restart from a stationary pose')
        tray, oblique_planes = locate_tray_from_oblique(oblique_color, oblique_depth, oblique_robot)
        run['located_tray'] = tray
        run['oblique_scene_planes'] = oblique_planes
        run['oblique_capture'] = oblique_capture
        tray_xy_mm = (np.asarray(tray['center_base_xy_m'], dtype=np.float64) * 1000.0).round(1).tolist()
        print(
            f"Tray center Base XY {tray_xy_mm} mm; "
            f"size {tray['long_side_m']*1000:.1f} x "
            f"{tray['short_side_m']*1000:.1f} mm.",
            flush=True,
        )
        run['stage'] = 'verticalize_camera'
        update_recovery_checkpoint(
            recovery_state,
            active_cycle=first_cycle_number,
            entered_stage=f"cycle_{first_cycle_number}:verticalize_camera",
        )
        verticalize_camera(home_robot, tray, run)
        update_recovery_checkpoint(recovery_state, completed_stage=f'cycle_{first_cycle_number}:verticalize_camera')
        run['stage'] = 'wide_capture'
        update_recovery_checkpoint(recovery_state, entered_stage=f'cycle_{first_cycle_number}:wide_capture')
        print('[2/5] Vertical RGB-D: detect samples inside the tray...', flush=True)
        wide_color, wide_depth, wide_capture, initial_robot = capture_global_wide_fast('batch_wide', receiver, camera)
        update_recovery_checkpoint(recovery_state, completed_stage=f'cycle_{first_cycle_number}:wide_capture')
        optical_axis = np.asarray(initial_robot['T_base_camera'][:3, 2], dtype=np.float64)
        vertical_dot = float(np.dot(optical_axis, [0.0, 0.0, -1.0]))
        onrobot_transform = initial_robot['T_base_flange'] @ vision.pose6_to_transform(robot.ONROBOT_SINGLE_TCP)
        tcp_vertical_dot = float(np.dot(onrobot_transform[:3, 2], [0.0, 0.0, -1.0]))
        if vertical_dot < MIN_DOWNWARD_OPTICAL_AXIS_DOT:
            raise RuntimeError(f'Camera is not vertical after the leveling move: optical-axis dot={vertical_dot:.6f}')
        if tcp_vertical_dot < MIN_DOWNWARD_TCP_AXIS_DOT:
            raise RuntimeError(f'RG2 TCP is not vertical after the leveling move: TCP-axis dot={tcp_vertical_dot:.6f}')
        run["measured_vertical_alignment"] = {
            "camera_optical_axis_dot_base_minus_z": vertical_dot,
            "onrobot_tcp_axis_dot_base_minus_z": tcp_vertical_dot,
        }
        strict_candidates, planes, _ = vision.coarse_depth_candidates(wide_color, wide_depth, initial_robot)
        candidates, recovered_before_tray_filter, circle_rejected_count = (
            recover_missed_coarse_candidates(
                wide_depth,
                initial_robot,
                strict_candidates,
                planes,
            )
        )
        vertical_tray = confirm_tray_from_vertical_capture(wide_color, wide_depth, initial_robot, tray)
        candidates, outside_tray_rejected_count = filter_candidates_to_tray(candidates, tray, vertical_tray)
        recovered_count = sum(
            item.get("coarse_detection_method") == "relaxed_fragment_recovery"
            for item in candidates
        )
        if len(candidates) < MIN_BATCH_CANDIDATES:
            raise RuntimeError(
                f"Batch overview contains only {len(candidates)} coarse sample(s). "
                "Return the robot to the wide overview pose before running ALL mode"
            )
        if len(candidates) > 20:
            raise RuntimeError(f'Implausible coarse candidate count: {len(candidates)}')
        wide_record = build_wide_record(wide_capture, initial_robot, candidates, planes)
        place_pixel = project_base_point_to_color(place, initial_robot, wide_color.shape)
        write_overview(wide_color, candidates, place_pixel)
        run.update(
            {
                "status": "coarse_detection_complete",
                "coarse_candidate_count": len(candidates),
                "strict_coarse_candidate_count": len(strict_candidates),
                "relaxed_recovered_candidate_count": recovered_count,
                "relaxed_recovered_before_tray_filter": recovered_before_tray_filter,
                "circle_rejected_candidate_count": circle_rejected_count,
                "outside_tray_rejected_candidate_count": outside_tray_rejected_count,
                "vertical_confirmed_tray": vertical_tray,
                "coarse_candidates": candidates,
                "scene_planes": planes,
                "workbench_place_base_m": place,
                "workbench_place_pixel_uv": place_pixel,
                "workbench_place_diagnostics": place_diagnostics,
                "initial_robot": vision.json_ready_snapshot(initial_robot),
            }
        )
        write_json(RUN_JSON, run)
        print(
            f"Detected {len(candidates)} coarse samples "
            f"({len(candidates) - recovered_count} strict + "
            f"{recovered_count} recovered; "
            f"{circle_rejected_count} non-circular rejected).",
            flush=True,
        )
        for item in candidates:
            xyz = (np.asarray(item['coarse_center_base_m']) * 1000.0).round(1)
            print(f"  order {item['candidate_id']}: coarse Base {xyz.tolist()} mm", flush=True)
        current = robot.read_current_robot(wide_record, receiver)
        robot.validate_start(wide_record, current)
        robot.verify_remote_and_stopped()
        print('Motion starts in 5 seconds; keep pendant Stop reachable.', flush=True)
        for remaining in range(5, 0, -1):
            print(f'  {remaining}', flush=True)
            time.sleep(1.0)
        prepare_rg2_command_program()
        run_loaded_rg2_command('OPEN', receiver)
        control = start_rtde_control_windows()
        current_pose = initial_onrobot_tcp_pose(initial_robot)
        first_cycle = {
            "cycle": first_cycle_number,
            "vision_enabled": True,
            "started_at_utc": run["started_at_utc"],
            "status": "running",
            "samples_completed": 0,
        }
        run['cycles'].append(first_cycle)
        for order, coarse in enumerate(candidates, start=1):
            tag = f'CYCLE_{first_cycle_number}_SAMPLE_{order}'
            sample_started_monotonic = time.monotonic()
            notify_loop_reporter(
                loop_reporter,
                "sample_start",
                cycle=first_cycle_number,
                total_cycles=total_cycles,
                sample=order,
                total_samples=len(candidates),
                vision_enabled=True,
                elapsed_s=sample_started_monotonic - program_started_monotonic,
            )
            print(
                f"\n===== CYCLE {first_cycle_number}/{total_cycles_label} | "
                f"SAMPLE {order}/{len(candidates)} | VISION =====",
                flush=True,
            )
            run['stage'] = f'{tag}:move_camera'
            update_recovery_checkpoint(
                recovery_state,
                active_cycle=first_cycle_number,
                active_sample=order,
                entered_stage=run["stage"],
                sample_held=False,
                gripper_expected="OPEN",
                gripper_transition_in_progress=False,
                automatic_recovery_allowed=True,
            )
            desired_camera, desired_view_pose = desired_batch_view_pose(coarse, initial_robot)
            current_pose = move_to_centered_view(control, current_pose, desired_view_pose, tag)
            update_recovery_checkpoint(recovery_state, completed_stage=run['stage'])
            stop_rtde_control_windows(control)
            control = None
            run['stage'] = f'{tag}:centered_capture'
            update_recovery_checkpoint(recovery_state, entered_stage=run['stage'])
            refine_color, refine_depth, refine_capture, refine_robot = (
                capture_centered_fast(
                    f"sample_{order}_centered", receiver, camera
                )
            )
            try:
                refined, refined_image = refine_selected_top(refine_color, refine_depth, refine_robot, coarse, planes)
            except RuntimeError as first_error:
                print(f'{tag}: center quality check failed; one fast recapture at the same pose.', flush=True)
                retry_color, retry_depth, retry_capture, retry_robot = (
                    capture_centered_fast(
                        f"sample_{order}_centered_retry", receiver, camera
                    )
                )
                try:
                    refined, refined_image = refine_selected_top(retry_color, retry_depth, retry_robot, coarse, planes)
                    refined['first_center_quality_error'] = str(first_error)
                    refine_capture = retry_capture
                    refine_robot = retry_robot
                except RuntimeError as retry_error:
                    raise RuntimeError(f'first center estimate: {first_error}; fast recapture: {retry_error}') from retry_error
            sample = build_sample_motion(order, coarse, refined, desired_view_pose, place, place_diagnostics)
            sample['desired_T_base_camera'] = desired_camera.tolist()
            sample['desired_view_pose_base_tcp_m_rad'] = [float(value) for value in desired_view_pose]
            sample['refine_capture'] = refine_capture
            sample['refine_robot'] = vision.json_ready_snapshot(refine_robot)
            sample['status'] = 'refined_not_yet_grasped'
            sample['cycle_history'] = []
            run['samples'].append(sample)
            write_json(RUN_JSON, run)
            update_recovery_checkpoint(recovery_state, completed_stage=run['stage'])
            center_mm = (np.asarray(sample['refined_center_base_m']) * 1000.0).round(2).tolist()
            print(
                f"{tag} refined Base center {center_mm} mm; "
                f"D={sample['diameter_estimate_mm']:.1f}, "
                f"H={sample['height_estimate_mm']:.1f} mm",
                flush=True,
            )
            control = start_rtde_control_windows()
            control_ref = [control]
            try:
                current_pose = perform_pick_place_return(
                    control_ref,
                    receiver,
                    sample,
                    run,
                    recovery_state,
                    cycle_index=first_cycle_number,
                )
            finally:
                control = control_ref[0]
            sample_elapsed = time.monotonic() - sample_started_monotonic
            sample['status'] = f'centred_and_cycle_{first_cycle_number}_completed'
            sample['completed_at_utc'] = utc_now()
            sample["cycle_history"].append(
                {
                    "cycle": first_cycle_number,
                    "status": "completed",
                    "completed_at_utc": sample["completed_at_utc"],
                    "elapsed_s": sample_elapsed,
                    "vision_used": True,
                }
            )
            first_cycle['samples_completed'] = order
            write_json(RUN_JSON, run)
            notify_loop_reporter(
                loop_reporter,
                "sample_end",
                cycle=first_cycle_number,
                total_cycles=total_cycles,
                sample=order,
                total_samples=len(candidates),
                vision_enabled=True,
                sample_elapsed_s=sample_elapsed,
                elapsed_s=time.monotonic() - program_started_monotonic,
            )
            print(f'{tag} complete; moving directly to the next sample.', flush=True)
        run['stage'] = f'cycle_{first_cycle_number}:return_initial_joints'
        update_recovery_checkpoint(
            recovery_state,
            active_sample=None,
            entered_stage=run["stage"],
            sample_held=False,
            gripper_expected="OPEN",
            gripper_transition_in_progress=False,
            automatic_recovery_allowed=True,
        )
        robot.move_j(control, home_robot['actual_q_rad'], f'CYCLE_{first_cycle_number}_FINAL_HOME_Q')
        update_recovery_checkpoint(recovery_state, completed_stage=run['stage'])
        first_cycle_elapsed = time.monotonic() - cycle_started_monotonic
        first_cycle.update({'status': 'completed', 'completed_at_utc': utc_now(), 'elapsed_s': first_cycle_elapsed})
        run['completed_cycles'] = first_cycle_number
        run['active_cycle'] = None
        write_json(RUN_JSON, run)
        update_recovery_checkpoint(
            recovery_state,
            completed_cycles=first_cycle_number,
            active_cycle=first_cycle_number + 1,
            active_sample=None,
            entered_stage=f"cycle_{first_cycle_number}:home_complete",
            completed_stage=f"cycle_{first_cycle_number}:home_complete",
        )
        notify_loop_reporter(
            loop_reporter,
            "cycle_end",
            cycle=first_cycle_number,
            total_cycles=total_cycles,
            vision_enabled=True,
            cycle_elapsed_s=first_cycle_elapsed,
            elapsed_s=time.monotonic() - program_started_monotonic,
        )
        print(
            f"CYCLE {first_cycle_number}/{total_cycles_label} COMPLETE: "
            f"{len(candidates)} samples; "
            f"precise centres measured; robot is home.",
            flush=True,
        )
        if total_cycles is None or first_cycle_number < total_cycles:
            print(
                "CAMERA REMAINS OPEN: every later cycle reuses the fixed global "
                "overview TCP, rediscovers the complete sample set, and refines "
                "each detected sample before grasping.",
                flush=True,
            )
        cycle_index = first_cycle_number + 1
        while total_cycles is None or cycle_index <= total_cycles:
            active_cycle = cycle_index
            cycle_started_monotonic = time.monotonic()
            run['active_cycle'] = cycle_index
            run['status'] = 'running_global_discovery_cycle'
            cycle_record = {
                "cycle": cycle_index,
                "vision_enabled": True,
                "global_discovery": True,
                "started_at_utc": utc_now(),
                "status": "running",
                "samples_completed": 0,
            }
            run['cycles'].append(cycle_record)
            write_json(RUN_JSON, run)
            update_recovery_checkpoint(
                recovery_state,
                active_cycle=cycle_index,
                active_sample=None,
                entered_stage=f"cycle_{cycle_index}:start",
                sample_held=False,
                gripper_expected="OPEN",
                gripper_transition_in_progress=False,
                automatic_recovery_allowed=True,
            )
            notify_loop_reporter(
                loop_reporter,
                "cycle_start",
                cycle=cycle_index,
                total_cycles=total_cycles,
                vision_enabled=True,
                elapsed_s=time.monotonic() - program_started_monotonic,
            )
            print(
                f"\n========== CYCLE {cycle_index}/{total_cycles_label} | "
                "GLOBAL DISCOVERY + FRESH REFINEMENT | CAMERA ON ==========",
                flush=True,
            )
            stop_rtde_control_windows(control)
            control = None
            candidates, cycle_planes, cycle_robot, discovery = (
                discover_global_samples_for_cycle(
                    cycle_index,
                    camera,
                    receiver,
                    home_robot,
                    tray,
                    place,
                    run,
                    recovery_state,
                )
            )
            cycle_record['global_discovery_diagnostics'] = discovery
            cycle_record['detected_sample_count'] = len(candidates)
            run['latest_global_discovery'] = discovery
            run['coarse_candidates'] = candidates
            run['scene_planes'] = cycle_planes
            cycle_samples = []
            run['samples'] = cycle_samples
            write_json(RUN_JSON, run)
            control = start_rtde_control_windows()
            current_pose = initial_onrobot_tcp_pose(cycle_robot)
            for order, coarse in enumerate(candidates, start=1):
                tag = f'CYCLE_{cycle_index}_SAMPLE_{order}'
                sample_started_monotonic = time.monotonic()
                notify_loop_reporter(
                    loop_reporter,
                    "sample_start",
                    cycle=cycle_index,
                    total_cycles=total_cycles,
                    sample=order,
                    total_samples=len(candidates),
                    vision_enabled=True,
                    elapsed_s=(
                        sample_started_monotonic - program_started_monotonic
                    ),
                )
                print(
                    f"\n===== CYCLE {cycle_index}/{total_cycles_label} | "
                    f"SAMPLE {order}/{len(candidates)} | FRESH VISION =====",
                    flush=True,
                )
                run['stage'] = f'{tag}:move_camera'
                update_recovery_checkpoint(
                    recovery_state,
                    active_cycle=cycle_index,
                    active_sample=order,
                    entered_stage=run["stage"],
                    sample_held=False,
                    gripper_expected="OPEN",
                    gripper_transition_in_progress=False,
                    automatic_recovery_allowed=True,
                )
                desired_camera, desired_view_pose = desired_batch_view_pose(coarse, cycle_robot)
                current_pose = move_to_centered_view(control, current_pose, desired_view_pose, tag)
                update_recovery_checkpoint(recovery_state, completed_stage=run['stage'])
                stop_rtde_control_windows(control)
                control = None
                run['stage'] = f'{tag}:centered_capture'
                update_recovery_checkpoint(recovery_state, entered_stage=run['stage'])
                refine_color, refine_depth, refine_capture, refine_robot = (
                    capture_centered_fast(
                        f"cycle_{cycle_index}_sample_{order}_centered",
                        receiver,
                        camera,
                    )
                )
                try:
                    refined, refined_image = refine_selected_top(refine_color, refine_depth, refine_robot, coarse, cycle_planes)
                except RuntimeError as first_error:
                    print(f'{tag}: center quality check failed; one fast recapture at the same pose.', flush=True)
                    retry_color, retry_depth, retry_capture, retry_robot = (
                        capture_centered_fast(
                            f"cycle_{cycle_index}_sample_{order}_centered_retry",
                            receiver,
                            camera,
                        )
                    )
                    try:
                        refined, refined_image = refine_selected_top(retry_color, retry_depth, retry_robot, coarse, cycle_planes)
                        refined['first_center_quality_error'] = str(first_error)
                        refine_capture = retry_capture
                        refine_robot = retry_robot
                    except RuntimeError as retry_error:
                        raise RuntimeError(f'first center estimate: {first_error}; fast recapture: {retry_error}') from retry_error
                sample = build_sample_motion(order, coarse, refined, desired_view_pose, place, place_diagnostics)
                sample['desired_T_base_camera'] = desired_camera.tolist()
                sample['desired_view_pose_base_tcp_m_rad'] = [float(value) for value in desired_view_pose]
                sample['refine_capture'] = refine_capture
                sample['refine_robot'] = vision.json_ready_snapshot(refine_robot)
                sample['status'] = 'refined_not_yet_grasped'
                sample['cycle_history'] = []
                cycle_samples.append(sample)
                center_mm = (np.asarray(sample['refined_center_base_m']) * 1000.0).round(2).tolist()
                print(
                    f"{tag}: fresh refined Base centre {center_mm} mm; "
                    f"D={sample['diameter_estimate_mm']:.1f}, "
                    f"H={sample['height_estimate_mm']:.1f} mm.",
                    flush=True,
                )
                write_json(RUN_JSON, run)
                update_recovery_checkpoint(recovery_state, completed_stage=run['stage'])
                control = start_rtde_control_windows()
                control_ref = [control]
                try:
                    current_pose = perform_pick_place_return(
                        control_ref,
                        receiver,
                        sample,
                        run,
                        recovery_state,
                        cycle_index=cycle_index,
                    )
                finally:
                    control = control_ref[0]
                sample_elapsed = time.monotonic() - sample_started_monotonic
                sample["cycle_history"].append(
                    {
                        "cycle": cycle_index,
                        "status": "completed",
                        "completed_at_utc": utc_now(),
                        "elapsed_s": sample_elapsed,
                        "vision_used": True,
                    }
                )
                cycle_record['samples_completed'] = order
                write_json(RUN_JSON, run)
                notify_loop_reporter(
                    loop_reporter,
                    "sample_end",
                    cycle=cycle_index,
                    total_cycles=total_cycles,
                    sample=order,
                    total_samples=len(candidates),
                    vision_enabled=True,
                    sample_elapsed_s=sample_elapsed,
                    elapsed_s=time.monotonic() - program_started_monotonic,
                )
            run['stage'] = f'cycle_{cycle_index}:return_initial_joints'
            update_recovery_checkpoint(
                recovery_state,
                active_sample=None,
                entered_stage=run["stage"],
                sample_held=False,
                gripper_expected="OPEN",
                gripper_transition_in_progress=False,
                automatic_recovery_allowed=True,
            )
            robot.move_j(control, home_robot['actual_q_rad'], f'CYCLE_{cycle_index}_FINAL_HOME_Q')
            update_recovery_checkpoint(recovery_state, completed_stage=run['stage'])
            current_pose = initial_onrobot_tcp_pose(home_robot)
            cycle_elapsed = time.monotonic() - cycle_started_monotonic
            cycle_record.update({'status': 'completed', 'completed_at_utc': utc_now(), 'elapsed_s': cycle_elapsed})
            run['completed_cycles'] = cycle_index
            run['active_cycle'] = None
            write_json(RUN_JSON, run)
            update_recovery_checkpoint(
                recovery_state,
                completed_cycles=cycle_index,
                active_cycle=cycle_index + 1,
                active_sample=None,
                entered_stage=f"cycle_{cycle_index}:home_complete",
                completed_stage=f"cycle_{cycle_index}:home_complete",
            )
            notify_loop_reporter(
                loop_reporter,
                "cycle_end",
                cycle=cycle_index,
                total_cycles=total_cycles,
                vision_enabled=True,
                cycle_elapsed_s=cycle_elapsed,
                elapsed_s=time.monotonic() - program_started_monotonic,
            )
            print(
                f"CYCLE {cycle_index}/{total_cycles_label} COMPLETE: "
                f"{len(candidates)} globally rediscovered and processed; "
                "robot is home.",
                flush=True,
            )
            cycle_index += 1
        run['status'] = 'completed'
        run['stage'] = 'complete'
        run['completed_at_utc'] = utc_now()
        run['total_elapsed_s'] = time.monotonic() - program_started_monotonic
        write_json(RUN_JSON, run)
        update_recovery_checkpoint(
            recovery_state,
            automatic_recovery_allowed=False,
            entered_stage="complete",
            completed_stage="complete",
        )
        print(f'COMPLETED: {total_cycles} cycle(s), {len(candidates)} samples per cycle; robot is home.', flush=True)
        print(f'Result image: {OVERVIEW_IMAGE}', flush=True)
        print(f'Result data: {RUN_JSON}', flush=True)
        return 0
    except KeyboardInterrupt:
        stopped_elapsed = time.monotonic() - program_started_monotonic
        run['status'] = 'stopped_by_user'
        run['active_cycle'] = active_cycle
        run['total_elapsed_s'] = stopped_elapsed
        run['stopped_at_utc'] = utc_now()
        run['stop_reason'] = 'KeyboardInterrupt'
        if run['cycles'] and run['cycles'][-1].get('status') == 'running':
            run["cycles"][-1].update(
                {
                    "status": "stopped_by_user",
                    "stopped_at_utc": utc_now(),
                    "elapsed_s": time.monotonic() - cycle_started_monotonic,
                }
            )
        notify_loop_reporter(
            loop_reporter,
            "run_stopped",
            cycle=active_cycle,
            total_cycles=total_cycles,
            elapsed_s=stopped_elapsed,
            stage=run.get("stage"),
            reason="KeyboardInterrupt",
        )
        print(f"STOPPED BY USER at {run.get('stage')}; no next cycle will start.", file=sys.stderr, flush=True)
        return 130
    except BaseException as error:
        error_elapsed = time.monotonic() - program_started_monotonic
        run['status'] = 'stopped_with_error'
        run['active_cycle'] = active_cycle
        run['total_elapsed_s'] = error_elapsed
        run['error'] = f'{type(error).__name__}: {error}'
        run['traceback'] = traceback.format_exc()
        if run['cycles'] and run['cycles'][-1].get('status') == 'running':
            run["cycles"][-1].update(
                {
                    "status": "failed",
                    "failed_at_utc": utc_now(),
                    "elapsed_s": time.monotonic() - cycle_started_monotonic,
                    "error": run["error"],
                }
            )
        notify_loop_reporter(
            loop_reporter,
            "run_error",
            cycle=active_cycle,
            total_cycles=total_cycles,
            elapsed_s=error_elapsed,
            stage=run.get("stage"),
            error=run["error"],
        )
        print(f"STOPPED at {run.get('stage')}: {error}", file=sys.stderr, flush=True)
        print(
            "Remaining samples were not attempted. Verify the stopped state on the pendant; "
            "no automatic robot-motion recovery was attempted.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        stop_rtde_control_windows(control, strict=False)
        if camera is not None:
            camera.close()
        if receiver is not None:
            receiver.close()
        run['finished_at_utc'] = utc_now()
        write_json(RUN_JSON, run)


if __name__ == '__main__':
    sys.exit(main())
