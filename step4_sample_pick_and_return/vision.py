from __future__ import annotations

from pathlib import Path
import math
import socket
import struct
import time
import warnings

import cv2
import numpy as np

from robot import ROBOT_IP, ONROBOT_SINGLE_TCP


SECONDARY_PORT = 30012
TCP_OFFSET_READ_TIMEOUT_S = 6.0
TCP_OFFSET_READ_ATTEMPTS = 3
TCP_OFFSET_RETRY_DELAY_S = 0.5

# Camera calibration
CAMERA_SERIAL = 'CV2L7610016A'
COLOR_WIDTH = 848
COLOR_HEIGHT = 530
COLOR_FPS = 15
T_FLANGE_CAMERA = np.array(
    [
        [-0.9997923621139053, -0.01727803275277129, 0.010802881231019579, 0.010020656535130776],
        [0.017295984865174023, -0.9998491797343178, 0.0015705706463489122, 0.07488965600334786],
        [0.010774115566534084, 0.0017570910066519174, 0.9999404137572161, 0.058478193092896445],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
COLOR_K = np.array(
    [
        [409.38232421875, 0.0, 421.5352783203125],
        [0.0, 409.3629455566406, 265.9438171386719],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
COLOR_D = np.array(
    [
        -1.2459391355514526,
        0.8174504637718201,
        0.0002063319116132334,
        0.00006101600592955947,
        -0.1829749494791031,
        -1.2284886837005615,
        0.7941957116127014,
        -0.17399488389492035,
    ],
    dtype=np.float64,
)

# Depth workspace
BASE_X_RANGE_M = (-0.95, -0.25)
BASE_Y_RANGE_M = (-0.5, 0.5)
BASE_Z_RANGE_M = (-0.08, 0.32)
DEPTH_RANGE_MM = (150.0, 1500.0)
WARMUP_VALID_FRAME_COUNT = 2
CAPTURE_VALID_FRAME_COUNT = 9
FRAME_WAIT_MS = 1500
TOTAL_CAPTURE_TIMEOUT_S = 20.0


def pose6_to_transform(pose: np.ndarray | list[float]) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(pose)):
        raise ValueError('Robot pose contains non-finite values')
    rotation, _ = cv2.Rodrigues(pose[3:6])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = pose[:3]
    return transform


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def rotation_difference_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def recv_exact(sock: socket.socket, size: int, deadline: float) -> bytes:
    data = bytearray()
    while len(data) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(f'No complete UR secondary state received within {TCP_OFFSET_READ_TIMEOUT_S:.1f} seconds')
        sock.settimeout(min(remaining, 1.0))
        try:
            chunk = sock.recv(size - len(data))
        except socket.timeout:
            continue
        if not chunk:
            raise ConnectionError('UR secondary state socket was closed')
        data.extend(chunk)
    return bytes(data)


def parse_cartesian_tcp_offset(payload: bytes) -> np.ndarray | None:
    result = None
    position = 0
    while position < len(payload):
        if len(payload) - position < 5:
            raise ValueError('Incomplete UR state subpackage header')
        size, kind = struct.unpack_from('!IB', payload, position)
        if size < 5 or position + size > len(payload):
            raise ValueError('Invalid UR state subpackage length')
        if kind == 4:
            if size < 101:
                raise ValueError('UR CartesianInfo does not include a TCP offset')
            values = struct.unpack_from('!12d', payload, position + 5)
            result = np.asarray(values[6:12], dtype=np.float64)
            if not np.all(np.isfinite(result)):
                raise ValueError('UR returned a non-finite active TCP offset')
        position += size
    return result


def _read_active_tcp_offset_once(sample_count: int) -> tuple[np.ndarray, float]:
    """Perform one read-only UR secondary-state acquisition."""
    samples: list[np.ndarray] = []
    deadline = time.monotonic() + TCP_OFFSET_READ_TIMEOUT_S
    with socket.create_connection((ROBOT_IP, SECONDARY_PORT), timeout=TCP_OFFSET_READ_TIMEOUT_S) as sock:
        while len(samples) < sample_count:
            size, message_type = struct.unpack('!IB', recv_exact(sock, 5, deadline))
            if not 5 <= size <= 2000000:
                raise ValueError('Invalid UR state message length')
            payload = recv_exact(sock, size - 5, deadline)
            if message_type != 16:
                continue
            offset = parse_cartesian_tcp_offset(payload)
            if offset is not None:
                samples.append(offset)
    array = np.stack(samples, axis=0)
    median = np.median(array, axis=0)
    maximum_spread = float(np.max(np.ptp(array, axis=0)))
    if maximum_spread > 1e-07:
        raise RuntimeError(f'Active TCP offset changed while reading (spread {maximum_spread:.3g})')
    return (median, maximum_spread)


def read_active_tcp_offset(sample_count: int=5) -> tuple[np.ndarray, float]:
    """Read active TCP with bounded retries; this never commands the robot."""
    last_error: Exception | None = None
    for attempt in range(1, TCP_OFFSET_READ_ATTEMPTS + 1):
        try:
            return _read_active_tcp_offset_once(sample_count)
        except (TimeoutError, socket.timeout, ConnectionError, OSError) as error:
            last_error = error
            if attempt >= TCP_OFFSET_READ_ATTEMPTS:
                break
            print(
                "WARNING: UR secondary-state TCP offset read failed "
                f"on attempt {attempt}/{TCP_OFFSET_READ_ATTEMPTS}: {error}; "
                f"retrying in {TCP_OFFSET_RETRY_DELAY_S:.1f} s.",
                flush=True,
            )
            time.sleep(TCP_OFFSET_RETRY_DELAY_S)
    raise TimeoutError(
        "UR secondary-state TCP offset read failed after "
        f"{TCP_OFFSET_READ_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def robot_snapshot(receiver, tcp_offset: np.ndarray) -> dict:
    pose = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
    speed = np.asarray(receiver.getActualTCPSpeed(), dtype=np.float64)
    joints = np.asarray(receiver.getActualQ(), dtype=np.float64)
    tcp_offset = np.asarray(tcp_offset, dtype=np.float64).reshape(6)
    payload = float(receiver.getPayload()) if hasattr(receiver, 'getPayload') else None
    payload_cog = (
        np.asarray(receiver.getPayloadCog(), dtype=np.float64)
        if hasattr(receiver, "getPayloadCog")
        else None
    )
    if pose.shape != (6,) or speed.shape != (6,) or joints.shape != (6,) or (tcp_offset.shape != (6,)):
        raise RuntimeError('UR RTDE returned an unexpected vector length')
    if not all((np.all(np.isfinite(v)) for v in (pose, speed, joints, tcp_offset))):
        raise RuntimeError('UR RTDE returned non-finite data')
    t_base_active_tcp = pose6_to_transform(pose)
    t_flange_active_tcp = pose6_to_transform(tcp_offset)
    t_base_flange = t_base_active_tcp @ inverse_transform(t_flange_active_tcp)
    t_base_camera = t_base_flange @ T_FLANGE_CAMERA
    return {
        "actual_tcp_pose_m_rad": pose.tolist(),
        "actual_tcp_speed_m_rad_s": speed.tolist(),
        "actual_q_rad": joints.tolist(),
        "active_tcp_offset_flange_to_tcp_m_rad": tcp_offset.tolist(),
        "payload_kg": payload,
        "payload_cog_m": None if payload_cog is None else payload_cog.tolist(),
        "T_base_active_tcp": t_base_active_tcp,
        "T_base_flange": t_base_flange,
        "T_base_camera": t_base_camera,
    }


def enum_name(value) -> str:
    return str(value).split('.')[-1].upper()


def color_frame_to_bgr(frame, ob_format) -> np.ndarray:
    width = int(frame.get_width())
    height = int(frame.get_height())
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    fmt = frame.get_format()
    name = enum_name(fmt)
    if fmt == ob_format.YUYV or 'YUYV' in name or 'YUY2' in name:
        expected = height * width * 2
        if data.size != expected:
            raise RuntimeError(f'Unexpected YUYV byte count: {data.size}, expected {expected}')
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
    if hasattr(ob_format, 'RGB') and (fmt == ob_format.RGB or name.endswith('RGB')):
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if hasattr(ob_format, 'BGR') and (fmt == ob_format.BGR or name.endswith('BGR')):
        return data.reshape(height, width, 3).copy()
    if 'MJPG' in name or 'MJPEG' in name:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Failed to decode MJPEG color frame')
        return image
    raise RuntimeError(f'Unsupported color format after D2C alignment: {fmt}')


def aligned_frameset(align_filter, frames):
    aligned = align_filter.process(frames)
    if aligned is None:
        return None
    return aligned.as_frame_set()


def extract_aligned_rgbd(frames, ob_format) -> tuple[np.ndarray, np.ndarray, float]:
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if color_frame is None or depth_frame is None:
        raise RuntimeError('Aligned FrameSet is missing color or depth')
    bgr = color_frame_to_bgr(color_frame, ob_format)
    dw, dh = (int(depth_frame.get_width()), int(depth_frame.get_height()))
    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    if raw.size != dw * dh:
        raise RuntimeError('Unexpected depth buffer size')
    scale = float(depth_frame.get_depth_scale())
    depth_mm = raw.reshape(dh, dw).astype(np.float32) * scale
    if depth_mm.shape != bgr.shape[:2]:
        raise RuntimeError(
            f"D2C output sizes differ: color={bgr.shape[1]}x{bgr.shape[0]}, "
            f"depth={dw}x{dh}. Send this error; do not resize either image."
        )
    if bgr.shape[:2] != (COLOR_HEIGHT, COLOR_WIDTH):
        raise RuntimeError(
            f"Color geometry changed to {bgr.shape[1]}x{bgr.shape[0]}; "
            f"hand-eye expects {COLOR_WIDTH}x{COLOR_HEIGHT}."
        )
    return (bgr, depth_mm, scale)


def capture_rgbd(pipeline, align_filter, ob_format) -> tuple[np.ndarray, np.ndarray, dict]:
    valid_seen = 0
    started = time.monotonic()
    while valid_seen < WARMUP_VALID_FRAME_COUNT:
        if time.monotonic() - started > TOTAL_CAPTURE_TIMEOUT_S:
            raise TimeoutError('Timed out while warming up synchronized RGB-D frames')
        try:
            frames = pipeline.wait_for_frames(FRAME_WAIT_MS)
        except RuntimeError as error:
            if 'timed out' in str(error).lower():
                continue
            raise
        if frames is None:
            continue
        aligned = aligned_frameset(align_filter, frames)
        if aligned is None:
            continue
        try:
            extract_aligned_rgbd(aligned, ob_format)
        except RuntimeError:
            continue
        valid_seen += 1
    colors: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    scales: list[float] = []
    started = time.monotonic()
    while len(depths) < CAPTURE_VALID_FRAME_COUNT:
        if time.monotonic() - started > TOTAL_CAPTURE_TIMEOUT_S:
            raise TimeoutError(f'Only received {len(depths)}/{CAPTURE_VALID_FRAME_COUNT} valid aligned frames')
        try:
            frames = pipeline.wait_for_frames(FRAME_WAIT_MS)
        except RuntimeError as error:
            if 'timed out' in str(error).lower():
                continue
            raise
        if frames is None:
            continue
        aligned = aligned_frameset(align_filter, frames)
        if aligned is None:
            continue
        bgr, depth_mm, scale = extract_aligned_rgbd(aligned, ob_format)
        colors.append(bgr.copy())
        depths.append(depth_mm.copy())
        scales.append(scale)
    depth_stack = np.stack(depths, axis=0)
    valid_count = np.count_nonzero(depth_stack > 0.0, axis=0)
    with np.errstate(invalid='ignore'):
        depth_stack[depth_stack <= 0.0] = np.nan
        median_depth = np.nanmedian(depth_stack, axis=0).astype(np.float32)
    median_depth[valid_count < max(3, CAPTURE_VALID_FRAME_COUNT // 3)] = 0.0
    median_depth[~np.isfinite(median_depth)] = 0.0
    middle_color = colors[len(colors) // 2]
    info = {
        "warmup_valid_frames": WARMUP_VALID_FRAME_COUNT,
        "captured_valid_frames": CAPTURE_VALID_FRAME_COUNT,
        "depth_scale_median": float(np.median(scales)),
        "valid_depth_fraction": float(np.mean(median_depth > 0.0)),
        "median_valid_observations_per_pixel": float(np.median(valid_count[valid_count > 0])),
    }
    return (middle_color, median_depth, info)


def depth_to_base_maps(depth_mm: np.ndarray, t_base_camera: np.ndarray) -> tuple[np.ndarray, ...]:
    height, width = depth_mm.shape
    vv, uu = np.indices((height, width), dtype=np.float32)
    distorted_pixels = np.stack((uu, vv), axis=-1).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(distorted_pixels, COLOR_K, COLOR_D).reshape(height, width, 2)
    z_camera = depth_mm.astype(np.float64) / 1000.0
    x_camera = normalized[:, :, 0].astype(np.float64) * z_camera
    y_camera = normalized[:, :, 1].astype(np.float64) * z_camera
    points_camera = np.stack((x_camera, y_camera, z_camera), axis=-1)
    points_base = points_camera @ t_base_camera[:3, :3].T + t_base_camera[:3, 3]
    invalid = (depth_mm < DEPTH_RANGE_MM[0]) | (depth_mm > DEPTH_RANGE_MM[1])
    points_base[invalid] = np.nan
    return (
        points_base[:, :, 0],
        points_base[:, :, 1],
        points_base[:, :, 2],
        normalized[:, :, 0],
        normalized[:, :, 1],
    )


def separated_histogram_peaks(z_values: np.ndarray) -> list[dict]:
    bins = np.arange(BASE_Z_RANGE_M[0], BASE_Z_RANGE_M[1] + 0.002, 0.002)
    histogram, edges = np.histogram(z_values, bins=bins)
    smooth = np.convolve(histogram.astype(np.float64), np.ones(5) / 5.0, mode='same')
    order = np.argsort(smooth)[::-1]
    peaks: list[dict] = []
    for index in order:
        center = float((edges[index] + edges[index + 1]) / 2.0)
        if smooth[index] < 30:
            break
        if any((abs(center - item['z_m']) < 0.012 for item in peaks)):
            continue
        near_count = int(np.count_nonzero(np.abs(z_values - center) <= 0.004))
        peaks.append({'z_m': center, 'near_pixel_count': near_count})
        if len(peaks) >= 6:
            break
    return sorted(peaks, key=lambda item: item['z_m'])


def json_ready_snapshot(snapshot: dict) -> dict:
    result = {}
    for key, value in snapshot.items():
        result[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return result


def save_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f'PNG encoding failed: {path.name}')
    encoded.tofile(str(path))

# Observation pose
CAMERA_CLEARANCE_ABOVE_TOP_M = 0.28
MIN_SAFE_TCP_Z_M = 0.165
MAX_VIEW_TRANSLATION_M = 0.35
VIEW_SPEED_M_S = 0.1
VIEW_ACCEL_M_S2 = 0.25

# Coarse sample detection
COARSE_HEIGHT_ABOVE_TRAY_MIN_M = 0.024
COARSE_HEIGHT_ABOVE_TRAY_MAX_M = 0.07
COARSE_MIN_AREA_PX = 180
COARSE_MAX_AREA_PX = 7000
COARSE_MIN_XY_EXTENT_M = 0.01
COARSE_MAX_XY_EXTENT_M = 0.08


def transform_to_pose6(transform: np.ndarray) -> list[float]:
    rotation_vector, _ = cv2.Rodrigues(transform[:3, :3])
    return [*transform[:3, 3].astype(float).tolist(), *rotation_vector.reshape(3).astype(float).tolist()]


def open_camera_pipeline():
    from pyorbbecsdk import AlignFilter, Config, OBFormat, OBFrameAggregateOutputMode, OBSensorType, OBStreamType, Pipeline
    pipeline = Pipeline()
    serial = pipeline.get_device().get_device_info().get_serial_number()
    if serial != CAMERA_SERIAL:
        raise RuntimeError(f'Camera serial is {serial}; expected {CAMERA_SERIAL}')
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_video_stream_profile(COLOR_WIDTH, COLOR_HEIGHT, OBFormat.YUYV, COLOR_FPS)
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_video_stream_profile(COLOR_WIDTH, COLOR_HEIGHT, OBFormat.Y16, COLOR_FPS)
    config = Config()
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    pipeline.enable_frame_sync()
    pipeline.start(config)
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    return (pipeline, align_filter, OBFormat)


def connect_receiver_with_retry():
    from rtde_receive import RTDEReceiveInterface
    last_error = None
    for attempt in range(1, 4):
        try:
            return RTDEReceiveInterface(ROBOT_IP)
        except Exception as error:
            last_error = error
            time.sleep(0.5 * attempt)
    raise RuntimeError(f'RTDE receive failed after 3 attempts: {last_error}')


def _robot_snapshot_from_receiver(receiver, tcp_offset: np.ndarray, *, reconnect_on_stale: bool) -> dict:
    """Support both the shared manager and the original standalone receiver."""
    if hasattr(receiver, 'snapshot'):
        return receiver.snapshot(robot_snapshot, tcp_offset, reconnect_on_stale=reconnect_on_stale)
    return robot_snapshot(receiver, tcp_offset)


def capture_at_current_pose(label: str, receiver=None) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Capture aligned RGB-D and the matching stationary robot transform."""
    owns_receiver = receiver is None
    pipeline = None
    try:
        if receiver is None:
            receiver = connect_receiver_with_retry()
        tcp_start, spread_start = read_active_tcp_offset()
        start_robot = _robot_snapshot_from_receiver(receiver, tcp_start, reconnect_on_stale=not owns_receiver)
        speed = np.asarray(start_robot['actual_tcp_speed_m_rad_s'], dtype=np.float64)
        if np.linalg.norm(speed[:3]) > 0.002 or np.linalg.norm(speed[3:]) > 0.02:
            raise RuntimeError(f'Robot is moving before {label} capture')
        pipeline, align_filter, ob_format = open_camera_pipeline()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
            color, depth_mm, capture_info = capture_rgbd(pipeline, align_filter, ob_format)
        tcp_end, spread_end = read_active_tcp_offset()
        end_robot = _robot_snapshot_from_receiver(receiver, tcp_end, reconnect_on_stale=not owns_receiver)
        if np.max(np.abs(tcp_end - tcp_start)) > 1e-07:
            raise RuntimeError(f'Active TCP changed during {label} capture')
        position_change_mm = float(
            np.linalg.norm(
                start_robot["T_base_flange"][:3, 3]
                - end_robot["T_base_flange"][:3, 3]
            ) * 1000.0
        )
        orientation_change_deg = rotation_difference_deg(start_robot['T_base_flange'], end_robot['T_base_flange'])
        if position_change_mm > 0.5 or orientation_change_deg > 0.15:
            raise RuntimeError(
                f"Robot moved during {label} capture: {position_change_mm:.3f} mm, "
                f"{orientation_change_deg:.4f} deg"
            )
        capture_info.update(
            {
                "label": label,
                "robot_position_change_mm": position_change_mm,
                "robot_orientation_change_deg": orientation_change_deg,
                "tcp_start_read_spread": spread_start,
                "tcp_end_read_spread": spread_end,
            }
        )
        return (color, depth_mm, capture_info, start_robot)
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if owns_receiver and receiver is not None:
            try:
                receiver.disconnect()
            except Exception:
                pass


def workspace_mask(x_base, y_base, z_base):
    return (
        np.isfinite(x_base) & np.isfinite(y_base) & np.isfinite(z_base)
        & (x_base >= BASE_X_RANGE_M[0]) & (x_base <= BASE_X_RANGE_M[1])
        & (y_base >= BASE_Y_RANGE_M[0]) & (y_base <= BASE_Y_RANGE_M[1])
        & (z_base >= BASE_Z_RANGE_M[0]) & (z_base <= BASE_Z_RANGE_M[1])
    )


def infer_scene_planes(z_base: np.ndarray, workspace: np.ndarray) -> dict:
    z_values = z_base[workspace]
    if z_values.size < 1000:
        raise RuntimeError('Too few valid Base-frame depth points')
    peaks = separated_histogram_peaks(z_values)
    low = [item for item in peaks if -0.08 <= float(item['z_m']) <= 0.04]
    if len(low) < 2:
        raise RuntimeError(f'Could not separate table/tray planes: {peaks}')
    table = max(low, key=lambda item: int(item['near_pixel_count']))
    table_z = float(table['z_m'])
    upper_candidates = [
        item for item in low
        if table_z + 0.010 <= float(item["z_m"]) <= table_z + 0.045
        and int(item["near_pixel_count"]) >= 2000
    ]
    if not upper_candidates:
        raise RuntimeError(f'Could not identify the tray upper plane: {peaks}')
    tray_upper = max(upper_candidates, key=lambda item: int(item['near_pixel_count']))
    tray_upper_z = float(tray_upper['z_m'])
    seat_candidates = [
        item for item in low
        if tray_upper_z - 0.026 <= float(item["z_m"]) <= tray_upper_z - 0.008
        and float(item["z_m"]) > table_z + 0.004
    ]
    if not seat_candidates:
        support_z = tray_upper_z - 0.016
    else:
        support_z = float(max(seat_candidates, key=lambda item: int(item['near_pixel_count']))['z_m'])
    return {'peaks': peaks, 'table_z_m': table_z, 'tray_upper_z_m': tray_upper_z, 'sample_support_z_m': support_z}


def component_xy_geometry(component, x_base, y_base, z_base) -> dict:
    valid_component = component & np.isfinite(x_base) & np.isfinite(y_base) & np.isfinite(z_base)
    rows, cols = np.nonzero(valid_component)
    if rows.size < 20:
        raise ValueError('Connected component has too few valid 3-D points')
    xs = x_base[valid_component]
    ys = y_base[valid_component]
    zs = z_base[valid_component]
    points = np.column_stack((xs, ys))
    center = np.median(points, axis=0)
    centered = points - center
    values, vectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    projected = centered @ vectors[:, ::-1]
    bounds = np.percentile(projected, [5, 95], axis=0)
    extents = bounds[1] - bounds[0]
    return {
        "rows": rows,
        "cols": cols,
        "valid_component": valid_component,
        "valid_area_px": int(rows.size),
        "pixel_centroid": [float(np.mean(cols)), float(np.mean(rows))],
        "center_xy_m": center,
        "axis_extents_m": extents,
        "top_z_m": float(np.percentile(zs, 90)),
        "z_median_m": float(np.median(zs)),
    }


def coarse_depth_candidates(color: np.ndarray, depth_mm: np.ndarray, robot: dict) -> tuple[list[dict], dict, np.ndarray]:
    x_base, y_base, z_base, _, _ = depth_to_base_maps(depth_mm, robot['T_base_camera'])
    workspace = workspace_mask(x_base, y_base, z_base)
    planes = infer_scene_planes(z_base, workspace)
    tray_z = planes['tray_upper_z_m']
    mask = (
        workspace
        & (z_base >= tray_z + COARSE_HEIGHT_ABOVE_TRAY_MIN_M)
        & (z_base <= tray_z + COARSE_HEIGHT_ABOVE_TRAY_MAX_M)
    ).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not COARSE_MIN_AREA_PX <= area <= COARSE_MAX_AREA_PX:
            continue
        component = labels == label
        try:
            geometry = component_xy_geometry(component, x_base, y_base, z_base)
        except ValueError:
            continue
        valid_area = int(geometry['valid_area_px'])
        if not COARSE_MIN_AREA_PX <= valid_area <= COARSE_MAX_AREA_PX:
            continue
        extents = geometry['axis_extents_m']
        if np.min(extents) < COARSE_MIN_XY_EXTENT_M:
            continue
        if np.max(extents) > COARSE_MAX_XY_EXTENT_M:
            continue
        bbox = [
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        ]
        pixel = geometry['pixel_centroid']
        principal_distance = float(math.hypot(pixel[0] - COLOR_K[0, 2], pixel[1] - COLOR_K[1, 2]))
        candidates.append(
            {
                "coarse_center_base_m": [
                    float(geometry["center_xy_m"][0]),
                    float(geometry["center_xy_m"][1]),
                    geometry["z_median_m"],
                ],
                "coarse_top_z_base_m": geometry["top_z_m"],
                "pixel_centroid": pixel,
                "bbox_xywh_px": bbox,
                "component_area_px": valid_area,
                "morphology_component_area_px": area,
                "top_patch_axis_extents_mm": (extents * 1000.0).tolist(),
                "principal_point_distance_px": principal_distance,
            }
        )
    candidates.sort(key=lambda item: item['principal_point_distance_px'])
    for index, item in enumerate(candidates, start=1):
        item['candidate_id'] = index
    if not candidates:
        raise RuntimeError('Depth saw no valid elevated cylinder region. No motion was sent.')
    return (candidates, planes, mask)


def make_coarse_view(color: np.ndarray, candidates: list[dict]) -> np.ndarray:
    view = color.copy()
    for item in candidates:
        x, y, w, h = item['bbox_xywh_px']
        selected = item['candidate_id'] == 1
        colour = (0, 255, 0) if selected else (0, 180, 255)
        cv2.rectangle(view, (x, y), (x + w, y + h), colour, 2)
        center = tuple((int(round(value)) for value in item['pixel_centroid']))
        cv2.drawMarker(view, center, colour, cv2.MARKER_CROSS, 18, 2)
        cv2.putText(
            view, f"coarse #{item['candidate_id']}",
            (x, max(42, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.54, colour, 2,
        )
    cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        view, "DEPTH COARSE ONLY | green will be centered before refinement",
        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2,
    )
    return view


def desired_centered_camera_pose(coarse: dict, initial_robot: dict) -> tuple[np.ndarray, list[float]]:
    target = np.asarray(coarse['coarse_center_base_m'], dtype=np.float64)
    target[2] = float(coarse['coarse_top_z_base_m'])
    rotation = initial_robot['T_base_camera'][:3, :3].copy()
    optical_axis = rotation[:, 2]
    if optical_axis[2] > -0.9:
        raise RuntimeError('Camera optical axis is not sufficiently downward; refusing automatic view move')
    camera_z = max(float(initial_robot['T_base_camera'][2, 3]), float(target[2]) + CAMERA_CLEARANCE_ABOVE_TOP_M)
    ray_length = (float(target[2]) - camera_z) / float(optical_axis[2])
    camera_center = target - ray_length * optical_axis
    camera_center[2] = camera_z
    t_base_camera = np.eye(4, dtype=np.float64)
    t_base_camera[:3, :3] = rotation
    t_base_camera[:3, 3] = camera_center
    t_base_flange = t_base_camera @ inverse_transform(T_FLANGE_CAMERA)
    t_flange_tcp = pose6_to_transform(ONROBOT_SINGLE_TCP)
    t_base_tcp = t_base_flange @ t_flange_tcp
    tcp_pose = transform_to_pose6(t_base_tcp)
    camera_translation = float(np.linalg.norm(t_base_camera[:3, 3] - initial_robot['T_base_camera'][:3, 3]))
    if camera_translation > MAX_VIEW_TRANSLATION_M:
        raise RuntimeError(
            f"Required camera translation is {camera_translation*1000:.1f} mm; "
            f"limit is {MAX_VIEW_TRANSLATION_M*1000:.0f} mm"
        )
    if not np.all(np.isfinite(tcp_pose)):
        raise RuntimeError('Computed centered-view TCP pose is not finite')
    return (t_base_camera, tcp_pose)
