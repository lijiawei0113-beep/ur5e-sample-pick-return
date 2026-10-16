"""Calibrate the green workbench top with a fresh RGB-D refinement."""
from __future__ import annotations
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


VERSION = 'STEP5_WORKBENCH_RGBD_CALIBRATION_V2_6_GREEN_30MM_HEIGHT_265MM'
ROOT = Path(__file__).resolve().parent
STEP4_DIR = ROOT.parent / 'step4_sample_pick_and_return'
sys.path.insert(0, str(STEP4_DIR))

import robot as motion
import vision

OUTPUT_JSON = ROOT / 'workbench_pose.json'
OUTPUT_IMAGE = ROOT / 'workbench_calibration.png'

# Green top dimensions and height above the supporting table
EXPECTED_TOP_SIDE_M = 0.03
COARSE_TOP_SIDE_RANGE_M = (0.023, 0.04)
REFINED_TOP_SIDE_RANGE_M = (0.024, 0.04)
EXPECTED_HEIGHT_ABOVE_TABLE_M = 0.265
HEIGHT_ABOVE_TABLE_RANGE_M = (0.24, 0.29)
TOP_Z_BASE_RANGE_M = (-0.05, 0.7)
TABLE_Z_BASE_RANGE_M = (-0.15, 0.2)
BASE_X_RANGE_M = (-1.0, -0.15)
BASE_Y_RANGE_M = (-0.6, 0.6)

# Green segmentation
GREEN_HSV_LOWER = np.array([32, 60, 35], dtype=np.uint8)
GREEN_HSV_UPPER = np.array([100, 255, 255], dtype=np.uint8)
GREEN_MIN_AREA_PX = 120
GREEN_MAX_AREA_PX = 30000
GREEN_OPEN_KERNEL_PX = 3
GREEN_CLOSE_KERNEL_PX = 7
GREEN_MIN_SOLIDITY = 0.72
GREEN_MIN_RECTANGULARITY = 0.68
GREEN_MAX_IMAGE_ASPECT = 2.0
GREEN_MIN_POLYGON_VERTICES = 4
GREEN_MAX_POLYGON_VERTICES = 8
IMAGE_BORDER_MARGIN_PX = 6

# Depth and plane fitting
COARSE_MIN_POINTS = 4
REFINED_MIN_POINTS = 6
MAX_COARSE_ASPECT = 1.45
MAX_REFINED_ASPECT = 1.3
MAX_COARSE_PLANE_RMS_M = 0.008
MAX_REFINED_PLANE_RMS_M = 0.006
MAX_REFINED_CENTER_SHIFT_M = 0.035

# Camera view and motion
CAMERA_CLEARANCE_ABOVE_TOP_M = 0.22
SAFE_TCP_CLEARANCE_ABOVE_TOP_M = 0.18
MIN_TCP_CLEARANCE_AT_VIEW_M = 0.03
SAFE_TRANSITION_EXTRA_M = 0.1
MAX_SAFE_TCP_Z_M = 0.75
MAX_CAMERA_TRANSLATION_M = 0.45
MOTION_SPEED_M_S = 0.05
MOTION_ACCEL_M_S2 = 0.15

# Skip repositioning when the current view is already sufficient
STATIONARY_REFINEMENT_MAX_CAMERA_TRANSLATION_M = 0.06
STATIONARY_REFINEMENT_MAX_TOP_TILT_DEG = 20.0
STATIONARY_REFINEMENT_MAX_ASPECT = 1.8
STATIONARY_REFINEMENT_MAX_PLANE_RMS_M = 0.003

# Fallback for sparse depth on the green top
GREEN_DEPTH_CLUSTER_HALF_BAND_M = 0.012
GREEN_DEPTH_CLUSTER_SCAN_WIDTH_M = 0.024
MAX_COLOR_DEPTH_Z_DISAGREEMENT_M = 0.06
Z_HISTOGRAM_STEP_M = 0.001


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def finite_workspace(x_base, y_base, z_base) -> np.ndarray:
    return (
        np.isfinite(x_base)
        & np.isfinite(y_base)
        & np.isfinite(z_base)
        & (x_base >= BASE_X_RANGE_M[0])
        & (x_base <= BASE_X_RANGE_M[1])
        & (y_base >= BASE_Y_RANGE_M[0])
        & (y_base <= BASE_Y_RANGE_M[1])
    )


def green_top_mask(color_bgr: np.ndarray) -> np.ndarray:
    """Return a cleaned 8-bit mask of the matte-green printed top."""
    blurred = cv2.GaussianBlur(color_bgr, (5, 5), 0.0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)
    open_kernel = np.ones((GREEN_OPEN_KERNEL_PX, GREEN_OPEN_KERNEL_PX), dtype=np.uint8)
    close_kernel = np.ones((GREEN_CLOSE_KERNEL_PX, GREEN_CLOSE_KERNEL_PX), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return mask


def contour_touches_border(contour: np.ndarray, image_width: int, image_height: int) -> bool:
    x, y, width, height = cv2.boundingRect(contour)
    margin = IMAGE_BORDER_MARGIN_PX
    return (
        x <= margin
        or y <= margin
        or x + width >= image_width - margin
        or y + height >= image_height - margin
    )


def dominant_height(z_values: np.ndarray, lower: float, upper: float) -> float | None:
    values = np.asarray(z_values, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= lower) & (values <= upper)]
    if values.size < 100:
        return None
    edges = np.arange(lower, upper + Z_HISTOGRAM_STEP_M, Z_HISTOGRAM_STEP_M)
    histogram, edges = np.histogram(values, bins=edges)
    if histogram.size == 0 or int(np.max(histogram)) < 20:
        return None
    index = int(np.argmax(histogram))
    return float((edges[index] + edges[index + 1]) / 2.0)


def pixels_to_horizontal_plane(
    pixels_uv: np.ndarray,
    t_base_camera: np.ndarray,
    top_z_base_m: float,
) -> np.ndarray:
    """Intersect undistorted color-camera rays with a live-estimated top Z."""
    image_points = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(
        image_points,
        np.asarray(vision.COLOR_K, dtype=np.float64),
        np.asarray(vision.COLOR_D, dtype=np.float64),
    ).reshape(-1, 2)
    rays_camera = np.column_stack((normalized[:, 0], normalized[:, 1], np.ones(normalized.shape[0])))
    rotation = np.asarray(t_base_camera[:3, :3], dtype=np.float64)
    origin = np.asarray(t_base_camera[:3, 3], dtype=np.float64)
    rays_base = (rotation @ rays_camera.T).T
    denominator = rays_base[:, 2]
    if np.any(np.abs(denominator) < 1e-06):
        raise RuntimeError('Camera ray is parallel to the detected top plane')
    scale = (float(top_z_base_m) - origin[2]) / denominator
    if np.any(scale <= 0.0) or np.any(scale > 3.0):
        raise RuntimeError('Detected top plane is behind the camera')
    return origin + rays_base * scale[:, None]


def projected_rectangle_geometry(center_uv: np.ndarray, box_uv_float: np.ndarray, t_base_camera: np.ndarray, top_z_base_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    projected = pixels_to_horizontal_plane(
        np.vstack((center_uv.reshape(1, 2), box_uv_float)),
        t_base_camera,
        top_z_base_m,
    )
    center = projected[0]
    corners = projected[1:]
    edges = np.linalg.norm(np.roll(corners[:, :2], -1, axis=0) - corners[:, :2], axis=1)
    ordered_edges = np.sort(edges)
    sides = np.array([np.mean(ordered_edges[:2]), np.mean(ordered_edges[2:])], dtype=np.float64)
    aspect = float('inf') if sides[0] <= 0.0 else float(sides[1] / sides[0])
    return (center, corners, sides, aspect)


def top_z_from_known_color_size(center_uv: np.ndarray, box_uv_float: np.ndarray, t_base_camera: np.ndarray) -> float:
    """Estimate top Z from the known square size when depth is incomplete."""
    camera_z = float(t_base_camera[2, 3])
    probe_z = camera_z - 0.25
    _, _, probe_sides, _ = projected_rectangle_geometry(center_uv, box_uv_float, t_base_camera, probe_z)
    mean_probe_side = float(np.mean(probe_sides))
    if not np.isfinite(mean_probe_side) or mean_probe_side <= 1e-06:
        raise RuntimeError('Color rectangle cannot determine metric scale')
    scale = EXPECTED_TOP_SIDE_M / mean_probe_side
    top_z = camera_z + (probe_z - camera_z) * scale
    if not TOP_Z_BASE_RANGE_M[0] <= top_z <= TOP_Z_BASE_RANGE_M[1]:
        raise RuntimeError('Color-size height is outside the robot workspace')
    return float(top_z)


def highest_dense_z_cluster(z_values: np.ndarray, min_points: int) -> np.ndarray:
    """Select the highest compact Z cluster, rejecting lower table/background."""
    values = np.sort(np.asarray(z_values, dtype=np.float64))
    values = values[np.isfinite(values)]
    if values.size < min_points:
        return np.empty(0, dtype=np.float64)
    best_start = -1
    best_end = -1
    start = 0
    for end in range(values.size):
        while values[end] - values[start] > GREEN_DEPTH_CLUSTER_SCAN_WIDTH_M:
            start += 1
        count = end - start + 1
        if count < min_points:
            continue
        if best_start < 0:
            best_start, best_end = (start, end)
            continue
        best_count = best_end - best_start + 1
        current_center = 0.5 * (values[start] + values[end])
        best_center = 0.5 * (values[best_start] + values[best_end])
        if count > 1.35 * best_count or (count >= 0.7 * best_count and current_center > best_center):
            best_start, best_end = (start, end)
    if best_start < 0:
        return np.empty(0, dtype=np.float64)
    seed = float(np.median(values[best_start:best_end + 1]))
    return values[np.abs(values - seed) <= GREEN_DEPTH_CLUSTER_HALF_BAND_M]


def find_top_candidates(color_bgr: np.ndarray, depth_mm: np.ndarray, robot: dict, refined: bool, expected_center_xy: np.ndarray | None=None, expected_top_z: float | None=None, expected_table_z: float | None=None) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, dict]:
    x_base, y_base, z_base, _, _ = vision.depth_to_base_maps(depth_mm, robot['T_base_camera'])
    if color_bgr.shape[:2] != z_base.shape:
        raise RuntimeError('Aligned color and depth sizes do not match')
    color_mask = green_top_mask(color_bgr)
    workspace = finite_workspace(x_base, y_base, z_base)
    live_table_z = dominant_height(
        z_base[workspace & (color_mask == 0)],
        TABLE_Z_BASE_RANGE_M[0],
        TABLE_Z_BASE_RANGE_M[1],
    )
    table_z = float(expected_table_z) if expected_table_z is not None else live_table_z
    search = workspace & (z_base >= TOP_Z_BASE_RANGE_M[0]) & (z_base <= TOP_Z_BASE_RANGE_M[1])
    if expected_center_xy is not None:
        distance = np.hypot(x_base - float(expected_center_xy[0]), y_base - float(expected_center_xy[1]))
        search &= distance <= 0.06
    if expected_top_z is not None:
        search &= np.abs(z_base - expected_top_z) <= 0.05
    min_points = REFINED_MIN_POINTS if refined else COARSE_MIN_POINTS
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diagnostics = {
        "green_pixels": int(np.count_nonzero(color_mask)),
        "contours": int(len(contours)),
        "area_pass": 0,
        "shape_pass": 0,
        "rectangle_pass": 0,
        "color_scale_z_pass": 0,
        "height_pass": 0,
        "sparse_depth_used": 0,
        "metric_size_pass": 0,
        "table_z_base_m": table_z,
        "live_table_z_base_m": live_table_z,
        "rejections": [],
    }
    candidates: list[dict] = []
    side_range = REFINED_TOP_SIDE_RANGE_M if refined else COARSE_TOP_SIDE_RANGE_M
    max_aspect = MAX_REFINED_ASPECT if refined else MAX_COARSE_ASPECT
    max_plane_rms = MAX_REFINED_PLANE_RMS_M if refined else MAX_COARSE_PLANE_RMS_M
    if table_z is None:
        diagnostics['rejections'].append('table_z_unavailable: cannot apply the 265 mm height constraint')
        return ([], x_base, y_base, z_base, diagnostics)
    for contour in contours:
        green_area = int(round(float(cv2.contourArea(contour))))
        if green_area < GREEN_MIN_AREA_PX or green_area > GREEN_MAX_AREA_PX:
            diagnostics['rejections'].append(f'area:{green_area}px')
            continue
        diagnostics['area_pass'] += 1
        if contour_touches_border(contour, color_mask.shape[1], color_mask.shape[0]):
            diagnostics['rejections'].append('image_border')
            continue
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = green_area / max(hull_area, 1.0)
        if hull_area <= 1.0 or solidity < GREEN_MIN_SOLIDITY:
            diagnostics['rejections'].append(f'solidity:{solidity:.2f}')
            continue
        diagnostics['shape_pass'] += 1
        perimeter = float(cv2.arcLength(hull, True))
        polygon = cv2.approxPolyDP(hull, 0.035 * perimeter, True)
        image_rectangle = cv2.minAreaRect(hull)
        rectangle_width, rectangle_height = image_rectangle[1]
        rectangle_area = float(rectangle_width * rectangle_height)
        rectangularity = green_area / max(rectangle_area, 1.0)
        image_short = min(float(rectangle_width), float(rectangle_height))
        image_long = max(float(rectangle_width), float(rectangle_height))
        image_aspect = float('inf') if image_short <= 0.0 else image_long / image_short
        polygon_vertices = int(len(polygon))
        if (
            not (
            rectangularity >= GREEN_MIN_RECTANGULARITY
            and image_aspect <= GREEN_MAX_IMAGE_ASPECT
            and GREEN_MIN_POLYGON_VERTICES
            <= polygon_vertices
            <= GREEN_MAX_POLYGON_VERTICES
                    )
        ):
            diagnostics["rejections"].append(
                f"rectangle:fill={rectangularity:.2f},"
                f"aspect={image_aspect:.2f},vertices={polygon_vertices}"
            )
            continue
        diagnostics['rectangle_pass'] += 1
        component_u8 = np.zeros_like(color_mask)
        cv2.drawContours(component_u8, [contour], -1, 255, thickness=-1)
        valid_component = (component_u8 > 0) & search
        center_uv = np.asarray(image_rectangle[0], dtype=np.float64)
        box_uv_float = cv2.boxPoints(image_rectangle).astype(np.float64)
        try:
            color_top_z = top_z_from_known_color_size(center_uv, box_uv_float, robot['T_base_camera'])
        except RuntimeError as error:
            diagnostics['rejections'].append(f'color_z:{error}')
            continue
        diagnostics['color_scale_z_pass'] += 1
        near_color_height = valid_component & (np.abs(z_base - color_top_z) <= MAX_COLOR_DEPTH_Z_DISAGREEMENT_M)
        z_cluster = highest_dense_z_cluster(z_base[near_color_height], min_points)
        depth_used = z_cluster.size >= min_points
        if depth_used:
            depth_top_z = float(np.median(z_cluster))
            depth_rms = float(np.sqrt(np.mean((z_cluster - depth_top_z) ** 2)))
            if depth_rms <= max_plane_rms:
                top_z = depth_top_z
                diagnostics['sparse_depth_used'] += 1
            else:
                top_z = color_top_z
                depth_used = False
                depth_rms = 0.0
        else:
            top_z = color_top_z
            depth_rms = 0.0
        measured_height = float(top_z - table_z)
        color_height = float(color_top_z - table_z)
        if (
            not (
            HEIGHT_ABOVE_TABLE_RANGE_M[0]
            <= measured_height
            <= HEIGHT_ABOVE_TABLE_RANGE_M[1]
                    ) and (
            HEIGHT_ABOVE_TABLE_RANGE_M[0]
            <= color_height
            <= HEIGHT_ABOVE_TABLE_RANGE_M[1]
                    )
        ):
            top_z = color_top_z
            measured_height = color_height
            depth_used = False
            depth_rms = 0.0
        if not HEIGHT_ABOVE_TABLE_RANGE_M[0] <= measured_height <= HEIGHT_ABOVE_TABLE_RANGE_M[1]:
            diagnostics['rejections'].append(f'height:{measured_height * 1000.0:.1f}mm above table')
            continue
        diagnostics['height_pass'] += 1
        inlier_mask = valid_component & (np.abs(z_base - top_z) <= GREEN_DEPTH_CLUSTER_HALF_BAND_M)
        rows, cols = np.nonzero(inlier_mask)
        try:
            center, _corners, sides, aspect = projected_rectangle_geometry(
                center_uv, box_uv_float, robot["T_base_camera"], top_z
            )
        except RuntimeError as error:
            diagnostics['rejections'].append(f'projection:{error}')
            continue
        if (
            not (
            side_range[0] <= sides[0] <= side_range[1]
            and side_range[0] <= sides[1] <= side_range[1]
            and aspect <= max_aspect
                    )
        ):
            diagnostics["rejections"].append(
                f"size:{sides[0]*1000.0:.1f}x{sides[1]*1000.0:.1f}mm,"
                f"aspect={aspect:.2f},z={top_z*1000.0:.1f}mm"
            )
            continue
        diagnostics['metric_size_pass'] += 1
        if expected_center_xy is not None:
            shift = float(np.linalg.norm(center[:2] - expected_center_xy))
            if shift > MAX_REFINED_CENTER_SHIFT_M:
                continue
        else:
            shift = 0.0
        mean_side = float(np.mean(sides))
        candidate = {
            "center_base_m": center,
            "normal_base": np.array([0.0, 0.0, 1.0], dtype=np.float64),
            "side_lengths_m": sides,
            "rectangle_angle_deg": float(image_rectangle[2]),
            "aspect": aspect,
            "plane_rms_m": depth_rms,
            "tilt_deg": 0.0,
            "center_disagreement_m": 0.0,
            "valid_points": int(rows.size),
            "rows": rows,
            "cols": cols,
            "mask": inlier_mask,
            "center_uv": center_uv,
            "box_uv": box_uv_float.astype(np.int32),
            "green_area_px": green_area,
            "green_solidity": solidity,
            "green_rectangularity": rectangularity,
            "green_polygon_vertices": polygon_vertices,
            "table_z_base_m": table_z,
            "height_above_table_m": measured_height,
            "depth_method": (
                "sparse_green_depth_plus_horizontal_color_plane"
                if depth_used
                else "known_30mm_color_scale_horizontal_plane"
            ),
            "color_scale_z_base_m": color_top_z,
        }
        candidate["score"] = (
            abs(mean_side - EXPECTED_TOP_SIDE_M) / 0.010
            + 2.5 * abs(aspect - 1.0)
            + 2.0 * (1.0 - rectangularity)
            + 0.20 * abs(polygon_vertices - 4)
            + abs(measured_height - EXPECTED_HEIGHT_ABOVE_TABLE_M) / 0.020
            + depth_rms / 0.002
            + shift / 0.020
        )
        candidates.append(candidate)
    candidates.sort(key=lambda item: item['score'])
    return (candidates, x_base, y_base, z_base, diagnostics)


def vertical_camera_rotation(current: np.ndarray) -> np.ndarray:
    z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    x_axis = np.asarray(current[:3, 0], dtype=np.float64).copy()
    x_axis[2] = 0.0
    if np.linalg.norm(x_axis) < 1e-06:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if not np.all(np.isfinite(rotation)) or abs(np.linalg.det(rotation) - 1.0) > 1e-06:
        raise RuntimeError('Could not construct a vertical camera rotation')
    return rotation


def pose_plan(initial_robot: dict, coarse: dict) -> dict:
    top = np.asarray(coarse['center_base_m'], dtype=np.float64)
    desired_camera = np.eye(4, dtype=np.float64)
    desired_camera[:3, :3] = vertical_camera_rotation(initial_robot['T_base_camera'][:3, :3])
    desired_camera[:3, 3] = [float(top[0]), float(top[1]), float(top[2] + CAMERA_CLEARANCE_ABOVE_TOP_M)]
    camera_translation = float(np.linalg.norm(desired_camera[:3, 3] - initial_robot['T_base_camera'][:3, 3]))
    if camera_translation > MAX_CAMERA_TRANSLATION_M:
        raise RuntimeError(
            f"Required camera translation is {camera_translation*1000:.1f} mm; "
            f"limit is {MAX_CAMERA_TRANSLATION_M*1000:.0f} mm"
        )
    flange = desired_camera @ vision.inverse_transform(vision.T_FLANGE_CAMERA)
    desired_tcp_transform = flange @ vision.pose6_to_transform(motion.ONROBOT_SINGLE_TCP)
    desired_pose = vision.transform_to_pose6(desired_tcp_transform)
    if float(desired_pose[2]) < float(top[2]) + MIN_TCP_CLEARANCE_AT_VIEW_M:
        raise RuntimeError(
            "Calibrated camera/tool geometry would put the OnRobot TCP less than "
            f"{MIN_TCP_CLEARANCE_AT_VIEW_M*1000:.0f} mm above the top"
        )
    current_tcp_transform = initial_robot['T_base_flange'] @ vision.pose6_to_transform(motion.ONROBOT_SINGLE_TCP)
    start_pose = vision.transform_to_pose6(current_tcp_transform)
    safe_z = max(
        float(start_pose[2]),
        float(desired_pose[2]) + SAFE_TRANSITION_EXTRA_M,
        float(top[2]) + SAFE_TCP_CLEARANCE_ABOVE_TOP_M,
    )
    if safe_z > MAX_SAFE_TCP_Z_M:
        raise RuntimeError(
            f"Required safe TCP Z is {safe_z*1000:.1f} mm; "
            f"limit is {MAX_SAFE_TCP_Z_M*1000:.0f} mm"
        )
    lift_pose = list(start_pose)
    lift_pose[2] = safe_z
    high_target_pose = list(desired_pose)
    high_target_pose[2] = safe_z
    for name, pose in (('start', start_pose), ('lift', lift_pose), ('high target', high_target_pose), ('centered view', desired_pose)):
        if not np.all(np.isfinite(pose)):
            raise RuntimeError(f'Non-finite {name} pose')
    return {
        "T_base_camera": desired_camera,
        "start_pose": start_pose,
        "lift_pose": lift_pose,
        "high_target_pose": high_target_pose,
        "desired_pose": desired_pose,
        "camera_translation_m": camera_translation,
    }


def stationary_refinement_is_sufficient(plan: dict, coarse: dict) -> bool:
    """True when a fresh capture is preferable to an unnecessary robot move."""
    return (
        float(plan["camera_translation_m"])
        <= STATIONARY_REFINEMENT_MAX_CAMERA_TRANSLATION_M
        and float(coarse["tilt_deg"])
        <= STATIONARY_REFINEMENT_MAX_TOP_TILT_DEG
        and float(coarse["aspect"])
        <= STATIONARY_REFINEMENT_MAX_ASPECT
        and float(coarse["plane_rms_m"])
        <= STATIONARY_REFINEMENT_MAX_PLANE_RMS_M
    )


def connect_control():
    from rtde_control import RTDEControlInterface
    control = RTDEControlInterface(motion.ROBOT_IP)
    time.sleep(0.4)
    control.setTcp(motion.ONROBOT_SINGLE_TCP)
    control.setPayload(motion.EXPECTED_PAYLOAD_KG, motion.EXPECTED_PAYLOAD_COG_M)
    return control


def checked_move_l(control, pose: list[float], label: str) -> None:
    print(f'MOVE {label}: {[round(float(value), 6) for value in pose]}', flush=True)
    if hasattr(control, 'isPoseWithinSafetyLimits'):
        if not control.isPoseWithinSafetyLimits(pose):
            raise RuntimeError(f'Controller safety-limit check rejected {label}')
    if control.moveL(pose, MOTION_SPEED_M_S, MOTION_ACCEL_M_S2) is False:
        raise RuntimeError(f'moveL returned False at {label}')


def verify_robot_unchanged(initial_robot: dict) -> None:
    receiver = None
    try:
        receiver = vision.connect_receiver_with_retry()
        tcp_offset, _ = vision.read_active_tcp_offset()
        current = vision.robot_snapshot(receiver, tcp_offset)
        position_mm = float(
            np.linalg.norm(
                current["T_base_flange"][:3, 3]
                - initial_robot["T_base_flange"][:3, 3]
            )
            * 1000.0
        )
        orientation_deg = float(
            vision.rotation_difference_deg(
                initial_robot["T_base_flange"], current["T_base_flange"]
            )
        )
        speed = np.asarray(current['actual_tcp_speed_m_rad_s'], dtype=np.float64)
        if position_mm > 1.0 or orientation_deg > 0.2:
            raise RuntimeError(f'Robot moved after coarse capture: {position_mm:.2f} mm, {orientation_deg:.3f} deg')
        if np.linalg.norm(speed[:3]) > 0.002 or np.linalg.norm(speed[3:]) > 0.02:
            raise RuntimeError('Robot is moving before workbench view motion')
    finally:
        if receiver is not None:
            try:
                receiver.disconnect()
            except Exception:
                pass


def move_to_centered_view(plan: dict, initial_robot: dict) -> None:
    verify_robot_unchanged(initial_robot)
    motion.verify_remote_and_stopped()
    control = None
    try:
        control = connect_control()
        checked_move_l(control, plan['lift_pose'], 'WORKBENCH_VERTICAL_LIFT')
        checked_move_l(control, plan['high_target_pose'], 'WORKBENCH_HIGH_TRANSITION')
        checked_move_l(control, plan['desired_pose'], 'WORKBENCH_CENTERED_VIEW')
    finally:
        motion.stop_rtde_control(control)


def retrace_to_start(plan: dict) -> None:
    motion.verify_remote_and_stopped()
    control = None
    try:
        control = connect_control()
        checked_move_l(control, plan['high_target_pose'], 'WORKBENCH_RETRACT')
        checked_move_l(control, plan['lift_pose'], 'WORKBENCH_REVERSE_TRANSITION')
        checked_move_l(control, plan['start_pose'], 'WORKBENCH_RETURN_START')
    finally:
        motion.stop_rtde_control(control)


def pixel_for_center(candidate: dict, x_base: np.ndarray, y_base: np.ndarray) -> np.ndarray:
    if 'center_uv' in candidate:
        return np.asarray(candidate['center_uv'], dtype=np.float64).copy()
    mask = candidate['mask']
    rows, cols = np.nonzero(mask)
    points = np.column_stack((x_base[mask], y_base[mask]))
    reference = np.mean(points, axis=0)
    design = np.column_stack((points - reference, np.ones(points.shape[0], dtype=np.float64)))
    target = np.column_stack((cols, rows)).astype(np.float64)
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    center_xy = candidate['center_base_m'][:2]
    if rank >= 3:
        pixel = np.append(center_xy - reference, 1.0) @ coefficients
    else:
        pixel = np.array([np.mean(cols), np.mean(rows)], dtype=np.float64)
    return pixel


def json_candidate(candidate: dict) -> dict:
    result = {
        "center_base_m": candidate["center_base_m"].astype(float).tolist(),
        "normal_base": candidate["normal_base"].astype(float).tolist(),
        "top_size_mm": (
            candidate["side_lengths_m"] * 1000.0
        ).astype(float).tolist(),
        "top_rectangle_angle_deg": float(candidate["rectangle_angle_deg"]),
        "aspect_ratio": float(candidate["aspect"]),
        "plane_rms_mm": float(candidate["plane_rms_m"] * 1000.0),
        "top_tilt_deg": float(candidate["tilt_deg"]),
        "center_estimator_disagreement_mm": float(
            candidate["center_disagreement_m"] * 1000.0
        ),
        "valid_depth_points": int(candidate["valid_points"]),
    }
    if 'green_area_px' in candidate:
        result['green_area_px'] = int(candidate['green_area_px'])
    if 'depth_method' in candidate:
        result['depth_method'] = str(candidate['depth_method'])
    if 'color_scale_z_base_m' in candidate:
        result['color_scale_z_base_m'] = float(candidate['color_scale_z_base_m'])
    if 'green_solidity' in candidate:
        result['green_solidity'] = float(candidate['green_solidity'])
    if 'green_rectangularity' in candidate:
        result['green_rectangularity'] = float(candidate['green_rectangularity'])
    if 'green_polygon_vertices' in candidate:
        result['green_polygon_vertices'] = int(candidate['green_polygon_vertices'])
    if 'table_z_base_m' in candidate:
        result['table_z_base_m'] = float(candidate['table_z_base_m'])
    if 'height_above_table_m' in candidate:
        result['height_above_table_mm'] = float(candidate['height_above_table_m'] * 1000.0)
    return result


def print_detection_diagnostics(label: str, diagnostics: dict) -> None:
    print(
        f"{label} diagnostics: green pixels={diagnostics['green_pixels']}; "
        f"contours={diagnostics['contours']}; area pass={diagnostics['area_pass']}; "
        f"shape pass={diagnostics['shape_pass']}; "
        f"rectangle pass={diagnostics['rectangle_pass']}; "
        f"color-Z pass={diagnostics['color_scale_z_pass']}; "
        f"height pass={diagnostics['height_pass']}; "
        f"depth used={diagnostics['sparse_depth_used']}; "
        f"size pass={diagnostics['metric_size_pass']}",
        flush=True,
    )
    rejections = diagnostics.get('rejections', [])
    if rejections:
        print(f'{label} rejection examples: ' + '; '.join(rejections[:8]), flush=True)


def save_failure_diagnostic(color: np.ndarray, diagnostics: dict, label: str) -> None:
    mask = green_top_mask(color)
    overlay = color.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    view = np.hstack((overlay, mask_bgr))
    cv2.rectangle(view, (0, 0), (view.shape[1], 50), (0, 0, 0), -1)
    text = (
        f"{label} FAIL | green {diagnostics['green_pixels']} | "
        f"area {diagnostics['area_pass']} | shape {diagnostics['shape_pass']} | "
        f"rectangle {diagnostics['rectangle_pass']} | "
        f"color-Z {diagnostics['color_scale_z_pass']} | "
        f"height {diagnostics['height_pass']} | "
        f"size {diagnostics['metric_size_pass']}"
    )
    cv2.putText(view, text, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    vision.save_png(OUTPUT_IMAGE, view)


def save_diagnostic(color: np.ndarray, candidate: dict, pixel: np.ndarray) -> None:
    view = color.copy()
    if 'box_uv' in candidate:
        cv2.drawContours(view, [candidate['box_uv']], -1, (0, 255, 0), 3)
    else:
        rows = candidate['rows']
        cols = candidate['cols']
        left, right = (int(np.min(cols)), int(np.max(cols)))
        top, bottom = (int(np.min(rows)), int(np.max(rows)))
        cv2.rectangle(view, (left, top), (right, bottom), (0, 255, 0), 3)
    center = (int(round(float(pixel[0]))), int(round(float(pixel[1]))))
    cv2.drawMarker(view, center, (0, 0, 255), cv2.MARKER_CROSS, 26, 3)
    cv2.rectangle(view, (0, 0), (view.shape[1], 42), (0, 0, 0), -1)
    size = candidate['side_lengths_m'] * 1000.0
    cv2.putText(
        view,
        (
            f"GREEN WORKBENCH TOP | {size[0]:.1f}x{size[1]:.1f} mm | "
            f"plane RMS {candidate['plane_rms_m']*1000.0:.2f} mm"
        ),
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )
    vision.save_png(OUTPUT_IMAGE, view)


def main() -> int:
    print(VERSION, flush=True)
    print('Dynamic-height green-top RGB-D calibration; no gripper command.', flush=True)
    print('Live RGB-D determines Base XYZ; candidates must be about 265 mm above the detected table.', flush=True)
    print(
        "Only the complete green 30 x 30 mm top must be visible; the white base "
        "does not need valid depth.",
        flush=True,
    )
    if len(sys.argv) != 1:
        raise RuntimeError('Run this file without command-line arguments')
    moved = False
    returned = False
    try:
        print('[1/4] Coarse RGB-D capture of the EMPTY workbench top...', flush=True)
        color, depth, capture, initial_robot = vision.capture_at_current_pose('workbench_coarse')
        candidates, x_base, y_base, z_base, coarse_diagnostics = find_top_candidates(
            color, depth, initial_robot, refined=False
        )
        print_detection_diagnostics('Coarse', coarse_diagnostics)
        if not candidates:
            save_failure_diagnostic(color, coarse_diagnostics, 'COARSE')
            raise RuntimeError(
                "No valid green 23-40 mm square top about 265 mm above the table "
                "was found; see diagnostics "
                "above and workbench_calibration.png"
            )
        coarse = candidates[0]
        coarse_json = json_candidate(coarse)
        center_mm = (coarse['center_base_m'] * 1000.0).round(2).tolist()
        size_mm = (coarse['side_lengths_m'] * 1000.0).round(2).tolist()
        print(
            f"Coarse top Base {center_mm} mm; size {size_mm} mm; "
            f"tilt {coarse['tilt_deg']:.2f} deg; "
            f"Z source {coarse['depth_method']}.",
            flush=True,
        )
        table_z = float(coarse_diagnostics['table_z_base_m'])
        measured_height = float(coarse['center_base_m'][2] - table_z)
        print(
            f"Estimated table Z {table_z*1000.0:.1f} mm; "
            f"workbench height {measured_height*1000.0:.1f} mm "
            f"(required {HEIGHT_ABOVE_TABLE_RANGE_M[0]*1000.0:.0f}-"
            f"{HEIGHT_ABOVE_TABLE_RANGE_M[1]*1000.0:.0f} mm).",
            flush=True,
        )
        plan = pose_plan(initial_robot, coarse)
        stationary_refinement = stationary_refinement_is_sufficient(plan, coarse)
        if stationary_refinement:
            print(
                f"[2/4] Top is already in a valid close view: planned camera "
                f"translation is only {plan['camera_translation_m']*1000.0:.1f} mm.",
                flush=True,
            )
            print('No robot motion is needed; keeping the complete TCP pose unchanged.', flush=True)
        else:
            print(
                f"[2/4] Planned vertical camera clearance: "
                f"{CAMERA_CLEARANCE_ABOVE_TOP_M*1000:.0f} mm; "
                f"camera translation {plan['camera_translation_m']*1000.0:.1f} mm.",
                flush=True,
            )
            print(
                "The entire arm, RG2, camera and cable paths must be clear. "
                "Motion starts in 5 seconds; keep pendant Stop reachable.",
                flush=True,
            )
            for remaining in range(5, 0, -1):
                print(f'  {remaining}', flush=True)
                time.sleep(1.0)
            moved = True
            move_to_centered_view(plan, initial_robot)
        print('[3/4] Fresh RGB-D refinement...', flush=True)
        refined_color, refined_depth, refined_capture, refined_robot = (
            vision.capture_at_current_pose("workbench_centered")
        )
        refined_candidates, rx, ry, _, refined_diagnostics = find_top_candidates(
            refined_color,
            refined_depth,
            refined_robot,
            refined=True,
            expected_center_xy=coarse["center_base_m"][:2],
            expected_top_z=float(coarse["center_base_m"][2]),
            expected_table_z=table_z,
        )
        print_detection_diagnostics('Refined', refined_diagnostics)
        if not refined_candidates:
            save_failure_diagnostic(refined_color, refined_diagnostics, 'REFINED')
            raise RuntimeError(
                "Centered view did not contain a valid green 24-40 mm square top "
                "about 265 mm above the table; "
                "see diagnostics above and workbench_calibration.png"
            )
        refined = refined_candidates[0]
        shift = float(np.linalg.norm(refined['center_base_m'][:2] - coarse['center_base_m'][:2]))
        if shift > MAX_REFINED_CENTER_SHIFT_M:
            raise RuntimeError(f'Refined centre shifted {shift * 1000.0:.1f} mm from coarse centre')
        pixel = pixel_for_center(refined, rx, ry)
        save_diagnostic(refined_color, refined, pixel)
        result = {
            "schema": "ur5e_green_workbench_dynamic_height_calibration_v2",
            "version": VERSION,
            "created_at_utc": utc_now(),
            "status": (
                "calibrated_no_motion"
                if stationary_refinement
                else "calibrated_return_pending"
            ),
            "robot_commands_sent": bool(moved),
            "gripper_commands_sent": False,
            "transform_convention": "T_A_B maps coordinates in B into A",
            "workbench_nominal": {
                "height_above_table_mm": 265.0,
                "accepted_height_range_mm": [240.0, 290.0],
                "height_used_as_detection_constraint": True,
                "top_size_mm": [30.0, 30.0],
                "xy_may_move_after_calibration": True,
                "top_height_must_remain_constant_during_follow": True,
            },
            "coarse_detection": coarse_json,
            "refined_top": json_candidate(refined),
            "center_base_m": refined["center_base_m"].astype(float).tolist(),
            "normal_base": refined["normal_base"].astype(float).tolist(),
            "coarse_to_refined_xy_shift_mm": shift * 1000.0,
            "estimated_table_z_base_m": table_z,
            "estimated_height_above_table_mm": (
                measured_height * 1000.0
            ),
            "handeye": {
                "T_flange_camera": vision.T_FLANGE_CAMERA.tolist(),
                "source": "existing Gemini305 eye-in-hand calibration",
            },
            "camera": {
                "serial": vision.CAMERA_SERIAL,
                "K": vision.COLOR_K.tolist(),
                "D_opencv_k1_k2_p1_p2_k3_k4_k5_k6": vision.COLOR_D.tolist(),
                "coarse_capture": capture,
                "refined_capture": refined_capture,
            },
            "robot_at_coarse_capture": vision.json_ready_snapshot(initial_robot),
            "robot_at_refined_capture": vision.json_ready_snapshot(refined_robot),
            "motion_plan_base_tcp_m_rad": {
                "strategy": (
                    "stationary_fresh_capture"
                    if stationary_refinement
                    else "vertical_active_view_and_retrace"
                ),
                "lift": plan["lift_pose"],
                "high_target": plan["high_target_pose"],
                "centered_view": plan["desired_pose"],
            },
            "output_image": OUTPUT_IMAGE.name,
        }
        write_json(OUTPUT_JSON, result)
        if moved:
            print('[4/4] Retracing the verified Cartesian path to start...', flush=True)
            retrace_to_start(plan)
            returned = True
            result['status'] = 'calibrated_and_returned_to_start'
        else:
            print('[4/4] No return motion: the robot never moved.', flush=True)
            returned = True
            result['status'] = 'calibrated_without_robot_motion'
        result['finished_at_utc'] = utc_now()
        write_json(OUTPUT_JSON, result)
        center_mm = (refined['center_base_m'] * 1000.0).round(3).tolist()
        print(f'WORKBENCH_CALIBRATION_OK: Base top centre {center_mm} mm', flush=True)
        print(f'Saved: {OUTPUT_JSON}', flush=True)
        print(f'Saved: {OUTPUT_IMAGE}', flush=True)
        print('Keep the calibrated top height unchanged; XY may move on the same table.', flush=True)
        return 0
    except Exception as error:
        print(f'STOPPED: {error}', file=sys.stderr, flush=True)
        if moved and (not returned):
            print(
                "Robot may remain at/above the workbench view pose. Use the pendant; "
                "no automatic recovery was attempted after the failure.",
                file=sys.stderr,
                flush=True,
            )
        if '--debug' in sys.argv:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
