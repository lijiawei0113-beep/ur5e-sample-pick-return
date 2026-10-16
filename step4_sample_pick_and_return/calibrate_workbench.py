from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math
import sys
import traceback

import cv2
import numpy as np

from robot import PersistentRTDEReceive
import vision


VERSION = 'STEP6_NONFOLLOW_WORKBENCH_DEPTH_PLACEMENT_V4_COHERENT_EDGE'
ROOT = Path(__file__).resolve().parent
OUTPUT_JSON = ROOT / 'workbench_pose.json'
OUTPUT_IMAGE = ROOT / 'workbench_nonfollow_place_result.png'

# Top-plane extraction
MIN_HEIGHT_ABOVE_TABLE_M = 0.25
MAX_HEIGHT_ABOVE_TABLE_M = 0.38
TABLE_Z_BASE_RANGE_M = (-0.12, 0.22)
Z_HISTOGRAM_STEP_M = 0.002
TABLE_REFINEMENT_HALF_BAND_M = 0.006
TOP_SEED_HALF_BAND_M = 0.01
MIN_COMPONENT_DEPTH_POINTS = 120
MIN_COMPONENT_IMAGE_AREA_PX = 250.0
IMAGE_BORDER_MARGIN_PX = 5
MAX_TOP_TILT_DEG = 12.0
MAX_TOP_PLANE_RMS_M = 0.0045
MIN_PLANE_INLIERS = 100
MIN_TOP_AREA_M2 = 0.002
MAX_TOP_AREA_M2 = 0.1
MIN_TOP_SPAN_M = 0.04
MAX_TOP_SPAN_M = 0.6
MORPH_OPEN_KERNEL_PX = 3
MORPH_CLOSE_KERNEL_PX = 5
BOUNDARY_CLOSE_KERNEL_PX = 3
MAX_HEIGHT_SEEDS = 6

# Boundary refinement
EDGE_SEARCH_RADIUS_PX = 7
EDGE_MIN_GRADIENT = 35.0
EDGE_MIN_SUPPORT_FRACTION = 0.6
EDGE_DISTANCE_PENALTY = 2.0
EDGE_MIN_NORMAL_ALIGNMENT = 0.65
EDGE_LOCAL_WINDOW = 7
EDGE_LOCAL_TOLERANCE_PX = 2.0
EDGE_MIN_SECTOR_SUPPORT = 0.35
EDGE_SUBPIXEL_LIMIT = 0.75
MAX_DEPTH_RGB_CENTER_DISAGREEMENT_M = 0.0015
MAX_DEPTH_INLIER_CENTER_DISAGREEMENT_M = 0.01


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def finite_workspace(x_base, y_base, z_base) -> np.ndarray:
    return (
        np.isfinite(x_base)
        & np.isfinite(y_base)
        & np.isfinite(z_base)
        & (x_base >= vision.BASE_X_RANGE_M[0])
        & (x_base <= vision.BASE_X_RANGE_M[1])
        & (y_base >= vision.BASE_Y_RANGE_M[0])
        & (y_base <= vision.BASE_Y_RANGE_M[1])
        & (z_base >= vision.BASE_Z_RANGE_M[0])
        & (z_base <= vision.BASE_Z_RANGE_M[1])
    )


def dominant_table_height(z_values: np.ndarray) -> tuple[float, dict]:
    values = np.asarray(z_values, dtype=np.float64)
    lower = max(TABLE_Z_BASE_RANGE_M[0], float(vision.BASE_Z_RANGE_M[0]))
    upper = min(TABLE_Z_BASE_RANGE_M[1], float(vision.BASE_Z_RANGE_M[1]))
    values = values[np.isfinite(values) & (values >= lower) & (values <= upper)]
    if values.size < 1000:
        raise RuntimeError('Too few valid points to estimate the supporting-table height')
    edges = np.arange(lower, upper + Z_HISTOGRAM_STEP_M, Z_HISTOGRAM_STEP_M)
    histogram, edges = np.histogram(values, bins=edges)
    if histogram.size == 0 or int(np.max(histogram)) < 100:
        raise RuntimeError('No dominant supporting-table height was found')
    index = int(np.argmax(histogram))
    seed = float((edges[index] + edges[index + 1]) * 0.5)
    inliers = values[np.abs(values - seed) <= TABLE_REFINEMENT_HALF_BAND_M]
    if inliers.size < 500:
        raise RuntimeError('Supporting-table height cluster is too sparse')
    table_z = float(np.median(inliers))
    mad = float(np.median(np.abs(inliers - table_z)))
    return table_z, {
        "valid_points": int(values.size),
        "cluster_points": int(inliers.size),
        "histogram_peak_points": int(histogram[index]),
        "mad_mm": mad * 1000.0,
    }


def height_seeds(z_values: np.ndarray) -> list[float]:
    values = np.asarray(z_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < MIN_COMPONENT_DEPTH_POINTS:
        return []
    lower = float(np.min(values)) - 0.5 * Z_HISTOGRAM_STEP_M
    upper = float(np.max(values)) + 1.5 * Z_HISTOGRAM_STEP_M
    edges = np.arange(lower, upper, Z_HISTOGRAM_STEP_M)
    if edges.size < 2:
        return [float(np.median(values))]
    histogram, edges = np.histogram(values, bins=edges)
    order = np.argsort(histogram)[::-1]
    seeds: list[float] = []
    for index in order:
        if int(histogram[index]) < MIN_COMPONENT_DEPTH_POINTS:
            break
        seed = float((edges[index] + edges[index + 1]) * 0.5)
        if all((abs(seed - previous) > TOP_SEED_HALF_BAND_M for previous in seeds)):
            seeds.append(seed)
        if len(seeds) >= MAX_HEIGHT_SEEDS:
            break
    return seeds


def robust_plane(points_xyz: np.ndarray) -> dict:
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if not np.all(np.isfinite(points)):
        raise RuntimeError('Non-finite points in the top-plane fit')
    keep = np.ones(points.shape[0], dtype=bool)
    for _ in range(5):
        selected = points[keep]
        if selected.shape[0] < MIN_PLANE_INLIERS:
            raise RuntimeError('Too few inliers for the top-plane fit')
        design = np.column_stack((selected[:, 0], selected[:, 1], np.ones(selected.shape[0])))
        coefficients, _, rank, _ = np.linalg.lstsq(design, selected[:, 2], rcond=None)
        if rank < 3:
            raise RuntimeError('Top-plane fit is rank deficient')
        residual_all = points[:, 2] - (
            coefficients[0] * points[:, 0]
            + coefficients[1] * points[:, 1]
            + coefficients[2]
        )
        residual_kept = residual_all[keep]
        median = float(np.median(residual_kept))
        mad = float(np.median(np.abs(residual_kept - median)))
        threshold = float(np.clip(3.5 * 1.4826 * mad, 0.0025, 0.008))
        updated = np.abs(residual_all - median) <= threshold
        if np.array_equal(updated, keep):
            break
        keep = updated
    selected = points[keep]
    if selected.shape[0] < MIN_PLANE_INLIERS:
        raise RuntimeError('Too few inliers for the final top-plane fit')
    design = np.column_stack((selected[:, 0], selected[:, 1], np.ones(selected.shape[0])))
    coefficients, _, rank, _ = np.linalg.lstsq(design, selected[:, 2], rcond=None)
    if rank < 3:
        raise RuntimeError('Final top-plane fit is rank deficient')
    residual = selected[:, 2] - (
        coefficients[0] * selected[:, 0]
        + coefficients[1] * selected[:, 1]
        + coefficients[2]
    )
    rms = float(np.sqrt(np.mean(residual * residual)))
    normal = np.array([-coefficients[0], -coefficients[1], 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal = -normal
    tilt_deg = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
    return {
        "coefficients": coefficients,
        "inlier_mask": keep,
        "normal_base": normal,
        "tilt_deg": tilt_deg,
        "rms_m": rms,
        "inlier_count": int(np.count_nonzero(keep)),
    }


def pixels_to_plane(pixels_uv: np.ndarray, t_base_camera: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(
        pixels,
        np.asarray(vision.COLOR_K, dtype=np.float64),
        np.asarray(vision.COLOR_D, dtype=np.float64),
    ).reshape(-1, 2)
    rays_camera = np.column_stack((normalized[:, 0], normalized[:, 1], np.ones(normalized.shape[0])))
    transform = np.asarray(t_base_camera, dtype=np.float64).reshape(4, 4)
    origin = transform[:3, 3]
    rays_base = (transform[:3, :3] @ rays_camera.T).T
    a, b, c = np.asarray(coefficients, dtype=np.float64).reshape(3)
    numerator = a * origin[0] + b * origin[1] + c - origin[2]
    denominator = rays_base[:, 2] - a * rays_base[:, 0] - b * rays_base[:, 1]
    if np.any(np.abs(denominator) < 1e-08):
        raise RuntimeError('A top boundary ray is parallel to the fitted plane')
    scale = numerator / denominator
    if np.any(scale <= 0.0) or np.any(scale > 3.0):
        raise RuntimeError('The fitted top plane is behind or too far from the camera')
    return origin.reshape(1, 3) + scale[:, None] * rays_base


def contour_touches_border(contour: np.ndarray, shape: tuple[int, int]) -> bool:
    height, width = shape
    x, y, w, h = cv2.boundingRect(contour)
    margin = IMAGE_BORDER_MARGIN_PX
    return x <= margin or y <= margin or x + w >= width - margin or (y + h >= height - margin)


def metric_polygon_geometry(points_xy: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    polygon = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    signed_area = float(cv2.contourArea(polygon, oriented=True))
    area = abs(signed_area)
    moments = cv2.moments(polygon)
    if abs(float(moments['m00'])) < 1e-10:
        center = np.mean(polygon.reshape(-1, 2), axis=0).astype(np.float64)
    else:
        center = np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']], dtype=np.float64)
    rectangle = cv2.minAreaRect(polygon)
    spans = np.sort(np.asarray(rectangle[1], dtype=np.float64))
    return (center, area, spans)


def refine_boundary_with_rgb_edge(color_bgr: np.ndarray, depth_contour: np.ndarray) -> tuple[np.ndarray, dict]:
    """Refine a depth boundary using supported, coherent normal RGB edges.

    This is not colour segmentation: the workbench was already identified by
    depth height and plane geometry.  RGB supplies only a local edge correction
    within a seven-pixel normal search around that verified depth boundary.
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0.0)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    points = np.asarray(depth_contour, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < EDGE_LOCAL_WINDOW:
        raise RuntimeError('Too few boundary pixels for RGB refinement')
    refined = points.copy()
    accepted = np.zeros(points.shape[0], dtype=bool)
    signed_shifts = np.zeros(points.shape[0], dtype=np.float64)
    height, width = gray.shape
    tangent_step = max(1, min(4, points.shape[0] // 40))
    for index, point in enumerate(points):
        previous = points[(index - tangent_step) % points.shape[0]]
        following = points[(index + tangent_step) % points.shape[0]]
        tangent = following - previous
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-06:
            continue
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64) / norm
        best_score = -float('inf')
        best_point = point
        best_gradient = 0.0
        best_shift = 0.0
        sampled_gradients: dict[int, float] = {}
        for shift in range(-EDGE_SEARCH_RADIUS_PX, EDGE_SEARCH_RADIUS_PX + 1):
            probe = point + float(shift) * normal
            u = int(round(float(probe[0])))
            v = int(round(float(probe[1])))
            if not (1 <= u < width - 1 and 1 <= v < height - 1):
                continue
            magnitude = float(gradient[v, u])
            alignment = abs((float(grad_x[v, u]) * normal[0] + float(grad_y[v, u]) * normal[1]) / max(magnitude, 1e-09))
            if alignment < EDGE_MIN_NORMAL_ALIGNMENT:
                continue
            normal_gradient = magnitude * alignment
            sampled_gradients[shift] = normal_gradient
            score = normal_gradient - EDGE_DISTANCE_PENALTY * abs(float(shift))
            if score > best_score:
                best_score = score
                best_point = probe
                best_gradient = normal_gradient
                best_shift = float(shift)
        if best_gradient >= EDGE_MIN_GRADIENT:
            best_integer = int(round(best_shift))
            if best_integer - 1 in sampled_gradients and best_integer + 1 in sampled_gradients:
                left = sampled_gradients[best_integer - 1]
                middle = sampled_gradients[best_integer]
                right = sampled_gradients[best_integer + 1]
                denominator = left - 2.0 * middle + right
                if denominator < -1e-09:
                    subpixel = 0.5 * (left - right) / denominator
                    subpixel = float(np.clip(subpixel, -EDGE_SUBPIXEL_LIMIT, EDGE_SUBPIXEL_LIMIT))
                    best_shift += subpixel
                    best_point = point + best_shift * normal
            refined[index] = best_point
            accepted[index] = True
            signed_shifts[index] = best_shift
    initial = accepted.copy()
    half = EDGE_LOCAL_WINDOW // 2
    for index in range(points.shape[0]):
        neighbors = np.arange(index - half, index + half + 1) % points.shape[0]
        supported = initial[neighbors]
        if not initial[index] or np.count_nonzero(supported) < half + 1:
            accepted[index] = False
            continue
        median_shift = float(np.median(signed_shifts[neighbors][supported]))
        if abs(signed_shifts[index] - median_shift) > EDGE_LOCAL_TOLERANCE_PX:
            accepted[index] = False
    refined[~accepted] = points[~accepted]
    support = float(np.mean(accepted)) if accepted.size else 0.0
    sectors = np.array_split(accepted, 8)
    sector_support = [float(np.mean(sector)) for sector in sectors]
    coverage_ok = all((value >= EDGE_MIN_SECTOR_SUPPORT for value in sector_support))
    accepted_shifts = np.abs(signed_shifts[accepted])
    diagnostics = {
        "support_fraction": support,
        "passes_support_threshold": bool(
            support >= EDGE_MIN_SUPPORT_FRACTION and coverage_ok
        ),
        "sector_support_fractions": sector_support,
        "minimum_sector_support_fraction": EDGE_MIN_SECTOR_SUPPORT,
        "normal_alignment_threshold": EDGE_MIN_NORMAL_ALIGNMENT,
        "accepted_points": int(np.count_nonzero(accepted)),
        "total_points": int(points.shape[0]),
        "median_shift_px": (
            float(np.median(accepted_shifts))
            if accepted_shifts.size else None
        ),
        "max_shift_px": (
            float(np.max(accepted_shifts))
            if accepted_shifts.size else None
        ),
        "search_radius_px": int(EDGE_SEARCH_RADIUS_PX),
        "minimum_gradient": float(EDGE_MIN_GRADIENT),
    }
    return (refined, diagnostics)


def project_base_point_to_color(point_base: np.ndarray, t_base_camera: np.ndarray) -> np.ndarray:
    transform = np.asarray(t_base_camera, dtype=np.float64).reshape(4, 4)
    point = np.append(np.asarray(point_base, dtype=np.float64).reshape(3), 1.0)
    point_camera = np.linalg.inv(transform) @ point
    if point_camera[2] <= 0.0:
        raise RuntimeError('Placement centre is behind the camera')
    projected, _ = cv2.projectPoints(
        point_camera[:3].reshape(1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.asarray(vision.COLOR_K, dtype=np.float64),
        np.asarray(vision.COLOR_D, dtype=np.float64),
    )
    return projected.reshape(2).astype(np.float64)


def candidate_from_component(
    component: np.ndarray, color_bgr: np.ndarray,
    x_base: np.ndarray, y_base: np.ndarray, z_base: np.ndarray,
    t_base_camera: np.ndarray, table_z: float,
) -> tuple[dict | None, str | None]:
    valid = component & np.isfinite(x_base) & np.isfinite(y_base) & np.isfinite(z_base)
    rows, cols = np.nonzero(valid)
    if rows.size < MIN_COMPONENT_DEPTH_POINTS:
        return (None, 'depth_points')
    points = np.column_stack((x_base[valid], y_base[valid], z_base[valid]))
    try:
        plane = robust_plane(points)
    except RuntimeError as error:
        return (None, f'plane:{error}')
    if plane['tilt_deg'] > MAX_TOP_TILT_DEG:
        return (None, f"tilt:{plane['tilt_deg']:.1f}deg")
    if plane['rms_m'] > MAX_TOP_PLANE_RMS_M:
        return (None, f"plane_rms:{plane['rms_m'] * 1000.0:.1f}mm")
    inlier_component = np.zeros(component.shape, dtype=np.uint8)
    inlier_component[rows[plane['inlier_mask']], cols[plane['inlier_mask']]] = 255
    close_kernel = np.ones((BOUNDARY_CLOSE_KERNEL_PX, BOUNDARY_CLOSE_KERNEL_PX), dtype=np.uint8)
    inlier_component = cv2.morphologyEx(inlier_component, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    inlier_component = cv2.medianBlur(inlier_component, 3)
    contours, _ = cv2.findContours(inlier_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return (None, 'no_boundary')
    contour = max(contours, key=cv2.contourArea)
    image_area = float(cv2.contourArea(contour))
    if image_area < MIN_COMPONENT_IMAGE_AREA_PX:
        return (None, f'image_area:{image_area:.0f}px')
    if contour_touches_border(contour, component.shape):
        return (None, 'image_border')
    depth_boundary_uv = contour.reshape(-1, 2).astype(np.float64)
    try:
        depth_boundary_xyz = pixels_to_plane(depth_boundary_uv, t_base_camera, plane['coefficients'])
    except RuntimeError as error:
        return (None, f'projection:{error}')
    depth_center_xy, depth_area_m2, depth_spans = metric_polygon_geometry(depth_boundary_xyz[:, :2])
    plane_points = points[plane['inlier_mask']]
    inlier_center_xy = np.mean(plane_points[:, :2], axis=0)
    depth_inlier_disagreement = float(np.linalg.norm(depth_center_xy - inlier_center_xy))
    if depth_inlier_disagreement > MAX_DEPTH_INLIER_CENTER_DISAGREEMENT_M:
        return (None, f'depth_inlier_center_disagreement:{depth_inlier_disagreement * 1000.0:.2f}mm')
    center_xy = depth_center_xy
    area_m2 = depth_area_m2
    spans = depth_spans
    selected_boundary_uv = depth_boundary_uv
    center_method = 'depth_metric_area_centroid_fallback'
    depth_rgb_disagreement = None
    refined_inlier_disagreement = depth_inlier_disagreement
    edge_diagnostics = {
        "used": False,
        "support_fraction": 0.0,
        "passes_support_threshold": False,
        "fallback_reason": "RGB edge refinement was unavailable",
    }
    try:
        rgb_boundary_uv, rgb_edge_diagnostics = refine_boundary_with_rgb_edge(color_bgr, depth_boundary_uv)
        rgb_boundary_xyz = pixels_to_plane(rgb_boundary_uv, t_base_camera, plane['coefficients'])
        rgb_center_xy, rgb_area_m2, rgb_spans = metric_polygon_geometry(rgb_boundary_xyz[:, :2])
        depth_rgb_disagreement = float(np.linalg.norm(rgb_center_xy - depth_center_xy))
        rgb_inlier_disagreement = float(np.linalg.norm(rgb_center_xy - inlier_center_xy))
        edge_ok = bool(
            rgb_edge_diagnostics["passes_support_threshold"]
            and depth_rgb_disagreement
            <= MAX_DEPTH_RGB_CENTER_DISAGREEMENT_M
        )
        edge_diagnostics = dict(rgb_edge_diagnostics)
        edge_diagnostics['used'] = edge_ok
        edge_diagnostics['rgb_inlier_center_disagreement_mm'] = rgb_inlier_disagreement * 1000.0
        if (
            rgb_edge_diagnostics["passes_support_threshold"]
            and depth_rgb_disagreement > MAX_DEPTH_RGB_CENTER_DISAGREEMENT_M
        ):
            return (None, f'ambiguous_depth_rgb_center:{depth_rgb_disagreement * 1000.0:.2f}mm')
        if edge_ok:
            center_xy = rgb_center_xy
            area_m2 = rgb_area_m2
            spans = rgb_spans
            selected_boundary_uv = rgb_boundary_uv
            center_method = 'rgb_edge_refined_metric_area_centroid_on_depth_plane'
            refined_inlier_disagreement = rgb_inlier_disagreement
            edge_diagnostics['fallback_reason'] = None
        else:
            reasons = []
            if not rgb_edge_diagnostics['passes_support_threshold']:
                reasons.append(f"RGB support {rgb_edge_diagnostics['support_fraction'] * 100.0:.1f}%")
            if depth_rgb_disagreement > MAX_DEPTH_RGB_CENTER_DISAGREEMENT_M:
                reasons.append(f'depth/RGB centres differ {depth_rgb_disagreement * 1000.0:.2f} mm')
            edge_diagnostics['fallback_reason'] = '; '.join(reasons)
    except RuntimeError as error:
        edge_diagnostics['fallback_reason'] = str(error)
    if not MIN_TOP_AREA_M2 <= area_m2 <= MAX_TOP_AREA_M2:
        return (None, f'metric_area:{area_m2 * 10000.0:.1f}cm2')
    if (
        spans.size != 2 or not (
        MIN_TOP_SPAN_M <= spans[0] <= MAX_TOP_SPAN_M
        and MIN_TOP_SPAN_M <= spans[1] <= MAX_TOP_SPAN_M
            )
    ):
        return (None, f'span:{(spans * 1000.0).round(1).tolist()}mm')
    a, b, c = plane['coefficients']
    center_z = float(a * center_xy[0] + b * center_xy[1] + c)
    height = center_z - float(table_z)
    if not MIN_HEIGHT_ABOVE_TABLE_M <= height <= MAX_HEIGHT_ABOVE_TABLE_M:
        return (None, f'height:{height * 1000.0:.1f}mm')
    center = np.array([center_xy[0], center_xy[1], center_z], dtype=np.float64)
    score = float(
        area_m2
        * (plane["inlier_count"] / max(points.shape[0], 1))
        / (1.0 + plane["tilt_deg"] / 10.0 + plane["rms_m"] / 0.003)
    )
    return {
        "center_base_m": center,
        "normal_base": plane["normal_base"],
        "height_above_table_m": float(height),
        "table_z_base_m": float(table_z),
        "plane_coefficients_z_ax_by_c": plane["coefficients"],
        "plane_rms_m": float(plane["rms_m"]),
        "tilt_deg": float(plane["tilt_deg"]),
        "metric_area_m2": float(area_m2),
        "span_lengths_m": spans,
        "valid_depth_points": int(points.shape[0]),
        "plane_inliers": int(plane["inlier_count"]),
        "image_area_px": float(image_area),
        "center_method": center_method,
        "detection_method": (
            "depth_height_horizontal_plane_with_optional_generic_rgb_edge_refinement"
        ),
        "depth_center_base_xy_m": depth_center_xy,
        "inlier_mean_base_xy_m": inlier_center_xy,
        "depth_rgb_center_disagreement_m": depth_rgb_disagreement,
        "refined_inlier_center_disagreement_m": refined_inlier_disagreement,
        "depth_inlier_center_disagreement_m": depth_inlier_disagreement,
        "depth_metric_area_m2": float(depth_area_m2),
        "depth_span_lengths_m": depth_spans,
        "edge_refinement": edge_diagnostics,
        "depth_contour_uv": contour,
        "refined_contour_uv": np.rint(selected_boundary_uv).astype(np.int32).reshape(-1, 1, 2),
        "center_uv": project_base_point_to_color(center, t_base_camera),
        "mask": inlier_component > 0,
        "score": score,
    }, None


def extract_workbench_placement_point(color_bgr: np.ndarray, depth_mm: np.ndarray, robot: dict) -> tuple[dict, dict]:
    """Return the best placement point and diagnostics for one RGB-D capture."""
    x_base, y_base, z_base, _, _ = vision.depth_to_base_maps(depth_mm, robot['T_base_camera'])
    if color_bgr.shape[:2] != z_base.shape:
        raise RuntimeError('Aligned color and depth sizes do not match')
    workspace = finite_workspace(x_base, y_base, z_base)
    table_z, table_diagnostics = dominant_table_height(z_base[workspace])
    relative_height = z_base - table_z
    elevated = (
        workspace
        & (relative_height >= MIN_HEIGHT_ABOVE_TABLE_M)
        & (relative_height <= MAX_HEIGHT_ABOVE_TABLE_M)
    )
    seeds = height_seeds(z_base[elevated])
    if not seeds:
        raise RuntimeError('No dense depth surface at least 250 mm above the supporting table')
    open_kernel = np.ones((MORPH_OPEN_KERNEL_PX, MORPH_OPEN_KERNEL_PX), dtype=np.uint8)
    close_kernel = np.ones((MORPH_CLOSE_KERNEL_PX, MORPH_CLOSE_KERNEL_PX), dtype=np.uint8)
    candidates: list[dict] = []
    rejections: list[str] = []
    for seed in seeds:
        seed_mask = (elevated & (np.abs(z_base - float(seed)) <= TOP_SEED_HALF_BAND_M)).astype(np.uint8) * 255
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(seed_mask, connectivity=8)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) < MIN_COMPONENT_DEPTH_POINTS:
                continue
            candidate, rejection = candidate_from_component(
                labels == label,
                color_bgr,
                x_base,
                y_base,
                z_base,
                robot["T_base_camera"],
                table_z,
            )
            if candidate is None:
                if rejection is not None and len(rejections) < 20:
                    rejections.append(rejection)
                continue
            center = candidate['center_base_m']
            duplicate = next(
                (
                    index for index, previous in enumerate(candidates)
                    if np.linalg.norm(center - previous["center_base_m"]) < 0.020
                ),
                None,
            )
            if duplicate is None:
                candidates.append(candidate)
            elif candidate['score'] > candidates[duplicate]['score']:
                candidates[duplicate] = candidate
    if not candidates:
        summary = '; '.join(rejections[:8]) or 'no connected component passed'
        raise RuntimeError('Elevated depth was present, but no complete horizontal top passed: ' + summary)
    candidates.sort(key=lambda item: item['score'], reverse=True)
    selected = candidates[0]
    diagnostics = {
        "table": table_diagnostics,
        "table_z_base_m": float(table_z),
        "elevated_depth_points": int(np.count_nonzero(elevated)),
        "height_seeds_base_z_m": [float(value) for value in seeds],
        "accepted_candidates": int(len(candidates)),
        "rejection_examples": rejections,
        "selection_rule": (
            "largest reliable horizontal elevated surface; use coherent RGB "
            "edge refinement only with perimeter coverage and centre agreement"
        ),
        "colour_used_for_target_identity": False,
        "generic_rgb_edge_used_for_boundary_refinement": bool(
            selected["edge_refinement"]["used"]
        ),
        "shape_model_used": False,
    }
    return (selected, diagnostics)


def json_candidate(candidate: dict) -> dict:
    return {
        "center_base_m": candidate["center_base_m"].astype(float).tolist(),
        "normal_base": candidate["normal_base"].astype(float).tolist(),
        "height_above_table_mm": float(
            candidate["height_above_table_m"] * 1000.0
        ),
        "table_z_base_m": float(candidate["table_z_base_m"]),
        "plane_rms_mm": float(candidate["plane_rms_m"] * 1000.0),
        "top_tilt_deg": float(candidate["tilt_deg"]),
        "detected_area_cm2": float(candidate["metric_area_m2"] * 10000.0),
        "span_lengths_mm": (
            candidate["span_lengths_m"] * 1000.0
        ).astype(float).tolist(),
        "valid_depth_points": int(candidate["valid_depth_points"]),
        "plane_inliers": int(candidate["plane_inliers"]),
        "center_method": candidate["center_method"],
        "detection_method": candidate["detection_method"],
        "depth_center_base_xy_m": (
            candidate["depth_center_base_xy_m"].astype(float).tolist()
        ),
        "inlier_mean_base_xy_m": (
            candidate["inlier_mean_base_xy_m"].astype(float).tolist()
        ),
        "depth_rgb_center_disagreement_mm": (
            float(candidate["depth_rgb_center_disagreement_m"] * 1000.0)
            if candidate["depth_rgb_center_disagreement_m"] is not None
            else None
        ),
        "refined_inlier_center_disagreement_mm": float(
            candidate["refined_inlier_center_disagreement_m"] * 1000.0
        ),
        "depth_inlier_center_disagreement_mm": float(
            candidate["depth_inlier_center_disagreement_m"] * 1000.0
        ),
        "edge_refinement": candidate["edge_refinement"],
    }


def save_diagnostic(color: np.ndarray, candidate: dict, diagnostics: dict) -> None:
    view = color.copy()
    tint = np.zeros_like(view)
    tint[:, :, 0] = candidate['mask'].astype(np.uint8) * 180
    view = cv2.addWeighted(view, 1.0, tint, 0.45, 0.0)
    cv2.drawContours(view, [candidate['depth_contour_uv']], -1, (255, 255, 0), 1)
    cv2.drawContours(view, [candidate['refined_contour_uv']], -1, (0, 255, 255), 3)
    u, v = np.rint(candidate['center_uv']).astype(int)
    cv2.drawMarker(view, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
    center_mm = candidate['center_base_m'] * 1000.0
    cv2.rectangle(view, (0, 0), (view.shape[1], 78), (0, 0, 0), -1)
    line1 = f'DEPTH TOP CENTER Base [{center_mm[0]:.1f}, {center_mm[1]:.1f}, {center_mm[2]:.1f}] mm'
    edge_used = bool(candidate['edge_refinement']['used'])
    centre_check_m = candidate['depth_rgb_center_disagreement_m']
    centre_check_text = f'{centre_check_m * 1000.0:.2f} mm' if centre_check_m is not None else 'n/a'
    line2 = (
        f"edge {'USED' if edge_used else 'DEPTH FALLBACK'} "
        f"{candidate['edge_refinement']['support_fraction']*100.0:.0f}% | "
        f"depth/RGB {centre_check_text} | tilt {candidate['tilt_deg']:.2f} deg"
    )
    cv2.putText(view, line1, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(view, line2, (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
    vision.save_png(OUTPUT_IMAGE, view)


def main() -> int:
    print(VERSION, flush=True)
    print('READ ONLY: no robot motion and no gripper command.', flush=True)
    print(
        "Depth height and horizontal-plane geometry identify the target; "
        "a generic RGB intensity edge only refines its boundary. No colour "
        "class or known shape is used.",
        flush=True,
    )
    print('Keep the complete empty top and a large part of the supporting table visible.', flush=True)
    receiver = None
    try:
        receiver = PersistentRTDEReceive(
            vision.ROBOT_IP,
            frequency_hz=125.0,
            connect_attempts=4,
            freshness_timeout_s=0.60,
        )
        color, depth, capture, robot = vision.capture_at_current_pose('workbench_depth_top', receiver)
        selected, diagnostics = extract_workbench_placement_point(color, depth, robot)
        save_diagnostic(color, selected, diagnostics)
        candidate_json = json_candidate(selected)
        result = {
            "schema": "ur5e_workbench_depth_top_center_v3",
            "version": VERSION,
            "created_at_utc": utc_now(),
            "status": "calibrated_without_robot_motion",
            "robot_commands_sent": False,
            "gripper_commands_sent": False,
            "center_base_m": candidate_json["center_base_m"],
            "normal_base": candidate_json["normal_base"],
            "refined_top": candidate_json,
            "estimated_table_z_base_m": diagnostics["table_z_base_m"],
            "estimated_height_above_table_mm": candidate_json[
                "height_above_table_mm"
            ],
            "detection_constraints": {
                "minimum_height_above_table_mm": (
                    MIN_HEIGHT_ABOVE_TABLE_M * 1000.0
                ),
                "maximum_height_above_table_mm": (
                    MAX_HEIGHT_ABOVE_TABLE_M * 1000.0
                ),
                "maximum_top_tilt_deg": MAX_TOP_TILT_DEG,
                "colour_used_for_target_identity": False,
                "generic_rgb_edge_refinement_is_optional": True,
                "generic_rgb_edge_used_for_this_result": bool(
                    selected["edge_refinement"]["used"]
                ),
                "known_shape_used": False,
                "known_top_size_used": False,
                "maximum_depth_rgb_center_disagreement_mm": (
                    MAX_DEPTH_RGB_CENTER_DISAGREEMENT_M * 1000.0
                ),
                "maximum_depth_inlier_fallback_disagreement_mm": (
                    MAX_DEPTH_INLIER_CENTER_DISAGREEMENT_M * 1000.0
                ),
            },
            "diagnostics": diagnostics,
            "camera": {
                "serial": vision.CAMERA_SERIAL,
                "K": vision.COLOR_K.tolist(),
                "D_opencv_k1_k2_p1_p2_k3_k4_k5_k6": vision.COLOR_D.tolist(),
                "capture": capture,
            },
            "robot_at_capture": vision.json_ready_snapshot(robot),
            "output_image": OUTPUT_IMAGE.name,
        }
        write_json(OUTPUT_JSON, result)
        center_mm = (selected['center_base_m'] * 1000.0).round(2).tolist()
        print(f'WORKBENCH_TOP_OK: placement Base {center_mm} mm', flush=True)
        print(
            f"Center method: {selected['center_method']}; "
            f"RGB edge used={selected['edge_refinement']['used']}",
            flush=True,
        )
        if not selected['edge_refinement']['used']:
            print(
                "RGB edge refinement was rejected safely; depth-boundary "
                "centroid used. Reason: "
                f"{selected['edge_refinement']['fallback_reason']}",
                flush=True,
            )
        print(
            f"Supporting table Z {diagnostics['table_z_base_m']*1000.0:.1f} mm; "
            f"top height {selected['height_above_table_m']*1000.0:.1f} mm.",
            flush=True,
        )
        print(f'Saved: {OUTPUT_JSON}', flush=True)
        print(f'Saved: {OUTPUT_IMAGE}', flush=True)
        return 0
    except Exception as error:
        print(f'STOPPED: {error}', file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    finally:
        if receiver is not None:
            receiver.close()


if __name__ == '__main__':
    sys.exit(main())
