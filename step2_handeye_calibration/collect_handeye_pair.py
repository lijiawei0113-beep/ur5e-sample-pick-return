import json
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np

from check_setup import ROOT, ROBOT_IP, WIDTH, HEIGHT, FPS, PATTERN, SQUARE_M, load_intrinsics


_params, K, D = load_intrinsics()
SERIAL = _params['serial_number']
OUTPUT = ROOT / 'handeye_data'
WINDOW = 'UR5e + Gemini305 | hand-eye pair'
OBJ = np.zeros((48, 3), dtype=np.float64)
OBJ[:, :2] = np.mgrid[0:6, 0:8].T.reshape(-1, 2) * SQUARE_M
CAMERA = {"serial_number": SERIAL, "width": WIDTH, "height": HEIGHT,
          "fps": FPS, "format": "YUYV", "K": K.tolist(),
          "D_opencv": D.tolist(), "D_order": "k1,k2,p1,p2,k3,k4,k5,k6",
          "source": "user-exported Orbbec factory calibration; raw color"}
BOARD = {"pattern_columns_rows": list(PATTERN), "square_size_m": SQUARE_M,
         "origin": "same physical extreme inner corner, manually selected",
         "x_axis": "along the 6-corner edge", "y_axis": "along the 8-corner edge"}
STILL_SECONDS = 2.0
MAX_DRIFT_M = 0.00025
MAX_DRIFT_RAD = np.deg2rad(0.05)
MAX_LINEAR_SPEED = 0.0005
MAX_ANGULAR_SPEED = 0.003
MAX_RMSE = 0.8


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
                if now - last_new > 1.0:
                    raise RuntimeError('超过1秒未收到有效的新机器人数据')
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


def load_previous():
    samples = []
    for path in sorted(OUTPUT.glob('sample_*/pair.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema') != 'ur5e_gemini305_pair_v1' or data.get('camera') != CAMERA or data.get('board') != BOARD:
            raise ValueError(f'已有数据配置不同，不能混用：{path}')
        T = np.asarray(data['T_base_tcp'], dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise ValueError(f'已有位姿损坏：{path}')
        samples.append(T)
    return samples


def is_duplicate(T, previous):
    return any(d < 0.010 and a < np.deg2rad(4.0)
               for d, a in (pose_difference(T, old) for old in previous))


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


def order_from_origin(corners, row, column):
    grid = np.asarray(corners, dtype=np.float64).reshape(PATTERN[1], PATTERN[0], 2)
    if row not in (0, PATTERN[1] - 1) or column not in (0, PATTERN[0] - 1):
        raise ValueError('原点必须是内角点矩形的四个顶点之一')
    if row != 0:
        grid = grid[::-1, :, :]
    if column != 0:
        grid = grid[:, ::-1, :]
    return np.ascontiguousarray(grid.reshape(-1, 1, 2))


def choose_origin(candidate):
    grid = candidate['corners'].reshape(PATTERN[1], PATTERN[0], 2)
    choices = [(0, 0), (0, PATTERN[0] - 1), (PATTERN[1] - 1, 0), (PATTERN[1] - 1, PATTERN[0] - 1)]
    points = np.array([grid[r, c] for r, c in choices])
    selected = []

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            distances = np.linalg.norm(points - [x, y], axis=1)
            index = int(np.argmin(distances))
            if distances[index] <= 18:
                selected[:] = [choices[index], [x, y]]
    view = candidate['bgr'].copy()
    cv2.drawChessboardCorners(view, PATTERN, candidate['corners'], True)
    for point in points:
        cv2.circle(view, tuple(np.rint(point).astype(int).tolist()), 12, (0, 255, 255), 2)
    print('\n画面已经冻结，图像和机器人位姿已配对。')
    print('请点击你选定的那个实际内角点（四个黄色圆圈之一）。')
    print('每组必须点击棋盘上同一个物理角点；不要固定点击画面的左下角！')
    print('点击后自动保存，不用按 S；点错前可 Q/Esc 退出重采。')
    cv2.setMouseCallback(WINDOW, mouse)
    while not selected:
        if show(view, 'FROZEN: click the SAME physical origin corner'):
            return None
        time.sleep(0.015)
    cv2.setMouseCallback(WINDOW, lambda *args: None)
    (row, column), click = selected
    return (order_from_origin(candidate['corners'], row, column), [row, column], click)


def board_pose(corners):
    ok, rvec, tvec = cv2.solvePnP(OBJ, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.isfinite(rvec).all() or (not np.isfinite(tvec).all()):
        raise RuntimeError('棋盘位姿求解失败，本组未保存')
    T = pose_matrix(np.r_[tvec.ravel(), rvec.ravel()])
    if np.any((OBJ @ T[:3, :3].T + T[:3, 3])[:, 2] <= 0):
        raise RuntimeError('棋盘位姿无效，本组未保存')
    projected = cv2.projectPoints(OBJ, rvec, tvec, K, D)[0].reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((projected - corners.reshape(-1, 2)) ** 2, axis=1))))
    if rmse > MAX_RMSE:
        raise RuntimeError(f'重投影 RMSE={rmse:.3f}px > {MAX_RMSE}px，本组未保存；检查清晰度、反光和平整度后重采')
    return (T, rvec, tvec, rmse)


def state_json(s):
    return {"host_monotonic_s": s["host_s"], "controller_time_s": s["controller_s"],
            "actual_tcp_pose_m_rad": s["pose"].tolist(),
            "actual_tcp_speed_m_s_rad_s": s["speed"].tolist()}


def save_pair(candidate, ordered, origin_rc, click):
    T_camera_board, rvec, tvec, rmse = board_pose(ordered)
    view = candidate['bgr'].copy()
    cv2.drawChessboardCorners(view, PATTERN, ordered.astype(np.float32), True)
    cv2.drawFrameAxes(view, K, D, rvec, tvec, 2 * SQUARE_M, 2)
    origin = tuple(np.rint(ordered.reshape(-1, 2)[0]).astype(int).tolist())
    cv2.putText(view, 'O', origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(view, f"48 corners | RMSE {rmse:.3f}px | paired with active TCP",
                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (0, 255, 0), 2)
    record = {"schema": "ur5e_gemini305_pair_v1", "mode": "eye_in_hand",
              "captured_at_utc": candidate["captured_at_utc"],
              "robot_ip": ROBOT_IP, "robot_reference_frame": "robot_base",
              "robot_tool_frame": "active_TCP", "tcp_offset_verified": False,
              "tcp_offset": None, "position_unit": "m",
              "orientation_representation": "rotation_vector_rad",
              "transform_convention": "T_A_B maps coordinates from B to A",
              "camera": CAMERA, "board": BOARD,
              "raw_image": "raw.png", "annotated_image": "check.png",
              "T_base_tcp": candidate["robot"]["T"].tolist(),
              "T_camera_board": T_camera_board.tolist(),
              "robot_at_image": state_json(candidate["robot"]),
              "corners_px": ordered.reshape(-1, 2).tolist(),
              "detected_corners_px": candidate["corners"].reshape(-1, 2).tolist(),
              "origin_rc_in_detected_grid": origin_rc, "origin_click_px": click,
              "reprojection_rmse_px": rmse,
              "pairing": {"method": "stationary_window_software_pairing",
                          "hardware_synchronized": False,
                          "image_host_receive_monotonic_s": candidate["image_received_host_s"],
                          "note": "Host receive times are not exposure times; keep robot and board stationary.",
                          "stability": candidate["stability"],
                          "robot_trace": [state_json(s) for s in candidate["states"]]}}
    OUTPUT.mkdir(exist_ok=True)
    indices = [int(p.name.split("_")[1]) for p in OUTPUT.glob("sample_*")
               if p.is_dir() and p.name.split("_")[1].isdigit()]
    folder = OUTPUT / f'sample_{max(indices, default=0) + 1:04d}'
    pending = OUTPUT / f'pending_{uuid.uuid4().hex}'
    pending.mkdir()
    for name, image in (('raw.png', candidate['bgr']), ('check.png', view)):
        ok, data = cv2.imencode('.png', image)
        if not ok:
            raise RuntimeError('PNG 编码失败，本组未提交')
        data.tofile(str(pending / name))
    record['sample_id'] = folder.name
    (pending / "pair.json").write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                                allow_nan=False), encoding="utf-8")
    pending.rename(folder)
    print(f'\n已自动保存：{folder}')
    print('文件：raw.png、check.png、pair.json')
    print(f'48 corners | reprojection RMSE: {rmse:.3f} px')
    print('T_base_tcp（当前 TCP -> 基座，平移 m）：\n', candidate['robot']['T'])
    print('T_camera_board（棋盘 -> 彩色相机，平移 m）：\n', T_camera_board)
    print('本次采集完成。改变机器人位置/倾斜姿态、停稳后，再次运行采集下一组。')
    return folder


def main():
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    from rtde_receive import RTDEReceiveInterface
    np.set_printoptions(precision=8, suppress=True)
    previous = load_previous()
    print('保存目录：', OUTPUT)
    print(f'已有 {len(previous)} 组；本次只采一组。请保持棋盘、相机安装和 TCP 配置不变。')
    print('程序只读机器人状态。请关闭 Viewer，将整块棋盘放入画面，机器人停稳。')
    receiver = monitor = pipeline = None
    started = False
    try:
        receiver = RTDEReceiveInterface(ROBOT_IP, 125.0,
                                       ["timestamp", "actual_TCP_pose", "actual_TCP_speed"])
        monitor = RobotMonitor(receiver)
        monitor.thread.start()
        pipeline = Pipeline()
        serial = pipeline.get_device().get_device_info().get_serial_number()
        if serial != SERIAL:
            raise RuntimeError(f'当前相机序列号 {serial} 与已知内参 {SERIAL} 不匹配')
        profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(
            WIDTH, HEIGHT, OBFormat.YUYV, FPS)
        config = Config()
        config.enable_stream(profile)
        pipeline.start(config)
        started = True
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        candidate = capture_candidate(pipeline, monitor, previous, OBFormat.YUYV)
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
    if candidate is None:
        print('已退出，本次未保存。')
        return
    selection = choose_origin(candidate)
    if selection is not None:
        save_pair(candidate, *selection)
    else:
        print('已退出，本次未保存。')


if __name__ == '__main__':
    try:
        main()
    finally:
        cv2.destroyAllWindows()
