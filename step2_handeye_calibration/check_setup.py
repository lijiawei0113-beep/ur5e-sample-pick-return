import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
ROBOT_IP = '192.168.1.102'
WIDTH, HEIGHT, FPS = (848, 530, 15)
PATTERN = (6, 8)
SQUARE_M = 0.0255


def load_intrinsics():
    params = json.loads((ROOT / 'gemini305_color_intrinsics.json').read_text(encoding='utf-8-sig'))
    expected = {
        "width": WIDTH, "height": HEIGHT, "fps": FPS,
        "format": "OBFormat.YUYV", "sensor_type": "COLOR_SENSOR",
        "distortion_model": "OBCameraDistortionModel.BROWN_CONRADY_K6",
    }
    if any((params.get(key) != value for key, value in expected.items())):
        raise ValueError('内参文件与彩色流配置或畸变模型不一致，请重新读取内参')
    K = np.array(params['K'], dtype=np.float64)
    coefficients = params['distortion_coefficients']
    D = np.array([coefficients[k] for k in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')], dtype=np.float64)
    return (params, K, D)


def read_intrinsics():
    from pyorbbecsdk import Pipeline, OBSensorType, OBFormat
    pipeline = Pipeline()
    info = pipeline.get_device().get_device_info()
    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    profile = profiles.get_video_stream_profile(WIDTH, HEIGHT, OBFormat.YUYV, FPS)
    intrinsic = profile.get_intrinsic()
    distortion = profile.get_distortion()
    fx = float(intrinsic.fx)
    fy = float(intrinsic.fy)
    cx = float(intrinsic.cx)
    cy = float(intrinsic.cy)
    K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    coefficients = {
        name: float(getattr(distortion, name))
        for name in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
    }
    result = {
        "source": "Orbbec SDK factory calibration",
        "camera_name": info.get_name(),
        "serial_number": info.get_serial_number(),
        "sensor_type": "COLOR_SENSOR",
        "width": profile.get_width(),
        "height": profile.get_height(),
        "format": str(profile.get_format()),
        "fps": profile.get_fps(),
        "intrinsics_unit": "pixel",
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": K,
        "distortion_model": str(distortion.model),
        "distortion_coefficients": coefficients,
    }
    print('相机：', result['camera_name'])
    print(f'配置：{WIDTH} × {HEIGHT} / YUYV / {FPS} FPS')
    print('\n内参矩阵 K：')
    for row in K:
        print('  ', row)
    print('\n畸变模型：', result['distortion_model'])
    print('畸变系数：')
    for name, value in coefficients.items():
        print(f'  {name} = {value}')
    output = Path(__file__).resolve().parent / 'gemini305_color_intrinsics.json'
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print('\n已保存：', output)


def check_board_pose():
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    params, K, D = load_intrinsics()
    obj = np.zeros((PATTERN[0] * PATTERN[1], 3), dtype=np.float64)
    obj[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_M
    pipeline = Pipeline()
    serial = pipeline.get_device().get_device_info().get_serial_number()
    if serial != params['serial_number']:
        raise ValueError('当前相机序列号与内参文件不一致')
    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    profile = profiles.get_video_stream_profile(WIDTH, HEIGHT, OBFormat.YUYV, FPS)
    config = Config()
    config.enable_stream(profile)
    pipeline.start(config)
    try:
        last_frame = time.monotonic()
        print('将整块棋盘格放入画面；检测到48个角点并求得位姿后自动保存一组并退出。')
        print('保存目录：', ROOT / 'board_check')
        while True:
            frames = pipeline.wait_for_frames(1000)
            frame = frames.get_color_frame() if frames is not None else None
            if frame is None:
                if time.monotonic() - last_frame > 10:
                    raise RuntimeError('超过10秒未收到彩色帧，请检查连接及Viewer是否关闭')
                continue
            last_frame = time.monotonic()
            if (frame.get_width(), frame.get_height(), frame.get_format()) != (WIDTH, HEIGHT, OBFormat.YUYV):
                raise RuntimeError('收到的彩色帧与指定配置不一致')
            raw = np.frombuffer(frame.get_data(), dtype=np.uint8).reshape(HEIGHT, WIDTH, 2)
            bgr = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(gray, PATTERN, cv2.CALIB_CB_NORMALIZE_IMAGE)
            view = bgr.copy()
            text, color = ('Board NOT found', (0, 0, 255))
            pose = None
            if found:
                cv2.drawChessboardCorners(view, PATTERN, corners, True)
                text, color = ('48 corners found; pose unavailable', (0, 165, 255))
                ok, rvec, tvec = cv2.solvePnP(obj, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok and np.isfinite(rvec).all() and np.isfinite(tvec).all():
                    R, _ = cv2.Rodrigues(rvec)
                    if np.all((obj @ R.T + tvec.reshape(1, 3))[:, 2] > 0):
                        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
                        residual = projected.reshape(-1, 2) - corners.reshape(-1, 2)
                        rmse = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
                        pose = np.eye(4)
                        pose[:3, :3], pose[:3, 3] = (R, tvec.ravel())
                        cv2.drawFrameAxes(view, K, D, rvec, tvec, SQUARE_M * 2, 2)
                        xy = np.asarray(corners).reshape(-1, 2)[0]
                        origin = (int(round(float(xy[0]))), int(round(float(xy[1]))))
                        cv2.putText(view, 'O', origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        text, color = (f'48 corners | reprojection RMSE: {rmse:.3f} px', (0, 255, 0))
                        xyz = tvec.ravel() * 1000
                        cv2.putText(view, f"Origin in camera [mm]: {xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}",
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
            cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(
                view, 'AUTO SAVE on valid pose | Q/Esc: quit', (10, HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1,
            )
            cv2.imshow('Gemini 305 - board check', view)
            key = cv2.waitKey(1) & 255
            if key in (ord('q'), ord('Q'), 27):
                break
            if pose is not None:
                folder = ROOT / 'board_check'
                folder.mkdir(exist_ok=True)
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                for name, image in (('raw', bgr), ('check', view)):
                    encoded_ok, data = cv2.imencode('.png', image)
                    if not encoded_ok:
                        raise RuntimeError('PNG编码失败')
                    output = folder / f'{stamp}_{name}.png'
                    data.tofile(str(output))
                    print('已保存：', output)
                print(text)
                print('T_camera_board（标定板 -> 相机，平移单位m；仅本帧预览）：\n', pose)
                break
    finally:
        try:
            pipeline.stop()
        finally:
            cv2.destroyAllWindows()


def read_robot_pose():
    from rtde_receive import RTDEReceiveInterface
    print(f'连接 UR5e：{ROBOT_IP}，读取实际 TCP 位姿……')
    robot = RTDEReceiveInterface(ROBOT_IP, 125.0, ['timestamp', 'actual_TCP_pose'])
    try:
        if not robot.isConnected():
            raise RuntimeError('RTDE 未连接成功')
        initial_time = robot.getTimestamp()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not robot.isConnected():
                raise RuntimeError('读取过程中 RTDE 连接断开')
            stamp_before = robot.getTimestamp()
            pose = np.asarray(robot.getActualTCPPose(), dtype=np.float64)
            stamp_after = robot.getTimestamp()
            if stamp_before > initial_time and stamp_before == stamp_after:
                if pose.shape == (6,) and np.isfinite(pose).all():
                    break
            time.sleep(0.01)
        else:
            raise RuntimeError('5秒内未读到有效的新 TCP 位姿')
        R, _ = cv2.Rodrigues(pose[3:].reshape(3, 1))
        T_base_tcp = np.eye(4)
        T_base_tcp[:3, :3] = R
        T_base_tcp[:3, 3] = pose[:3]
        print('\n实际 TCP 位姿 [x, y, z, rx, ry, rz]：')
        print(pose.tolist())
        print('\nTCP 位置，基座坐标系，单位 mm：')
        for label, value in zip(('X', 'Y', 'Z'), pose[:3] * 1000):
            print(f'{label} = {value:.3f}')
        print('\nTCP 旋转向量，单位 rad：', pose[3:].tolist())
        print('\nT_base_tcp（TCP -> 基座，平移单位 m）：')
        print(T_base_tcp)
        record = {
            "purpose": "robot_pose_connection_check_only",
            "robot_ip": ROBOT_IP,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "controller_time_s": float(stamp_after),
            "source": "RTDE actual_TCP_pose",
            "reference_frame": "robot_base",
            "tool_frame": "active_TCP",
            "position_unit": "m",
            "orientation_representation": "rotation_vector_rad",
            "actual_tcp_pose": pose.tolist(),
            "T_base_tcp": T_base_tcp.tolist(),
            "paired_camera_image": None,
        }
        folder = Path(__file__).resolve().parent / 'robot_check'
        folder.mkdir(exist_ok=True)
        filename = datetime.now().strftime('ur5e_pose_%Y%m%d_%H%M%S_%f.json')
        output = folder / filename
        output.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print('\n已自动保存：', output)
    finally:
        robot.disconnect()


def main():
    actions = {
        "1": ("读取并保存相机内参", read_intrinsics),
        "2": ("预览棋盘位姿并保存检查图", check_board_pose),
        "3": ("读取并保存机器人 TCP 位姿", read_robot_pose),
    }
    for key, (label, _) in actions.items():
        print(f'{key}. {label}')
    choice = input('请选择 1 / 2 / 3：').strip()
    if choice not in actions:
        raise ValueError('请选择 1、2 或 3')
    actions[choice][1]()


if __name__ == '__main__':
    main()
