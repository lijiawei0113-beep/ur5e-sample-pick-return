import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'handeye_data'
# Samples 26-30 are reserved for validation.
TRAINING_SAMPLE_COUNT = 25


def solve(data_dir):
    if not hasattr(cv2, 'calibrateHandEye'):
        raise RuntimeError('This OpenCV build does not provide calibrateHandEye().')
    files = [data_dir / f'sample_{i:04d}' / 'pair.json' for i in range(1, TRAINING_SAMPLE_COUNT + 1)]
    files = [file for file in files if file.is_file()]
    if len(files) < 3:
        raise RuntimeError(f'Need at least 3 distinct poses. Data folder: {data_dir}')
    robot, camera = ([], [])
    for file in files:
        d = json.loads(file.read_text(encoding='utf-8-sig'))
        if d['position_unit'] != 'm':
            raise ValueError(f'Expected translation in meters: {file}')
        robot.append(np.array(d['T_base_tcp'], dtype=np.float64))
        camera.append(np.array(d['T_camera_board'], dtype=np.float64))
    R, t = cv2.calibrateHandEye(
        [T[:3, :3] for T in robot],
        [T[:3, 3:4] for T in robot],
        [T[:3, :3] for T in camera],
        [T[:3, 3:4] for T in camera],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = (R, t.ravel())
    if not np.isfinite(T).all() or not np.isclose(np.linalg.det(R), 1.0):
        raise RuntimeError('Invalid solution. Check pose diversity and input data.')
    print(f'Samples: {len(files)}')
    return T


if __name__ == '__main__':
    T_tcp_camera = solve(DATA)
    np.set_printoptions(precision=9, suppress=True)
    print('T_tcp_camera: color camera -> active TCP; translation in meters')
    print(T_tcp_camera)
    output = ROOT / 'T_tcp_camera.txt'
    np.savetxt(output, T_tcp_camera, fmt='%.12f')
    print(f'Saved: {output}')
