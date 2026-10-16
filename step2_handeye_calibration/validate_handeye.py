import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def validate(root):
    X = np.loadtxt(root / 'T_tcp_camera.txt')
    if X.shape != (4, 4) or not np.isfinite(X).all():
        raise ValueError('Invalid T_tcp_camera.txt')

    def board_pose(index):
        file = root / 'handeye_data' / f'sample_{index:04d}' / 'pair.json'
        d = json.loads(file.read_text(encoding='utf-8-sig'))
        if d['position_unit'] != 'm':
            raise ValueError(f'Expected meters: {file}')
        G = np.array(d['T_base_tcp'], dtype=float)
        C = np.array(d['T_camera_board'], dtype=float)
        return G @ X @ C
    training = np.array([board_pose(i) for i in range(1, 26)])
    reference = np.eye(4)
    reference[:3, 3] = training[:, :3, 3].mean(axis=0)
    U, _, Vt = np.linalg.svd(training[:, :3, :3].mean(axis=0))
    reference[:3, :3] = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    errors = []
    lines = ['Sample      Position_mm    Rotation_deg']
    for i in range(26, 31):
        observed = board_pose(i)
        position = np.linalg.norm(observed[:3, 3] - reference[:3, 3]) * 1000
        R = reference[:3, :3].T @ observed[:3, :3]
        rotation = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        errors.append([position, rotation])
        lines.append(f'{i:6d}      {position:11.6f}    {rotation:12.6f}')
    errors = np.array(errors)
    if not np.isfinite(errors).all():
        raise ValueError('Non-finite validation errors. Check input matrices.')
    rms = np.sqrt(np.mean(errors ** 2, axis=0))
    maximum = errors.max(axis=0)
    lines += [
        f'RMS         {rms[0]:11.6f}    {rms[1]:12.6f}',
        f'Max         {maximum[0]:11.6f}    {maximum[1]:12.6f}',
        'Consistency errors; absolute positioning accuracy is not measured.',
    ]
    text = '\n'.join(lines)
    output = root / 'validation_result.txt'
    output.write_text(text + '\n', encoding='utf-8')
    print(text)
    print(f'Saved: {output}')
    return errors


if __name__ == '__main__':
    validate(ROOT)
