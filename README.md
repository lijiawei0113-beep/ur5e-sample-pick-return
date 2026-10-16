# Moving Workbench Follow

UR5e + Gemini 305: pick a sample, follow the green workbench in real time,
hold the sample at the measured centre, and return it to its original tray point.
The gripper remains closed at the workbench. Every detected tray sample is
processed once, then the robot returns to its starting joint pose.

## Files

| File | Purpose |
| --- | --- |
| `calibrate_green_workbench.py` | Locate the green top, take a fresh refined measurement, and save its centre and height |
| `pick_follow_return.py` | Tray detection, grasp, live XY follow, workbench dwell, and tray return |

Keep `step4_sample_pick_and_return` and `step5_moving_workbench_follow` beside
each other. Step 5 imports `pick_place_return.py`, `vision.py`, and `robot.py`
from Step 4. The archive includes the unchanged Step 4 folder for convenience.
Old detector files ending in `2.py` are not needed.

## Run

Use the existing Python environment with NumPy, OpenCV, `ur-rtde`, and the
matching Orbbec Python SDK. Close Orbbec Viewer. Robot/tool settings are in
Step 4's `robot.py`; camera parameters and hand-eye calibration are in
Step 4's `vision.py`. Keep the existing robot-side RG2 command program and
OnRobot installation configured as described in the Step 4 README.

1. Keep the empty green top and part of its supporting table visible. Run
   `calibrate_green_workbench.py` directly, or from this folder:

   ```powershell
   python calibrate_green_workbench.py
   ```

   The calibration expects an approximately 30 × 30 mm green top, 240–290 mm
   above the supporting table. It may reposition the camera for a perpendicular
   view and retrace its path afterwards. If the initial view is sufficient,
   it takes the fresh refinement without moving. Inspect
   `workbench_calibration.png` and the reported outcome.

2. Prepare the tray and starting robot pose, then run `pick_follow_return.py`
   directly:

   ```powershell
   python pick_follow_return.py
   ```

   No command-line arguments are required. The preview retains the enlarged
   camera image, target marker, XY error, follow state, and stability timer.
   Esc stops the follow stage.

## Retained behaviour

- Wide capture: 2 warmup + 7 measured frames; centred capture: 3 + 9.
- Live tracking projects the green outline onto the calibrated horizontal plane
  and uses its area centroid. Top Z remains fixed while workbench XY may move.
- During follow, TCP height and orientation stay fixed. The 60 mm lateral
  offset keeps the held sample away from the camera's view of the green top.
- A short stop window holds XY speed first. The following 3-second stability
  window determines the final centre before the placement motion.
- Gripper order: open, close at the tray, stay closed at the workbench,
  open after returning to the tray.
- Camera recovery, RTDE Receive recovery, the single RTDE Control owner,
  stale-command stopping, motion parameters, and existing stop conditions remain.

Calibration uses `GREEN_HSV_LOWER/UPPER`; tracking uses `GREEN_LOWER/UPPER`.
Their original ranges remain separate. Adjust the relevant entry's parameters
if the green top is not detected under the actual lighting.

## Outputs and checks

Coordinates and generated images/logs stay in this Step 5 folder:
`workbench_pose.json`, `workbench_calibration.png`,
`green_workbench_follow_run.json`, and `green_workbench_follow_latest.png`.
Step 4's saved workbench coordinates are not used by the Step 5 entry.

The cleanup removes the old dynamic module loader, three unused geometry
functions, unused thresholds, and verbose comments. Shared Step 4 code is
unchanged. Retained function bodies were compared after the explicit import,
Receive-callback, and output-path changes. Offline checks covered geometry,
view poses, capture-setting restoration, stop decisions, mocked follow/resume
and connection recovery, and RTDE worker failure handling.
Physical robot motion, real camera processing, and the actual GUI have not
been tested in this environment.
