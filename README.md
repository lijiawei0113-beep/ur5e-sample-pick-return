# UR5e Vision-Guided Sample Pick and Return

A vision-guided sample handling project using a **UR5e robotic arm**, an **Orbbec Gemini 305 RGB-D camera**, and an **OnRobot RG2 gripper**.

The system detects cylindrical samples in a tray, refines their centre positions from a centred camera view, transfers them to a workbench, and returns them to their original tray positions. Custom 3D-printed gripper fingertips improve sample centring during grasping.

## Preview

![UR5e workflow preview](ur5e_pick_place_v3.gif)

## Hardware

- Universal Robots UR5e robotic arm
- Orbbec Gemini 305 RGB-D camera
- OnRobot RG2 gripper
- 3D-printed verification probe and custom gripper fingertips
- Calibration chessboard, sample tray, and workbench

## Repository Structure

| Directory | Description |
| --- | --- |
| [Step 1 — Environment Setup and Warmup](step1_environment_setup_and_warmup/) | Keyboard control and initial robot setup. |
| [Step 2 — Hand–Eye Calibration](step2_handeye_calibration/) | Camera and robot checks, data collection, calibration, and validation. |
| [Step 3 — Hand–Eye Verification](step3_handeye_verification/) | Plane alignment, target selection, and point-touch testing. |
| [Step 4 — Sample Pick and Return](step4_sample_pick_and_return/) | Sample detection, workbench calibration, and pick-and-return cycles. |
| [Step 5 — Moving Workbench Follow](step5_moving_workbench_follow/) | Green workbench calibration and live XY following. |

The instructions below cover Steps 1–3.

## Software Environment

The project was developed on Windows. Use the Python environment tested with your Orbbec SDK build.

```powershell
python -m pip install numpy opencv-python ur-rtde pynput
```

Install the compatible Orbbec Python SDK separately. It must provide `pyorbbecsdk`.
Close Orbbec Viewer before running camera scripts.

Robot IP, camera serial number, calibration parameters, and tool geometry must match the actual setup. Run each stage separately, with robot motion supervised from the teach pendant.

## Step 1 — Environment Setup and Warmup

This script controls individual robot joints and triggers the existing RG2 programs.
The keys command joint rotation, rather than Cartesian X/Y/Z translation.

### Configuration and Run

Check these values in `step1_environment_setup_and_warmup/keyboard_control.py`:

| Setting | Included value |
| --- | --- |
| `ROBOT_IP` | `192.168.1.102` |
| `RG2_OPEN_PROGRAM` | `/programs/rg2_open_Li.urp` |
| `RG2_CLOSE_PROGRAM` | `/programs/rg2_close_Li.urp` |
| `JOINT_SPEED` | `0.04 rad/s` |

The two RG2 programs must already exist on the robot controller. The close action depends on the settings saved in that robot-side program.

From the repository root:

```powershell
python step1_environment_setup_and_warmup/keyboard_control.py
```

### Keyboard Controls

Hold **Space** and one joint key to move. Positive and negative directions refer to the robot's joint-angle convention.

| Joint | Positive rotation | Negative rotation |
| --- | --- | --- |
| J1 — Base | Space + Q | Space + A |
| J2 — Shoulder | Space + W | Space + S |
| J3 — Elbow | Space + E | Space + D |
| J4 — Wrist 1 | Space + R | Space + F |
| J5 — Wrist 2 | Space + T | Space + G |
| J6 — Wrist 3 | Space + Y | Space + H |

| Key or action | Behaviour |
| --- | --- |
| Space + O | Open the RG2 gripper. |
| Space + C | Run the configured RG2 close/grasp program. |
| Release a joint key | Stop that joint command. |
| Release Space | Clear all joint commands and command zero joint velocity. |
| Esc | Request a stop and exit the program. |

Only one joint key is accepted at a time. Holding multiple joint keys commands zero joint velocity.
Joint motion is disabled while an RG2 action is running. After the gripper action finishes and the RTDE control session is re-established, release Space and hold it again to resume joint control.

## Step 2 — Hand–Eye Calibration

The camera is mounted on the robot, while the chessboard remains fixed during calibration data collection.

### Calibration Board

<img src="assets/calibration_board.png" alt="Calibration chessboard with 6 by 8 inner corners" width="360">

| Parameter | Configuration |
| --- | --- |
| Board layout | 7 × 9 squares |
| Inner corners | 6 × 8, or 48 corners |
| `PATTERN` | `(6, 8)` |
| Square size used by the scripts | `25.5 mm` |
| `SQUARE_M` | `0.0255` |

The supplied PDF is a large-format source with approximately 108 mm squares at its original document size. Scale it when printing to obtain the required physical square size. A scale of approximately 23.61% corresponds to 25.5 mm squares; measure the printed result with a ruler.

Use the **measured printed square size** in both Step 2 and Step 3. The on-screen preview does not specify physical print dimensions. Mount the board on a flat, rigid support.

### Setup Checks

Set `ROBOT_IP`, `PATTERN`, and `SQUARE_M` in `step2_handeye_calibration/check_setup.py`, then run:

```powershell
python step2_handeye_calibration/check_setup.py
```

Run the checker again for each required option:

| Option | Purpose |
| --- | --- |
| 1 | Read camera intrinsics and save `gemini305_color_intrinsics.json`. |
| 2 | Preview the detected chessboard pose. |
| 3 | Check robot communication and read the actual TCP pose. |

### Collect Calibration Pairs

```powershell
python step2_handeye_calibration/collect_handeye_pair.py
```

Each execution saves one robot/camera pair in `step2_handeye_calibration/handeye_data/`.
Change the robot position and orientation between captures, then wait for it to stop. Keep the board, camera mounting, and active TCP configuration unchanged throughout collection. Select the same physical chessboard origin corner for each saved pair.

The current workflow uses 30 pairs:

- Samples 1–25: calibration.
- Samples 26–30: held-out validation.

### Solve and Validate

```powershell
python step2_handeye_calibration/solve_handeye.py
python step2_handeye_calibration/validate_handeye.py
```

| Output | Meaning |
| --- | --- |
| `T_tcp_camera.txt` | Transformation from camera coordinates into the active TCP frame. |
| `validation_result.txt` | Board-pose consistency results for the held-out pairs. |

The solver uses OpenCV's Park hand–eye method. Validation reports position and rotation consistency; it does not independently measure absolute robot positioning accuracy.

Calibration results are not automatically loaded by the touch-test script. Update its `H` matrix, camera intrinsics, distortion coefficients, and serial number to match the calibration used.

Step 3 expects a flange-referenced hand–eye transformation and zero active TCP. If Step 2 is collected with zero active TCP, `T_tcp_camera` is also `T_flange_camera`. Otherwise, convert it before use:

```text
T_flange_camera = T_flange_tcp × T_tcp_camera
```

Here, `T_A_B` maps coordinates from frame B into frame A; translation is in metres.

## Step 3 — Hand–Eye Verification

This stage checks the combined camera, hand–eye, robot, and probe geometry by moving a mounted probe to a selected chessboard corner.

### Verification Probe

<img src="assets/verification_probe.png" alt="Slicer preview of the supplied verification probe part" width="420">

The attachment is a sliced Bambu Studio printing file. It contains the print instructions and a preview, rather than an editable STL mesh. The preview shows the supplied printed part; it does not show the complete mounted tool assembly.

### Probe Tip Offset and TCP

The current script uses the following nominal geometry:

| Parameter | Value |
| --- | --- |
| Controller active TCP during this test | Zero: `[0, 0, 0, 0, 0, 0]` |
| Probe tip relative to the zero-TCP/flange frame | `[0, 0, 245.6] mm` |
| Corresponding position vector in the script | `[0.0, 0.0, 0.2456] m` |

The **245.6 mm** value is the nominal distance along the local tool +Z axis from the zero-TCP origin to the physical tip after mounting. It is not the printed part's length alone and cannot be established from the printing file alone. Verify it on the installed assembly.

The script accounts for this offset when calculating the commanded pose:

```text
tip position in Base = TCP position in Base + TCP rotation × tip offset
commanded TCP position = target position in Base − TCP rotation × tip offset
```

Keep the controller's active TCP at zero for this script: the probe-tip offset is applied in the position calculation. Do not also enter 245.6 mm as the controller's active TCP offset.

The default probe length is set by the `--tip-z-mm` argument in `touch_target.py`. Running without arguments uses `245.6 mm`. Match it to the measured installed geometry before testing.

### Run

Check `IP`, `SERIAL`, `K`, `D`, `H`, the chessboard dimensions, and the probe geometry in `step3_handeye_verification/touch_target.py`.

With the probe installed and the board visible, run:

```powershell
python step3_handeye_verification/touch_target.py
```

The sequence is:

1. Capture the chessboard and estimate its plane.
2. Reposition the robot to obtain a perpendicular camera view.
3. Capture a new image and select the intended chessboard corner.
4. Complete the selection in the window; the script verifies the scene and moves the probe to the calculated target pose.
5. Stop at the target without an automatic withdrawal.

Session images, target information, motion plans, and execution records are saved under `step3_handeye_verification/handeye_touch_runs/`.

Inspect and record the tip-to-corner discrepancy. Reaching the commanded pose alone does not quantify the positioning error, and chessboard reprojection RMSE is not a hand–eye positioning error.
