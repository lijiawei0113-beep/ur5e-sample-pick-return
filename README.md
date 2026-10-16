# UR5e Vision-Guided Sample Pick and Return

A vision-guided sample handling project using a **UR5e robotic arm**, an **Orbbec Gemini 305 RGB-D camera**, and an **OnRobot RG2 gripper**.

The system detects cylindrical samples in a tray, refines their centre positions from a centred camera view, transfers them to a workbench, and returns them to their original tray positions. Custom 3D-printed gripper fingertips improve sample centring during grasping.

## Preview

![UR5e workflow preview](ur5e_pick_place_v3.gif)

## Hardware

- Universal Robots UR5e robotic arm
- Orbbec Gemini 305 RGB-D camera
- OnRobot RG2 gripper
- Custom 3D-printed gripper fingertips
- Sample tray and workbench

## Main Features

- Keyboard-based robot control for initial setup.
- Eye-in-hand calibration and point-touch verification.
- Coarse sample detection followed by centred RGB-D refinement.
- Workbench top-centre estimation.
- Automated sample picking, workbench dwell, and tray return.
- Camera stream and RTDE Receive connection recovery.
- Experimental real-time following of a green workbench.

## Repository Structure

| Directory | Description |
| --- | --- |
| [Step 1 — Environment Setup and Warmup](step1_environment_setup_and_warmup/) | Keyboard control and initial robot setup. |
| [Step 2 — Hand–Eye Calibration](step2_handeye_calibration/) | Setup checks, calibration data collection, transformation estimation, and validation. |
| [Step 3 — Hand–Eye Verification](step3_handeye_verification/) | Point-touch testing with a mounted probe. |
| [Step 4 — Sample Pick and Return](step4_sample_pick_and_return/) | Sample detection, workbench calibration, and automated pick-and-return cycles. |
| [Step 5 — Moving Workbench Follow](step5_moving_workbench_follow/) | Green workbench calibration and live XY following during sample handling. |

Step 5 reuses the vision and robot modules from Step 4. Keep these two directories beside each other.
