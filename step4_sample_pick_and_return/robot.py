from __future__ import annotations

from typing import Any
import math
import socket
import threading
import time

import numpy as np



# Robot connection and tool
ROBOT_IP = '192.168.1.102'
DASHBOARD_PORT = 29999
DASHBOARD_TIMEOUT_S = 4.0
ONROBOT_SINGLE_TCP = [0.0, 0.0, 0.2286, 0.0, 0.0, 0.0]
EXPECTED_PAYLOAD_KG = 1.05
EXPECTED_PAYLOAD_COG_M = [0.004, 0.01, 0.064]
PAYLOAD_TOLERANCE_KG = 0.08
COG_TOLERANCE_M = 0.015

# Grasp and motion parameters
GRASP_HEIGHT_ABOVE_SUPPORT_M = 0.032
TRANSFER_CLEARANCE_M = 0.105
MIN_TRANSFER_Z_M = 0.105
WAIT_AT_PLACE_S = 2.0
JOINT_SPEED_RAD_S = 0.4
JOINT_ACCEL_RAD_S2 = 0.6
TRANSFER_SPEED_M_S = 0.12
TRANSFER_ACCEL_M_S2 = 0.25
VERTICAL_SPEED_M_S = 0.035
VERTICAL_ACCEL_M_S2 = 0.1
MAX_START_JOINT_CHANGE_RAD = math.radians(1.0)


def dashboard_session():
    sock = socket.create_connection((ROBOT_IP, DASHBOARD_PORT), timeout=DASHBOARD_TIMEOUT_S)
    sock.settimeout(DASHBOARD_TIMEOUT_S)
    stream = sock.makefile('rwb', buffering=0)
    banner = stream.readline().decode('utf-8', errors='replace').strip()
    return (sock, stream, banner)


def dashboard_command(stream, command: str) -> str:
    stream.write((command + '\n').encode('utf-8'))
    response = stream.readline().decode('utf-8', errors='replace').strip()
    print(f'Dashboard: {command} -> {response}', flush=True)
    return response


def verify_remote_and_stopped() -> None:
    sock, stream, banner = dashboard_session()
    try:
        print(banner, flush=True)
        remote = dashboard_command(stream, 'is in remote control')
        if remote.strip().lower() != 'true':
            raise RuntimeError('Robot is not in Remote Control mode')
        state = dashboard_command(stream, 'programState')
        if not state.upper().startswith('STOPPED'):
            raise RuntimeError(f'Another program is active: {state}')
    finally:
        stream.close()
        sock.close()


def read_current_robot(record: dict, receiver=None) -> dict:
    from rtde_receive import RTDEReceiveInterface
    owns_receiver = receiver is None
    last_error = None
    for attempt in range(1, 4):
        try:
            if receiver is None:
                receiver = RTDEReceiveInterface(ROBOT_IP)
            if hasattr(receiver, 'read_basic_state'):
                state = receiver.read_basic_state(reconnect_on_stale=True)
                pose = np.asarray(state['pose'], dtype=np.float64)
                joints = np.asarray(state['joints'], dtype=np.float64)
                payload = state['payload']
                cog = None if state['cog'] is None else np.asarray(state['cog'], dtype=np.float64)
            else:
                pose = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
                joints = np.asarray(receiver.getActualQ(), dtype=np.float64)
                payload = float(receiver.getPayload()) if hasattr(receiver, 'getPayload') else None
                cog = np.asarray(receiver.getPayloadCog(), dtype=np.float64) if hasattr(receiver, 'getPayloadCog') else None
            if owns_receiver:
                receiver.disconnect()
                receiver = None
            saved_q = np.asarray(record['robot_at_capture']['actual_q_rad'], dtype=np.float64)
            max_joint_change = float(np.max(np.abs(joints - saved_q)))
            return {
                "pose": pose,
                "joints": joints,
                "payload": payload,
                "cog": cog,
                "max_joint_change_rad": max_joint_change,
            }
        except Exception as error:
            last_error = error
            if owns_receiver and receiver is not None:
                try:
                    receiver.disconnect()
                except Exception:
                    pass
                receiver = None
            time.sleep(0.5 * attempt)
    raise RuntimeError(f'Could not read the UR state after 3 attempts: {last_error}')


def validate_start(record: dict, current: dict) -> None:
    if current['max_joint_change_rad'] > MAX_START_JOINT_CHANGE_RAD:
        raise RuntimeError(
            "Robot joints changed since STEP 1 "
            f"({math.degrees(current['max_joint_change_rad']):.3f} deg max). "
            "Run STEP 1 again; do not use stale object coordinates."
        )
    payload = current['payload']
    cog = current['cog']
    if payload is not None and abs(payload - EXPECTED_PAYLOAD_KG) > PAYLOAD_TOLERANCE_KG:
        raise RuntimeError(f'Current payload is {payload:.3f} kg; expected {EXPECTED_PAYLOAD_KG:.3f} kg')
    if cog is not None and np.linalg.norm(cog - EXPECTED_PAYLOAD_COG_M) > COG_TOLERANCE_M:
        raise RuntimeError(f'Current payload CoG does not match [4,10,64] mm: {(cog * 1000.0).round(2).tolist()} mm')


def start_rtde_control():
    from rtde_control import RTDEControlInterface
    RTDE_CONTROL_FREQUENCY_HZ = 125.0
    control = RTDEControlInterface(ROBOT_IP, RTDE_CONTROL_FREQUENCY_HZ)
    time.sleep(0.4)
    control.setTcp(ONROBOT_SINGLE_TCP)
    control.setPayload(EXPECTED_PAYLOAD_KG, EXPECTED_PAYLOAD_COG_M)
    return control


def stop_rtde_control(control) -> None:
    if control is None:
        return
    try:
        control.stopL(1.0)
    except Exception:
        pass
    try:
        control.stopScript()
    except Exception:
        pass
    try:
        control.disconnect()
    except Exception:
        pass
    time.sleep(0.5)


def move_l(control, pose, speed, acceleration, label: str) -> None:
    print(f'MOVE {label}: {[round(v, 6) for v in pose]}', flush=True)
    if hasattr(control, 'isPoseWithinSafetyLimits'):
        if not control.isPoseWithinSafetyLimits(pose):
            raise RuntimeError(f'Controller safety-limit check rejected {label}')
    result = control.moveL(pose, speed, acceleration)
    if result is False:
        raise RuntimeError(f'moveL returned False at {label}')


def move_j(control, joints, label: str) -> None:
    print(f'MOVE {label}: {[round(v, 6) for v in joints]}', flush=True)
    if hasattr(control, 'isJointsWithinSafetyLimits'):
        if not control.isJointsWithinSafetyLimits(joints):
            raise RuntimeError(f'Controller safety-limit check rejected {label}')
    result = control.moveJ(joints, JOINT_SPEED_RAD_S, JOINT_ACCEL_RAD_S2)
    if result is False:
        raise RuntimeError(f'moveJ returned False at {label}')


class StaleRTDEReceiveError(RuntimeError):
    """The receive socket may exist, but robot data is no longer advancing."""


class PersistentRTDEReceive:
    """Own one RTDEReceiveInterface and reject cached robot state.

    The robot controller timestamp must advance before every returned state
    snapshot.  A dead socket that keeps returning cached pose vectors is
    therefore rejected before it can be used for Base-frame placement.
    """

    def __init__(self, robot_ip: str, *, frequency_hz: float=125.0, connect_attempts: int=4, freshness_timeout_s: float=0.4) -> None:
        self.robot_ip = str(robot_ip)
        self.frequency_hz = float(frequency_hz)
        self.connect_attempts = int(connect_attempts)
        self.freshness_timeout_s = float(freshness_timeout_s)
        self._lock = threading.RLock()
        self._receiver: Any | None = None
        self._last_timestamp: float | None = None
        self._generation = 0
        self._closed = False
        self._connect_new_locked()

    @property
    def generation(self) -> int:
        return int(self._generation)

    def _construct_receiver(self):
        from rtde_receive import RTDEReceiveInterface
        try:
            return RTDEReceiveInterface(self.robot_ip, self.frequency_hz)
        except TypeError:
            return RTDEReceiveInterface(self.robot_ip)

    @staticmethod
    def _disconnect_raw(receiver) -> None:
        if receiver is None:
            return
        try:
            receiver.disconnect()
        except Exception:
            pass

    @staticmethod
    def _timestamp_from(receiver) -> float:
        if receiver is None or not hasattr(receiver, 'getTimestamp'):
            raise StaleRTDEReceiveError(
                "RTDEReceiveInterface has no getTimestamp(); a freshness-safe "
                "placement coordinate cannot be produced"
            )
        value = float(receiver.getTimestamp())
        if not math.isfinite(value):
            raise StaleRTDEReceiveError('RTDE robot timestamp is non-finite')
        return value

    @staticmethod
    def _finite_vector(values, expected_length: int, label: str) -> list[float]:
        result = [float(value) for value in values]
        if len(result) != expected_length or not all((math.isfinite(value) for value in result)):
            raise StaleRTDEReceiveError(f'RTDE returned an invalid {label} vector')
        return result

    def _wait_initial_stream_locked(self) -> None:
        assert self._receiver is not None
        first = self._timestamp_from(self._receiver)
        deadline = time.monotonic() + self.freshness_timeout_s
        while time.monotonic() < deadline:
            time.sleep(0.01)
            current = self._timestamp_from(self._receiver)
            if current > first + 1e-09:
                self._last_timestamp = current
                return
        raise StaleRTDEReceiveError('RTDE connected but the robot timestamp did not advance')

    def _connect_new_locked(self) -> None:
        self._disconnect_raw(self._receiver)
        self._receiver = None
        self._last_timestamp = None
        last_error: Exception | None = None
        for attempt in range(1, self.connect_attempts + 1):
            candidate = None
            try:
                candidate = self._construct_receiver()
                self._receiver = candidate
                self._wait_initial_stream_locked()
                self._generation += 1
                self._closed = False
                print(
                    "Persistent RTDE Receive ready: "
                    f"{self.frequency_hz:.0f} Hz; generation "
                    f"{self._generation}; timestamp is advancing.",
                    flush=True,
                )
                return
            except Exception as error:
                last_error = error
                self._disconnect_raw(candidate)
                self._receiver = None
                if attempt < self.connect_attempts:
                    time.sleep(0.5 * attempt)
        raise RuntimeError(
            "Persistent RTDE Receive failed after "
            f"{self.connect_attempts} attempts: {last_error}"
        ) from last_error

    def _require_fresh_packet_locked(self) -> float:
        if self._closed or self._receiver is None:
            raise StaleRTDEReceiveError('Persistent RTDE Receive is closed')
        baseline = self._last_timestamp
        deadline = time.monotonic() + self.freshness_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                current = self._timestamp_from(self._receiver)
                if baseline is None or current > baseline + 1e-09:
                    self._last_timestamp = current
                    return current
            except Exception as error:
                last_error = error
                break
            time.sleep(0.005)
        detail = 'timestamp stopped advancing' if last_error is None else str(last_error)
        raise StaleRTDEReceiveError(f'RTDE Receive data is stale: {detail}')

    def _snapshot_once_locked(self, snapshot_fn, tcp_offset) -> dict:
        timestamp = self._require_fresh_packet_locked()
        assert self._receiver is not None
        snapshot = snapshot_fn(self._receiver, tcp_offset)
        snapshot['rtde_robot_timestamp_s'] = float(timestamp)
        snapshot['rtde_receive_generation'] = int(self._generation)
        snapshot['rtde_receive_frequency_hz'] = float(self.frequency_hz)
        return snapshot

    def snapshot(self, snapshot_fn, tcp_offset, *, reconnect_on_stale: bool=False) -> dict:
        with self._lock:
            try:
                return self._snapshot_once_locked(snapshot_fn, tcp_offset)
            except Exception:
                if not reconnect_on_stale:
                    raise
                print(
                    "RTDE Receive freshness check failed while robot motion is "
                    "not being commanded; rebuilding the receive connection.",
                    flush=True,
                )
                self._connect_new_locked()
                return self._snapshot_once_locked(snapshot_fn, tcp_offset)

    def read_basic_state(self, *, reconnect_on_stale: bool=False) -> dict:
        with self._lock:
            try:
                timestamp = self._require_fresh_packet_locked()
                assert self._receiver is not None
                pose = self._finite_vector(self._receiver.getActualTCPPose(), 6, 'TCP pose')
                joints = self._finite_vector(self._receiver.getActualQ(), 6, 'joint position')
                payload = float(self._receiver.getPayload()) if hasattr(self._receiver, 'getPayload') else None
                cog = (
                    self._finite_vector(
                        self._receiver.getPayloadCog(), 3, "payload COG"
                    )
                    if hasattr(self._receiver, "getPayloadCog")
                    else None
                )
                return {
                    "pose": pose,
                    "joints": joints,
                    "payload": payload,
                    "cog": cog,
                    "rtde_robot_timestamp_s": float(timestamp),
                    "rtde_receive_generation": int(self._generation),
                }
            except Exception:
                if not reconnect_on_stale:
                    raise
                self._connect_new_locked()
                return self.read_basic_state(reconnect_on_stale=False)

    def read_output_int_register(self, register_id: int, *, reconnect_on_stale: bool=False) -> int:
        """Read one robot output-int register through this same connection.

        ur_rtde 1.6.5's default receive recipe exposes output integer
        registers 18..22.  The RG2 command program uses output register 18 as
        its acknowledgement channel, while Python writes the independent
        input register 18 as the command channel.
        """

        def read_once() -> int:
            self._require_fresh_packet_locked()
            assert self._receiver is not None
            getter = getattr(self._receiver, 'getOutputIntRegister', None)
            if getter is None:
                raise StaleRTDEReceiveError('RTDEReceiveInterface has no getOutputIntRegister()')
            value = int(getter(int(register_id)))
            return value
        with self._lock:
            try:
                return read_once()
            except Exception:
                if not reconnect_on_stale:
                    raise
                print(
                    "RTDE Receive failed while reading the RG2 acknowledgement; "
                    "rebuilding the same persistent receive connection.",
                    flush=True,
                )
                self._connect_new_locked()
                return read_once()

    def reconnect(self) -> None:
        with self._lock:
            print('Rebuilding the single persistent RTDE Receive connection...', flush=True)
            self._connect_new_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            receiver = self._receiver
            self._receiver = None
            self._last_timestamp = None
            self._disconnect_raw(receiver)
            print('Persistent RTDE Receive closed at program exit.', flush=True)
