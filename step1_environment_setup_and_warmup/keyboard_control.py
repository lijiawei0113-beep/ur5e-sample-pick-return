import gc
import queue
import socket
import threading
import time

from pynput import keyboard
from rtde_control import RTDEControlInterface


ROBOT_IP = '192.168.1.102'
DASHBOARD_PORT = 29999
RG2_OPEN_PROGRAM = '/programs/rg2_open_Li.urp'
RG2_CLOSE_PROGRAM = '/programs/rg2_close_Li.urp'
JOINT_SPEED = 0.04
JOINT_ACCELERATION = 0.1
COMMAND_TIME = 0.05
CONTROL_START_DELAY = 1.0
CONTROL_STOP_DELAY = 0.25
DASHBOARD_TIMEOUT = 3.0
RG2_PROGRAM_TIMEOUT = 10.0

JOINT_MAP = {
    'q': (0, +1, 'J1'),
    'a': (0, -1, 'J1'),
    'w': (1, +1, 'J2'),
    's': (1, -1, 'J2'),
    'e': (2, +1, 'J3'),
    'd': (2, -1, 'J3'),
    'r': (3, +1, 'J4'),
    'f': (3, -1, 'J4'),
    't': (4, +1, 'J5'),
    'g': (4, -1, 'J5'),
    'y': (5, +1, 'J6'),
    'h': (5, -1, 'J6'),
}

deadman = threading.Event()
quit_event = threading.Event()
gripper_busy = threading.Event()
state_lock = threading.Lock()
active_keys = set()
gripper_requests = queue.Queue(maxsize=1)


def clear_motion_keys():
    with state_lock:
        active_keys.clear()


def on_press(key):
    if key == keyboard.Key.esc:
        deadman.clear()
        clear_motion_keys()
        quit_event.set()
        print('Exit requested')
        return False

    if key == keyboard.Key.space:
        if gripper_busy.is_set():
            print('RG2 is running. Please wait.')
            return
        if not deadman.is_set():
            deadman.set()
            print('Motion enabled: ON')
        return

    try:
        char = key.char.lower()
    except AttributeError:
        return

    if char in ('o', 'c'):
        if gripper_busy.is_set():
            print('RG2 is running. Please wait.')
            return
        if not deadman.is_set():
            print(f'Ignored {char.upper()}: hold Space first')
            return
        deadman.clear()
        clear_motion_keys()
        action = 'OPEN' if char == 'o' else 'CLOSE'
        try:
            gripper_requests.put_nowait(action)
            print(f'RG2 {action} request received')
            print('Hold Space again after the gripper action is complete')
        except queue.Full:
            print('An RG2 command is already pending')
        return

    if char not in JOINT_MAP:
        return
    if gripper_busy.is_set():
        print('RG2 is running. Joint control is temporarily disabled.')
        return
    if not deadman.is_set():
        print(f'Ignored {char.upper()}: hold Space first')
        return
    with state_lock:
        if char in active_keys:
            return
        active_keys.add(char)

    _, direction, joint_name = JOINT_MAP[char]
    sign = '+' if direction > 0 else '-'
    print(f'{joint_name}{sign}')


def on_release(key):
    if key == keyboard.Key.space:
        deadman.clear()
        clear_motion_keys()
        print('Motion enabled: OFF / STOP')
        return

    try:
        char = key.char.lower()
    except AttributeError:
        return
    if char not in JOINT_MAP:
        return
    with state_lock:
        was_active = char in active_keys
        active_keys.discard(char)

    if was_active:
        _, _, joint_name = JOINT_MAP[char]
        print(f'{joint_name} STOP')


def calculate_joint_velocity():
    velocity = [0.0] * 6
    if not deadman.is_set():
        return velocity
    if gripper_busy.is_set():
        return velocity
    with state_lock:
        keys = list(active_keys)

    # Move one joint at a time.
    if len(keys) != 1:
        return velocity
    key = keys[0]
    if key not in JOINT_MAP:
        return velocity
    joint_index, direction, _ = JOINT_MAP[key]
    velocity[joint_index] = direction * JOINT_SPEED
    return velocity


def start_motion_control():
    print('Establishing joint RTDE control...')
    control = RTDEControlInterface(ROBOT_IP)
    time.sleep(CONTROL_START_DELAY)
    print('Joint RTDE control established')
    print('Please hold Space again')
    return control


def stop_motion_control(control):
    if control is None:
        return
    print('Stopping joint RTDE control...')
    try:
        control.speedStop()
        print('Joint motion stopped')
    except Exception as error:
        print(f'speedStop warning: {error}')
    time.sleep(0.1)
    try:
        control.stopScript()
        print('RTDE control script stopped')
    except Exception as error:
        print(f'stopScript warning: {error}')
    time.sleep(0.1)
    try:
        control.disconnect()
    except Exception:
        pass
    print('Joint RTDE control closed')


def send_dashboard_command(connection_file, command):
    connection_file.write((command + '\n').encode('utf-8'))
    response = connection_file.readline().decode('utf-8').strip()
    print(f'Dashboard: {command} -> {response}')
    return response


def stop_motion_control_for_gripper(control):
    if control is None:
        return True
    print('Stopping RTDE program through Dashboard Server...')
    stopped = False
    try:
        with socket.create_connection(
            (ROBOT_IP, DASHBOARD_PORT), timeout=DASHBOARD_TIMEOUT
        ) as dashboard_socket:
            dashboard_socket.settimeout(DASHBOARD_TIMEOUT)
            with dashboard_socket.makefile('rwb', buffering=0) as connection_file:
                banner = connection_file.readline().decode('utf-8').strip()
                print(banner)
                response = send_dashboard_command(connection_file, 'stop')
                stopped = response.strip().lower() == 'stopped'
                if not stopped:
                    state = send_dashboard_command(connection_file, 'programState')
                    stopped = state.upper().startswith('STOPPED')
    except Exception as error:
        print(f'Dashboard stop failed: {error}')

    try:
        control.disconnect()
    except Exception as error:
        print(f'RTDE disconnect warning: {error}')
    time.sleep(0.3)
    if stopped:
        print('RTDE program stopped')
        print('Joint RTDE control closed')
    else:
        print('Could not confirm that the RTDE program stopped')
        print('Press Stop on the teach pendant before continuing')
    return stopped


def run_rg2_program(program_path, action_name):
    print(f'Executing RG2 {action_name}...')
    with socket.create_connection(
        (ROBOT_IP, DASHBOARD_PORT), timeout=DASHBOARD_TIMEOUT
    ) as dashboard_socket:
        dashboard_socket.settimeout(DASHBOARD_TIMEOUT)
        with dashboard_socket.makefile('rwb', buffering=0) as connection_file:
            banner = connection_file.readline().decode('utf-8').strip()
            print(banner)
            remote_response = send_dashboard_command(connection_file, 'is in remote control')
            if remote_response.lower() != 'true':
                raise RuntimeError('The robot is not in Remote Control mode')

            load_response = send_dashboard_command(connection_file, f'load {program_path}')
            if 'loading program' not in load_response.lower():
                raise RuntimeError(f'Failed to load RG2 program: {load_response}')

            play_response = send_dashboard_command(connection_file, 'play')
            if 'starting program' not in play_response.lower():
                raise RuntimeError(f'Failed to start RG2 program: {play_response}')

            deadline = time.time() + RG2_PROGRAM_TIMEOUT
            while time.time() < deadline:
                if quit_event.is_set():
                    send_dashboard_command(connection_file, 'stop')
                    print('RG2 program stopped by Esc')
                    return False
                time.sleep(0.25)
                state = send_dashboard_command(connection_file, 'programState')
                if state.upper().startswith('STOPPED'):
                    print(f'RG2 {action_name} completed')
                    return True
            send_dashboard_command(connection_file, 'stop')
            raise TimeoutError(f'RG2 program did not finish within {RG2_PROGRAM_TIMEOUT} seconds')


def execute_gripper_action(control, action):
    deadman.clear()
    clear_motion_keys()
    gripper_busy.set()
    try:
        stopped = stop_motion_control_for_gripper(control)
        control = None
        if not stopped:
            print('RG2 action cancelled because the RTDE program could not be stopped safely')
            quit_event.set()
            return None

        gc.collect()
        time.sleep(CONTROL_STOP_DELAY)
        if action == 'OPEN':
            completed = run_rg2_program(RG2_OPEN_PROGRAM, 'OPEN')
        else:
            completed = run_rg2_program(RG2_CLOSE_PROGRAM, 'CLOSE')
        if not completed:
            return None
        if quit_event.is_set():
            return None

        gc.collect()
        time.sleep(CONTROL_STOP_DELAY)
        # Block keyboard motion until the RTDE session is ready.
        new_control = start_motion_control()
        # Require Space again after each gripper action.
        deadman.clear()
        clear_motion_keys()
        return new_control
    except Exception as error:
        print(f'Gripper control error: {error}')
        quit_event.set()
        return None
    finally:
        deadman.clear()
        clear_motion_keys()
        gripper_busy.clear()


def main():
    control = None
    listener = None
    try:
        print("""
Joint control:
    Space + Q/A: J1 +/-
    Space + W/S: J2 +/-
    Space + E/D: J3 +/-
    Space + R/F: J4 +/-
    Space + T/G: J5 +/-
    Space + Y/H: J6 +/-

RG2 control:
    Space + O: Open
    Space + C: Grip a 25 mm cylinder

After each RG2 action, wait until the new
RTDE control session is established, then
release and hold Space again.

Release Space: Stop immediately
Esc: Stop and exit
""")
        control = start_motion_control()
        listener = keyboard.Listener(on_press=on_press, on_release=on_release, suppress=True)
        listener.start()
        while not quit_event.is_set():
            try:
                action = gripper_requests.get_nowait()
            except queue.Empty:
                action = None
            if action is not None:
                control = execute_gripper_action(control, action)
                continue
            if control is None:
                time.sleep(COMMAND_TIME)
                continue
            joint_velocity = calculate_joint_velocity()
            control.speedJ(joint_velocity, acceleration=JOINT_ACCELERATION, time=COMMAND_TIME)
    except KeyboardInterrupt:
        print('Ctrl+C received. Preparing to exit.')
    except Exception as error:
        print(f'Program error: {error}')
    finally:
        deadman.clear()
        clear_motion_keys()
        quit_event.set()
        stop_motion_control(control)
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
        print('Robot stopped. Program exited.')


if __name__ == '__main__':
    main()
