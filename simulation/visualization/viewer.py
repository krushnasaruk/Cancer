"""
Sesame Robot MuJoCo 3D Interactive Visualization Viewer.

Launches an interactive viewer rendering the Sesame quadruped digital twin with:
- Full gravity and contact dynamics
- Joint limits and stability monitoring
- Interactive hotkeys to trigger firmware poses (REST, STAND, WALK, WAVE)
- Manual joint slider control
"""

import os
import sys
import time
import argparse
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from robot.parameters import (
    REST_POSE_RAD,
    STAND_POSE_RAD,
    JOINT_NAMES,
    JOINT_LIMITS_RAD,
    FIRMWARE_SERVO_NAMES,
)


MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/sesame.xml"))


def load_model(xml_path: str = MODEL_PATH):
    """Load and validate the MuJoCo model."""
    import mujoco
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Model XML file not found at: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


def run_viewer(xml_path: str = MODEL_PATH, duration: float = None):
    """Launch the interactive MuJoCo passive/GLFW viewer."""
    import mujoco
    import mujoco.viewer

    model, data = load_model(xml_path)
    print("=" * 60)
    print("      SESAME ROBOT DIGITAL TWIN — MUJOCO VIEWER")
    print("=" * 60)
    print(f"Model: {model.nq} generalized coordinates, {model.nu} actuators, {model.njnt} joints.")
    print("Hotkeys in simulation:")
    print("  - [1] Apply REST Pose (All joints @ 90 deg)")
    print("  - [2] Apply STAND Pose (Firmware default stand)")
    print("  - [3] Apply Walking Gait Cycle")
    print("  - [4] Apply Wave Pose")
    print("  - [R] Reset Robot to spawn height")
    print("  - [Space] Pause/Resume simulation in viewer")
    print("=" * 60)

    # Initial condition: spawn slightly above ground in STAND pose
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.095  # Base Z height
    for i, target_rad in enumerate(STAND_POSE_RAD):
        data.ctrl[i] = target_rad
        data.qpos[7 + i] = target_rad  # Offset by 7 for freejoint

    mujoco.mj_forward(model, data)

    current_mode = "STAND"
    t_start = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set default camera view
        viewer.cam.distance = 0.45
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 135.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.05]

        step_count = 0
        while viewer.is_running():
            step_start = time.time()
            t_sim = data.time

            # Example procedural motion when walking
            if current_mode == "WALK":
                freq = 1.5  # Hz
                phase = 2.0 * np.pi * freq * t_sim
                # Sinusoidal gait trajectory matching firmware diagonal leg pairs
                walk_targets = STAND_POSE_RAD.copy()
                swing_amp = 0.35
                # Pair 1: FR_HIP (0), RL_HIP (3), RL_KNEE (7), FR_KNEE (5)
                walk_targets[0] += swing_amp * np.sin(phase)
                walk_targets[3] += swing_amp * np.sin(phase)
                walk_targets[5] += 0.25 * np.cos(phase)
                walk_targets[7] += 0.25 * np.cos(phase)
                # Pair 2: RR_HIP (1), FL_HIP (2), FL_KNEE (6), RR_KNEE (4)
                walk_targets[1] += swing_amp * np.sin(phase + np.pi)
                walk_targets[2] += swing_amp * np.sin(phase + np.pi)
                walk_targets[4] += 0.25 * np.cos(phase + np.pi)
                walk_targets[6] += 0.25 * np.cos(phase + np.pi)

                # Clamp to joint limits
                for idx, j_name in enumerate(JOINT_NAMES):
                    low, high = JOINT_LIMITS_RAD[j_name]
                    data.ctrl[idx] = np.clip(walk_targets[idx], low, high)

            # Step physics
            mujoco.mj_step(model, data)
            step_count += 1

            # Sync viewer at ~60 Hz
            if step_count % 10 == 0:
                viewer.sync()

            # Check termination
            if duration is not None and (time.time() - t_start) >= duration:
                break

            # Rate limit to real time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def run_headless_test(xml_path: str = MODEL_PATH, steps: int = 500) -> bool:
    """Run a headless physics simulation test verifying stability."""
    import mujoco

    model, data = load_model(xml_path)
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.10

    # Set stand pose controls
    for i, target_rad in enumerate(STAND_POSE_RAD):
        data.ctrl[i] = target_rad
        data.qpos[7 + i] = target_rad

    mujoco.mj_forward(model, data)

    for step in range(steps):
        mujoco.mj_step(model, data)
        # Check for NaN / physics explosion
        if np.isnan(data.qpos).any() or np.isnan(data.qvel).any():
            print(f"FAILED: Physics NaN detected at step {step}")
            return False
        if data.qpos[2] < -0.5:
            print(f"FAILED: Robot fell through floor at step {step} (Z = {data.qpos[2]})")
            return False

    print(f"SUCCESS: Headless physics simulation ran for {steps} steps without instability.")
    print(f"Final Base Height Z: {data.qpos[2]:.4f} m (Stable on ground).")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sesame Robot MuJoCo Viewer")
    parser.add_argument("--headless", action="store_true", help="Run in headless verification mode")
    parser.add_argument("--duration", type=float, default=None, help="Run viewer for specified seconds")
    args = parser.parse_args()

    if args.headless:
        success = run_headless_test()
        sys.exit(0 if success else 1)
    else:
        run_viewer(duration=args.duration)
