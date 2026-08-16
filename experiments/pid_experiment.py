"""
PID Controller Trajectory Tracking Benchmark Experiment.

Executes a multi-joint tracking experiment on the Sesame digital twin,
evaluates joint tracking errors, end-effector trajectory, settling behavior,
and overshoot, and saves metrics and plots to results/pid/.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from robot.parameters import (
    JOINT_NAMES,
    JOINT_LIMITS_RAD,
    STAND_POSE_RAD,
    REST_POSE_RAD,
)
from robot.kinematics import SesameKinematics
from simulation.controllers.pid import JointPIDController
from simulation.controllers.trajectory import SinusoidalTrajectory, WalkingGaitTrajectory, StandTrajectory


RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/pid"))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulation/model/sesame.xml"))


def run_pid_experiment(
    trajectory_type: str = "sinusoidal",
    duration_s: float = 5.0,
    dt: float = 0.002,
    results_dir: str = RESULTS_DIR,
) -> dict:
    """
    Run full PID trajectory tracking experiment in MuJoCo.
    """
    import mujoco

    os.makedirs(results_dir, exist_ok=True)
    
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    kin = SesameKinematics()
    
    # Initialize PID controller
    pid = JointPIDController(kp=5.0, ki=0.08, kd=0.15, i_limit=0.25)
    
    # Select trajectory generator
    if trajectory_type == "sinusoidal":
        traj_gen = SinusoidalTrajectory(nominal_pose=STAND_POSE_RAD, amplitude_rad=0.25, frequency_hz=1.0)
    elif trajectory_type == "walking":
        traj_gen = WalkingGaitTrajectory(gait_frequency_hz=1.5)
    else:
        traj_gen = StandTrajectory(transition_time=1.0)
        
    # Reset simulation
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.10  # Spawn height
    
    # Initialize joint positions to t=0 reference
    q_init_ref, _ = traj_gen.get_reference(0.0)
    for i in range(8):
        data.qpos[7 + i] = q_init_ref[i]
        data.ctrl[i] = q_init_ref[i]
    mujoco.mj_forward(model, data)
    
    # History containers
    time_history = []
    target_q_history = []
    actual_q_history = []
    actual_ee_history = []
    target_ee_history = []
    
    num_steps = int(duration_s / dt)
    print(f"Running PID {trajectory_type} experiment for {duration_s}s ({num_steps} steps)...")
    
    for step in range(num_steps):
        t_sim = step * dt
        
        # Get reference
        target_q, target_dq = traj_gen.get_reference(t_sim)
        
        # Read current joint positions & velocities
        current_q = data.qpos[7:15].copy()
        current_dq = data.qvel[6:14].copy()
        
        # Compute PID control command
        cmd_ctrl = pid.compute(target_q, current_q, current_dq, dt=dt, t_sim=t_sim)
        data.ctrl[:] = cmd_ctrl
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Compute analytical end-effector positions
        actual_ee = kin.get_feet_positions_array(current_q)
        target_ee = kin.get_feet_positions_array(target_q)
        
        # Log data
        time_history.append(t_sim)
        target_q_history.append(target_q)
        actual_q_history.append(current_q)
        actual_ee_history.append(actual_ee)
        target_ee_history.append(target_ee)
        
    # Convert to NumPy arrays
    t_arr = np.array(time_history)
    tgt_q_arr = np.array(target_q_history)       # (N, 8)
    act_q_arr = np.array(actual_q_history)       # (N, 8)
    act_ee_arr = np.array(actual_ee_history)     # (N, 4, 3)
    tgt_ee_arr = np.array(target_ee_history)     # (N, 4, 3)
    
    # Calculate quantitative performance metrics
    q_errors = act_q_arr - tgt_q_arr
    rmse_per_joint = np.sqrt(np.mean(q_errors**2, axis=0))
    max_error_per_joint = np.max(np.abs(q_errors), axis=0)
    
    ee_dist_errors = np.linalg.norm(act_ee_arr - tgt_ee_arr, axis=2)  # (N, 4)
    ee_rmse_per_leg = np.sqrt(np.mean(ee_dist_errors**2, axis=0))
    
    overall_q_rmse = float(np.sqrt(np.mean(q_errors**2)))
    overall_ee_rmse = float(np.sqrt(np.mean(ee_dist_errors**2)))
    
    metrics = {
        "trajectory_type": trajectory_type,
        "duration_seconds": duration_s,
        "overall_joint_rmse_rad": overall_q_rmse,
        "overall_joint_rmse_deg": float(np.rad2deg(overall_q_rmse)),
        "overall_end_effector_rmse_m": overall_ee_rmse,
        "overall_end_effector_rmse_mm": overall_ee_rmse * 1000.0,
        "joint_rmse_rad": {name: float(rmse_per_joint[i]) for i, name in enumerate(JOINT_NAMES)},
        "joint_max_error_rad": {name: float(max_error_per_joint[i]) for i, name in enumerate(JOINT_NAMES)},
        "leg_ee_rmse_mm": {
            "FL": float(ee_rmse_per_leg[0] * 1000.0),
            "FR": float(ee_rmse_per_leg[1] * 1000.0),
            "RL": float(ee_rmse_per_leg[2] * 1000.0),
            "RR": float(ee_rmse_per_leg[3] * 1000.0),
        },
    }
    
    # Save metrics JSON
    metrics_path = os.path.join(results_dir, "tracking_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved tracking metrics to: {metrics_path}")
    
    # Generate Plots
    _generate_plots(t_arr, tgt_q_arr, act_q_arr, tgt_ee_arr, act_ee_arr, results_dir)
    
    return metrics


def _generate_plots(t, tgt_q, act_q, tgt_ee, act_ee, save_dir):
    """Generate high-resolution tracking analysis plots."""
    # 1. Joint Tracking Plot (2x4 Grid)
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Sesame Joint Trajectory Tracking Performance (PID Controller)", fontsize=14, fontweight="bold")
    
    for i, j_name in enumerate(JOINT_NAMES):
        row = i % 4
        col = i // 4
        ax = axes[row, col]
        ax.plot(t, np.rad2deg(tgt_q[:, i]), "k--", label="Target", alpha=0.8)
        ax.plot(t, np.rad2deg(act_q[:, i]), "b-", label="Actual", linewidth=1.5)
        ax.set_ylabel(f"{j_name}\n(deg)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        if row == 0 and col == 0:
            ax.legend(loc="upper right", fontsize=8)
            
    axes[3, 0].set_xlabel("Time (s)")
    axes[3, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    joint_plot_path = os.path.join(save_dir, "joint_tracking_trajectory.png")
    plt.savefig(joint_plot_path, dpi=200)
    plt.close()
    print(f"Saved joint tracking plot to: {joint_plot_path}")
    
    # 2. End-Effector Foot Trajectory Plot (4 Legs)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("End-Effector (Foot) Cartesian Tracking (Base Frame)", fontsize=14, fontweight="bold")
    legs = ["FL (Front Left)", "FR (Front Right)", "RL (Rear Left)", "RR (Rear Right)"]
    
    for leg_idx, leg_title in enumerate(legs):
        ax = axes[leg_idx // 2, leg_idx % 2]
        # Plot X-Z trajectory in base sagittal plane
        ax.plot(tgt_ee[:, leg_idx, 0] * 1000, tgt_ee[:, leg_idx, 2] * 1000, "k--", label="Target Foot", alpha=0.8)
        ax.plot(act_ee[:, leg_idx, 0] * 1000, act_ee[:, leg_idx, 2] * 1000, "r-", label="Actual Foot", linewidth=1.5)
        ax.set_title(leg_title, fontsize=11)
        ax.set_xlabel("X Position (mm)")
        ax.set_ylabel("Z Position (mm)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8)
        
    plt.tight_layout()
    ee_plot_path = os.path.join(save_dir, "end_effector_trajectory.png")
    plt.savefig(ee_plot_path, dpi=200)
    plt.close()
    print(f"Saved end-effector plot to: {ee_plot_path}")


if __name__ == "__main__":
    metrics = run_pid_experiment()
    print("=" * 60)
    print("PID TRACKING EXPERIMENT COMPLETED")
    print(f"Overall Joint RMSE: {metrics['overall_joint_rmse_deg']:.3f} deg")
    print(f"Overall Foot RMSE:  {metrics['overall_end_effector_rmse_mm']:.3f} mm")
    print("=" * 60)
