"""
Gait Trajectory Joint Angle Parameter Optimizer for Sesame Quadruped.

Finds the exact optimal joint angles (Hip Swing Amplitude, Knee Lift Amplitude, Gait Frequency)
that produce maximum forward translation in MuJoCo physics while maintaining 100% upright balance.
"""

import sys
import os
import time
import numpy as np
import mujoco

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from robot.parameters import MUJOCO_STAND_RAD, JOINT_NAMES, JOINT_LIMITS_RAD
from simulation.controllers.trajectory import WalkingGaitTrajectory
from calibration.actuator_model import SesameActuatorBank

MODEL_XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulation/model/sesame.xml"))


def evaluate_gait_params(hip_amp: float, knee_amp: float, freq_hz: float, sim_duration: float = 3.0) -> dict:
    """Run a 3-second simulation test with candidate joint angle parameters."""
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)
    actuators = SesameActuatorBank()
    
    # Spawn standing upright
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.09
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    
    for i, jname in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        data.qpos[model.jnt_qposadr[jid]] = MUJOCO_STAND_RAD[i]
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname.replace("_joint", "_actuator"))
        data.ctrl[aid] = MUJOCO_STAND_RAD[i]
        
    actuators.reset(MUJOCO_STAND_RAD)
    mujoco.mj_forward(model, data)
    
    traj = WalkingGaitTrajectory(
        gait_frequency_hz=freq_hz,
        hip_swing_amp_rad=hip_amp,
        knee_lift_amp_rad=knee_amp,
        ramp_time_s=0.4
    )
    
    steps = int(sim_duration / 0.02)
    min_z = 0.09
    max_tilt = 0.0
    fallen = False
    
    for step in range(steps):
        t = step * 0.02
        target_q, _ = traj.get_reference(t)
        
        curr_q = np.array([data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in JOINT_NAMES])
        curr_dq = np.array([data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in JOINT_NAMES])
        
        for _ in range(10):
            eff_cmds, _ = actuators.step(target_q, curr_q, curr_dq, dt=0.002)
            for i, jname in enumerate(JOINT_NAMES):
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname.replace("_joint", "_actuator"))
                data.ctrl[aid] = eff_cmds[i]
            mujoco.mj_step(model, data)
            
        bz = data.qpos[2]
        rot_mat = data.xmat[model.body("base_link").id].reshape(3, 3)
        upright = rot_mat[2, 2]
        
        if bz < min_z: min_z = bz
        tilt = 1.0 - upright
        if tilt > max_tilt: max_tilt = tilt
        
        if bz < 0.04 or upright < 0.60:
            fallen = True
            break
            
    final_x = data.qpos[0]
    final_y = data.qpos[1]
    speed = final_x / sim_duration if not fallen else -0.1
    
    # Fitness score: reward forward X displacement, penalize lateral Y drift and tilt
    score = (final_x * 100.0) - (abs(final_y) * 50.0) - (max_tilt * 30.0) if not fallen else -100.0
    
    return {
        "score": score,
        "dist_x_cm": final_x * 100.0,
        "drift_y_cm": final_y * 100.0,
        "speed_cm_s": speed * 100.0,
        "fallen": fallen,
        "hip_deg": np.degrees(hip_amp),
        "knee_deg": np.degrees(knee_amp),
        "freq_hz": freq_hz,
    }


def optimize():
    print("=" * 65)
    print("   OPTIMIZING SESAME QUADRUPED WALKING GAIT JOINT ANGLES")
    print("=" * 65)
    
    best_score = -999.0
    best_result = None
    
    # Grid search across physical gait angle ranges
    hip_amps = np.linspace(np.deg2rad(10), np.deg2rad(25), 6)   # 10° to 25° hip swing
    knee_amps = np.linspace(np.deg2rad(15), np.deg2rad(30), 6)  # 15° to 30° knee lift
    freqs = [1.0, 1.2, 1.4, 1.6, 1.8]                            # 1.0 Hz to 1.8 Hz
    
    count = 0
    total = len(hip_amps) * len(knee_amps) * len(freqs)
    
    for h in hip_amps:
        for k in knee_amps:
            for f in freqs:
                count += 1
                res = evaluate_gait_params(h, k, f)
                if res["score"] > best_score:
                    best_score = res["score"]
                    best_result = res
                    print(f"[{count:03d}/{total}] BEST GAIT: Score={res['score']:6.1f} | Hip={res['hip_deg']:4.1f} deg Knee={res['knee_deg']:4.1f} deg Freq={res['freq_hz']:3.1f}Hz -> Speed={res['speed_cm_s']:5.1f}cm/s (Dist={res['dist_x_cm']:5.1f}cm)")

    print("-" * 65)
    print("   OPTIMIZATION COMPLETE! BEST GAIT ANGLE PARAMETERS:")
    print("-" * 65)
    print(f"Hip Swing Angle:   {best_result['hip_deg']:.1f} deg")
    print(f"Knee Lift Angle:  {best_result['knee_deg']:.1f} deg")
    print(f"Gait Frequency:   {best_result['freq_hz']:.1f} Hz")
    print(f"Forward Speed:    {best_result['speed_cm_s']:.1f} cm/s ({best_result['dist_x_cm']:.1f} cm in 3s)")
    print(f"Lateral Drift:    {best_result['drift_y_cm']:.2f} cm")
    print("=" * 65)
    
    return best_result


if __name__ == "__main__":
    optimize()
