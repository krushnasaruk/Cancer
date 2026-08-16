"""
Test and tune quadruped walking trot gait in MuJoCo physics.
"""

import os
import sys
import mujoco
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from robot.parameters import STAND_POSE_RAD, JOINT_NAMES, JOINT_LIMITS_RAD

def test_trot_gait():
    model = mujoco.MjModel.from_xml_path("simulation/model/sesame.xml")
    data = mujoco.MjData(model)
    
    # Stand initialization
    data.qpos[2] = 0.05
    data.qpos[7:15] = STAND_POSE_RAD
    mujoco.mj_forward(model, data)
    
    # Run 5 seconds (2500 steps)
    dt = model.opt.timestep
    freq = 1.2  # 1.2 Hz
    omega = 2.0 * np.pi * freq
    hip_amp = 0.25
    knee_lift = 0.35
    
    # Joint Indices:
    # 0: fr_hip, 1: rr_hip, 2: fl_hip, 3: rl_hip
    # 4: rr_knee, 5: fr_knee, 6: fl_knee, 7: rl_knee
    
    for step in range(2500):
        t = step * dt
        phi = omega * t
        
        target = STAND_POSE_RAD.copy()
        
        # Smooth transition into gait over 0.5s
        ramp = min(1.0, t / 0.5)
        
        # Diagonal Pair 1: FL (Hip 2, Knee 6) & RR (Hip 1, Knee 4)
        s1 = np.sin(phi)
        c1 = np.cos(phi)
        target[2] += ramp * hip_amp * s1  # FL Hip
        target[1] -= ramp * hip_amp * s1  # RR Hip
        target[6] -= ramp * knee_lift * max(0.0, c1)  # FL Knee lift
        target[4] -= ramp * knee_lift * max(0.0, c1)  # RR Knee lift
        
        # Diagonal Pair 2: FR (Hip 0, Knee 5) & RL (Hip 3, Knee 7) [shifted by pi]
        s2 = np.sin(phi + np.pi)
        c2 = np.cos(phi + np.pi)
        target[0] -= ramp * hip_amp * s2  # FR Hip
        target[3] += ramp * hip_amp * s2  # RL Hip
        target[5] -= ramp * knee_lift * max(0.0, c2)  # FR Knee lift
        target[7] -= ramp * knee_lift * max(0.0, c2)  # RL Knee lift
        
        # Clamp to joint limits
        for i, name in enumerate(JOINT_NAMES):
            low, high = JOINT_LIMITS_RAD[name]
            target[i] = np.clip(target[i], low, high)
            
        data.ctrl[:] = target
        mujoco.mj_step(model, data)
        
    pos = data.qpos[0:3]
    print(f"Final Base Position after 5s: X={pos[0]:+.4f} m (Forward), Y={pos[1]:+.4f} m, Z={pos[2]:.4f} m (Height)")
    is_upright = pos[2] > 0.035
    print(f"Robot is upright: {is_upright}")
    return is_upright

if __name__ == "__main__":
    test_trot_gait()
