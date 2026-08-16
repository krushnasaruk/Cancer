"""
Gymnasium Locomotion Environment for Sesame Quadruped Robot (Autonomous Walking).

Focuses on:
- Fast forward locomotion tracking (+X velocity)
- Phase-guided cyclic gait coordination (CPG Phase Clock)
- Dynamic balance, anti-drift, and low-energy torque regularization
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from robot.parameters import (
    JOINT_NAMES,
    JOINT_LIMITS_RAD,
    MUJOCO_STAND_RAD,
)
from calibration.actuator_model import SesameActuatorBank

DEFAULT_XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/sesame.xml"))


class SesameWalkEnv(gym.Env):
    """
    Gymnasium environment for Sesame quadruped autonomous walking locomotion.
    
    Action Space: Box(-1.0, 1.0, shape=(8,)) -> Continuous joint angle offsets around STAND pose.
    Observation Space: Box(shape=(36,)) -> Joint angles, velocities, IMU orientation, base velocities, foot contacts, and 2D Phase Clock.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = DEFAULT_XML_PATH,
        render_mode: Optional[str] = None,
        frame_skip: int = 10,  # 500 Hz physics / 10 = 50 Hz control loop
        target_velocity: float = 0.15,  # Target forward speed: 0.15 m/s (~15 cm/s)
        gait_freq_hz: float = 1.4,     # Nominal stepping frequency (1.4 Hz)
        use_actuator_model: bool = True,
        max_episode_steps: int = 400,
    ):
        super().__init__()
        import mujoco

        self.xml_path = os.path.abspath(xml_path)
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.target_velocity = target_velocity
        self.gait_freq = gait_freq_hz
        self.use_actuator_model = use_actuator_model
        self.max_episode_steps = max_episode_steps

        # Load MuJoCo Model
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Joint Limits & Scaling
        self.joint_min = np.array([JOINT_LIMITS_RAD[j][0] for j in JOINT_NAMES], dtype=np.float64)
        self.joint_max = np.array([JOINT_LIMITS_RAD[j][1] for j in JOINT_NAMES], dtype=np.float64)
        self.stand_pose = MUJOCO_STAND_RAD.copy()
        self.action_scale = 0.35  # ±0.35 rad (~20 deg) swing around stand pose

        # Actuator Bank (Parametric Non-Linear Servo Dynamics)
        self.actuator_bank = SesameActuatorBank() if use_actuator_model else None

        # Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        # Obs Dim = 8 (q) + 8 (dq) + 3 (euler) + 3 (linvel) + 3 (angvel) + 4 (contacts) + 4 (feet Z) + 1 (base Z) + 2 (Phase Clock) + 1 (target v) = 37
        self.obs_dim = 37
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.current_step = 0
        self.prev_x = 0.0
        self.prev_action = np.zeros(8, dtype=np.float64)
        self.viewer = None

    def _get_obs(self) -> np.ndarray:
        # 1. Joint positions normalized relative to stand pose
        q_raw = self.data.qpos[7:15]
        q_norm = (q_raw - self.stand_pose) / self.action_scale
        
        # 2. Joint velocities
        dq = self.data.qvel[6:14] * 0.1
        
        # 3. Base orientation (Euler roll, pitch, yaw)
        w, x, y, z = self.data.qpos[3:7]
        roll = np.arctan2(2.0 * (w * x + y * y), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        euler = np.array([roll, pitch, yaw], dtype=np.float64)
        
        # 4. Base linear and angular velocities
        base_linvel = self.data.qvel[0:3]
        base_angvel = self.data.qvel[3:6] * 0.1
        
        # 5. Foot ground contact states
        foot_sites = ["fl_foot", "fr_foot", "rl_foot", "rr_foot"]
        contacts = []
        feet_z = []
        for site_name in foot_sites:
            site_id = self.model.site(site_name).id
            z_pos = self.data.site_xpos[site_id][2]
            feet_z.append(z_pos)
            contacts.append(1.0 if z_pos < 0.012 else 0.0)
            
        # 6. Periodic Gait Phase Clock (Explicit coordination signal)
        t = self.current_step * 0.02
        phi = 2.0 * np.pi * self.gait_freq * t
        clock = np.array([np.sin(phi), np.cos(phi)], dtype=np.float64)
        
        obs = np.concatenate([
            q_norm,
            dq,
            euler,
            base_linvel,
            base_angvel,
            np.array(contacts, dtype=np.float64),
            np.array(feet_z, dtype=np.float64),
            np.array([self.data.qpos[2]], dtype=np.float64),  # base Z height
            clock,                                           # sin(phi), cos(phi)
            np.array([self.target_velocity], dtype=np.float64), # target velocity command
        ]).astype(np.float32)
        
        return obs

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        import mujoco

        self.current_step = 0
        self.prev_x = 0.0
        self.prev_action = np.zeros(8, dtype=np.float64)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Spawn standing at X=0, Y=0, Z=0.09
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.09
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        
        noise = self.np_random.uniform(-0.02, 0.02, size=8)
        initial_q = np.clip(self.stand_pose + noise, self.joint_min, self.joint_max)
        for i in range(8):
            self.data.qpos[7 + i] = initial_q[i]
            self.data.ctrl[i] = initial_q[i]
            
        if self.actuator_bank is not None:
            self.actuator_bank.reset(initial_q)
            
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        import mujoco

        action = np.asarray(action, dtype=np.float64)
        target_q = np.clip(
            self.stand_pose + action * self.action_scale,
            self.joint_min,
            self.joint_max,
        )
        
        substep_dt = self.model.opt.timestep
        for _ in range(self.frame_skip):
            if self.actuator_bank is not None:
                curr_q = self.data.qpos[7:15]
                curr_dq = self.data.qvel[6:14]
                eff_cmds, _ = self.actuator_bank.step(
                    target_q, curr_q, curr_dq, dt=substep_dt
                )
                self.data.ctrl[:] = eff_cmds
            else:
                self.data.ctrl[:] = target_q
            mujoco.mj_step(self.model, self.data)
            
        self.current_step += 1
        obs = self._get_obs()
        
        # ======================================================================
        # WALKING LOCOMOTION REWARD FUNCTION (CPG + PROGRESS + SPEED TRACKING)
        # ======================================================================
        curr_x = self.data.qpos[0]
        vx = self.data.qvel[0]  # Forward velocity (+X)
        vy = self.data.qvel[1]  # Lateral drift (Y)
        base_z = self.data.qpos[2]
        
        # 1. Forward Displacement Progress Reward (+X)
        delta_x = curr_x - self.prev_x
        self.prev_x = curr_x
        r_progress = 120.0 * delta_x
        
        # 2. Target Forward Speed Tracking (Bell-curve tracking)
        r_speed = 10.0 * np.exp(-((vx - self.target_velocity) ** 2) / 0.015)
        
        # 3. Lateral Drift Penalty (Keep in straight lane)
        r_drift = -4.0 * (vy ** 2)
        
        # 4. Upright & Heading Orientation Reward
        rot_mat = self.data.xmat[self.model.body("base_link").id].reshape(3, 3)
        upright_factor = rot_mat[2, 2]  # Cosine with vertical
        heading_factor = rot_mat[0, 0]  # Forward facing alignment
        r_upright = 3.0 * upright_factor if upright_factor > 0.7 else -12.0
        r_heading = 2.0 * max(0.0, heading_factor)
        
        # 5. Energy and Smoothness Penalties
        r_energy = -0.005 * np.sum(np.square(action))
        r_smooth = -0.01 * np.sum(np.square(action - self.prev_action))
        self.prev_action = action.copy()
        
        # 6. Survival Bonus
        r_alive = 0.5
        
        reward = r_progress + r_speed + r_drift + r_upright + r_heading + r_energy + r_smooth + r_alive
        
        # Termination conditions (Fall detection)
        terminated = False
        if base_z < 0.035 or upright_factor < 0.35:
            terminated = True
            reward -= 20.0  # Fall penalty
            
        truncated = bool(self.current_step >= self.max_episode_steps)
        
        info = {
            "forward_vel_mps": float(vx),
            "lateral_drift_mps": float(vy),
            "base_height_m": float(base_z),
            "x_displacement_m": float(curr_x),
            "upright_factor": float(upright_factor),
        }
        
        return obs, float(reward), terminated, truncated, info
