"""
Sesame Quadruped Gymnasium Reinforcement Learning Environment.

Provides a standard Gymnasium continuous-control environment for the Sesame digital twin,
featuring:
- Configurable reaching and trajectory tasks
- Modular reward computation (distance, energy, smoothness, stability)
- Integrated realistic MG90S actuator dynamics option
- Automatic observation normalization
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple
import numpy as np

import gymnasium as gym
from gymnasium import spaces

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from robot.parameters import (
    JOINT_NAMES,
    JOINT_LIMITS_RAD,
    MUJOCO_STAND_RAD,
    REST_POSE_RAD,
    HIP_OFFSETS,
)
from calibration.actuator_model import SesameActuatorBank


MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/sesame.xml"))


class SesameEnv(gym.Env):
    """
    Gymnasium Environment for Sesame Robot.
    
    Task: Reaching task where the robot maneuvers its front foot (or base) toward a target.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = MODEL_PATH,
        render_mode: Optional[str] = None,
        frame_skip: int = 10,
        use_actuator_model: bool = True,
        max_episode_steps: int = 500,
        target_reaching_mode: str = "foot_reaching",  # 'foot_reaching' or 'base_tracking'
    ):
        super().__init__()
        import mujoco

        self.xml_path = xml_path
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.use_actuator_model = use_actuator_model
        self.max_episode_steps = max_episode_steps
        self.target_reaching_mode = target_reaching_mode
        
        # Load MuJoCo model
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Model XML file not found at: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.dt = self.model.opt.timestep * self.frame_skip  # Control timestep (e.g. 0.002 * 10 = 0.02s / 50Hz)
        
        # Joint Limits & Scaling (ordered by JOINT_NAMES)
        self.joint_min = np.array([JOINT_LIMITS_RAD[j][0] for j in JOINT_NAMES], dtype=np.float64)
        self.joint_max = np.array([JOINT_LIMITS_RAD[j][1] for j in JOINT_NAMES], dtype=np.float64)
        self.joint_mid = (self.joint_min + self.joint_max) / 2.0
        self.joint_range = (self.joint_max - self.joint_min) / 2.0
        
        # Cache joint index maps for JOINT_NAMES order
        self.qpos_indices = np.array([self.model.joint(name).qposadr[0] for name in JOINT_NAMES], dtype=np.int32)
        self.qvel_indices = np.array([self.model.joint(name).dofadr[0] for name in JOINT_NAMES], dtype=np.int32)
        self.act_indices = np.array([self.model.actuator(name.replace("_joint", "_actuator")).id for name in JOINT_NAMES], dtype=np.int32)
        
        # Actuator bank
        self.actuator_bank = SesameActuatorBank() if use_actuator_model else None
        
        # Action space: 8 continuous values in [-1.0, 1.0] mapping linearly to [joint_min, joint_max]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        # Observation space breakdown:
        # - Joint positions normalized [-1, 1]: (8,)
        # - Joint velocities (rad/s): (8,)
        # - Base orientation quaternion (4,) or roll, pitch, yaw (3,)
        # - Base linear velocity (3,)
        # - Base angular velocity (3,)
        # - 4 Foot positions relative to base (4 * 3 = 12,)
        # - Target position relative to base (3,)
        # Total observation dimension = 8 + 8 + 3 + 3 + 3 + 12 + 3 = 40
        self.obs_dim = 40
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Internal episode tracking
        self.current_step = 0
        self.prev_action = np.zeros(8, dtype=np.float64)
        self.target_pos = np.array([0.08, 0.04, 0.02], dtype=np.float64)
        
        # Viewer handle
        self.viewer = None

    def _action_to_joint_targets(self, action: np.ndarray) -> np.ndarray:
        """Map normalized action [-1, 1] to physical joint angles (rad)."""
        action = np.clip(action, -1.0, 1.0)
        return self.joint_mid + action * self.joint_range

    def _get_foot_positions_world(self) -> np.ndarray:
        """Get 4x3 world coordinates of all 4 foot sites [FL, FR, RL, RR]."""
        foot_sites = ["fl_foot", "fr_foot", "rl_foot", "rr_foot"]
        positions = []
        for site_name in foot_sites:
            site_id = self.model.site(site_name).id
            positions.append(self.data.site_xpos[site_id].copy())
        return np.array(positions, dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
        """Construct the observation vector."""
        # 1. Joint positions normalized
        q_raw = self.data.qpos[self.qpos_indices]
        q_norm = (q_raw - self.joint_mid) / self.joint_range
        
        # 2. Joint velocities
        dq = self.data.qvel[self.qvel_indices]
        
        # 3. Base orientation (Euler angles roll, pitch, yaw)
        base_quat = self.data.qpos[3:7]  # [w, x, y, z]
        # Convert quat to euler
        w, x, y, z = base_quat
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        euler = np.array([roll, pitch, yaw], dtype=np.float64)
        
        # 4. Base linear and angular velocity in base frame
        base_linvel = self.data.qvel[0:3]
        base_angvel = self.data.qvel[3:6]
        
        # 5. Foot positions relative to base
        base_pos = self.data.qpos[0:3]
        feet_world = self._get_foot_positions_world()
        feet_rel = (feet_world - base_pos).flatten()
        
        # 6. Target position relative to base
        target_rel = self.target_pos - base_pos
        
        obs = np.concatenate([
            q_norm,
            dq * 0.1,         # Scale velocity
            euler,
            base_linvel,
            base_angvel * 0.1,
            feet_rel,
            target_rel,
        ]).astype(np.float32)
        
        return obs

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        import mujoco

        self.current_step = 0
        self.prev_action = np.zeros(8, dtype=np.float64)
        
        # Reset physics data
        mujoco.mj_resetData(self.model, self.data)
        
        # Initial robot position: spawn slightly above ground
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.095  # 95 mm ground clearance
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
        
        # Set joint angles to STAND pose with slight random perturbation
        noise = self.np_random.uniform(-0.05, 0.05, size=8)
        initial_q = np.clip(MUJOCO_STAND_RAD + noise, self.joint_min, self.joint_max)
        for i in range(8):
            self.data.qpos[self.qpos_indices[i]] = initial_q[i]
        self.data.ctrl[self.act_indices] = initial_q
            
        # Reset actuator bank
        if self.actuator_bank is not None:
            self.actuator_bank.reset(initial_q)
            
        # Sample random target position in reach volume
        self.target_pos = np.array([
            self.np_random.uniform(0.05, 0.12),
            self.np_random.uniform(-0.06, 0.06),
            self.np_random.uniform(0.01, 0.06),
        ], dtype=np.float64)
        
        # Update target marker visual in MuJoCo mocap
        target_mocap_id = self.model.body("target_marker").mocapid[0]
        if target_mocap_id >= 0:
            self.data.mocap_pos[target_mocap_id] = self.target_pos
            
        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        import mujoco

        action = np.asarray(action, dtype=np.float64)
        target_joint_angles = self._action_to_joint_targets(action)
        
        # Step physics sub-steps with frame skipping
        substep_dt = self.model.opt.timestep
        for _ in range(self.frame_skip):
            if self.actuator_bank is not None:
                curr_q = self.data.qpos[self.qpos_indices]
                curr_dq = self.data.qvel[self.qvel_indices]
                eff_cmds, _ = self.actuator_bank.step(
                    target_joint_angles, curr_q, curr_dq, dt=substep_dt
                )
                self.data.ctrl[self.act_indices] = eff_cmds
            else:
                self.data.ctrl[self.act_indices] = target_joint_angles
                
            mujoco.mj_step(self.model, self.data)
            
        self.current_step += 1
        obs = self._get_obs()
        
        # ======================================================================
        # MODULAR REWARD FUNCTION
        # ======================================================================
        # 1. Distance Reward (Reaching distance from nearest front foot FL/FR to target)
        fl_pos = self.data.site_xpos[self.model.site("fl_foot").id]
        fr_pos = self.data.site_xpos[self.model.site("fr_foot").id]
        dist_fl = np.linalg.norm(fl_pos - self.target_pos)
        dist_fr = np.linalg.norm(fr_pos - self.target_pos)
        dist_to_target = float(min(dist_fl, dist_fr))
        
        r_dist = -25.0 * dist_to_target + 45.0 * np.exp(-50.0 * (dist_to_target**2))
        if dist_to_target < 0.035:
            r_dist += 100.0  # +100 precision touch bonus!
            # Auto-respawn target to a new random location in front of robot
            base_p = self.data.qpos[0:3]
            self.target_pos = base_p + np.array([
                self.np_random.uniform(0.08, 0.16),
                self.np_random.uniform(-0.10, 0.10),
                self.np_random.uniform(0.02, 0.08),
            ], dtype=np.float64)
            target_mocap_id = self.model.body("target_marker").mocapid[0]
            if target_mocap_id >= 0:
                self.data.mocap_pos[target_mocap_id] = self.target_pos
        
        # 2. Control Energy Penalty
        r_ctrl = -0.01 * np.sum(np.square(action))
        
        # 3. Smoothness / Action Rate Penalty
        r_smooth = -0.02 * np.sum(np.square(action - self.prev_action))
        self.prev_action = action.copy()
        
        # 4. Joint Limit Penalty
        curr_q = self.data.qpos[7:15]
        limit_violations = np.maximum(0.0, self.joint_min - curr_q) + np.maximum(0.0, curr_q - self.joint_max)
        r_limit = -5.0 * np.sum(limit_violations)
        
        # 5. Stability & Upright Bonus
        base_z = self.data.qpos[2]
        base_rot_z = self.data.xmat[self.model.body("base_link").id].reshape(3, 3)[2, 2]
        r_upright = 2.0 * base_rot_z if base_rot_z > 0.6 else -5.0
        
        reward = r_dist + r_ctrl + r_smooth + r_limit + r_upright
        
        # ======================================================================
        # TERMINATION CONDITIONS
        # ======================================================================
        terminated = False
        # Terminate if robot falls over (Base Z too low or tilted upside down)
        if base_z < 0.035 or base_rot_z < 0.3:
            terminated = True
            reward -= 20.0
            
        truncated = bool(self.current_step >= self.max_episode_steps)
        
        info = {
            "dist_to_target": float(dist_to_target),
            "base_height": float(base_z),
            "upright_factor": float(base_rot_z),
            "reward_dist": float(r_dist),
            "reward_ctrl": float(r_ctrl),
            "reward_smooth": float(r_smooth),
        }
        
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            import mujoco.viewer
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
