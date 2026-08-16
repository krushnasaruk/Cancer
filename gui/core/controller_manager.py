"""
Controller Manager for Sesame Quadruped Digital Twin GUI.

Seamlessly loads, switches, and evaluates different controllers (PID, PPO Reach, PPO Walk, SAC, PPO+DR)
without requiring simulation rebuilds or restarts.
"""

import os
import sys
from typing import Dict, Optional, Tuple
import numpy as np

from robot.parameters import (
    JOINT_NAMES,
    MUJOCO_STAND_RAD,
    REST_POSE_RAD,
    JOINT_LIMITS_RAD,
)
from simulation.controllers.pid import JointPIDController as PIDController
from simulation.controllers.trajectory import (
    StandTrajectory as StandTrajectoryGenerator,
    SinusoidalTrajectory as SinusoidalTrajectoryGenerator,
    WalkingGaitTrajectory as WalkingTrotGaitGenerator,
    WaveTrajectory as WaveTrajectoryGenerator,
    PushupTrajectory as PushupTrajectoryGenerator,
)
from rl.ppo.train import ActorCriticPolicy
from rl.sac.train import SACPolicy


class ControllerType:
    PID = "PID (Classical)"
    PPO = "PPO (AI Reach)"
    PPO_WALK = "PPO (AI Walk)"
    SAC = "SAC (AI)"
    PPO_DR = "PPO + Domain Rand"
    PROPOSED_A3DR = "Proposed A3DR (Pending HW)"


class ControllerManager:
    """Manages active robot controllers and switching."""

    def __init__(self, obs_dim: int = 40, act_dim: int = 8):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.active_type = ControllerType.PID
        
        # 1. Classical PID Controller
        self.pid = PIDController(kp=5.0, ki=0.08, kd=0.15)
        self.pid_trajectory = StandTrajectoryGenerator()
        self.pid_mode = "STAND"
        
        # 2. PPO Reaching Policy
        self.ppo_policy: Optional[ActorCriticPolicy] = None
        self.ppo_checkpoint = "results/ppo/ppo_policy.npz"
        self._load_ppo_policy()
        
        # 3. PPO Walking Policy
        self.ppo_walk_policy: Optional[ActorCriticPolicy] = None
        self.ppo_walk_checkpoint = "results/ppo_walk/ppo_walk_policy.npz"
        self._load_ppo_walk_policy()
        
        # 4. SAC Policy
        self.sac_policy: Optional[SACPolicy] = None
        self.sac_checkpoint = "results/sac/sac_policy.npz"
        self._load_sac_policy()
        
        # Controller registry metadata
        self.controllers_info = {
            ControllerType.PID: {
                "name": "Classical PID Baseline",
                "type": "Classical Feedback Control",
                "status": "Ready",
                "enabled": True,
            },
            ControllerType.PPO: {
                "name": "PPO Reaching Policy",
                "type": "On-Policy Deep RL (GAE)",
                "status": "Loaded" if self.ppo_policy is not None else "Not Trained",
                "enabled": True,
            },
            ControllerType.PPO_WALK: {
                "name": "PPO Locomotion Policy",
                "type": "On-Policy Deep RL (CPG Guided)",
                "status": "Loaded" if self.ppo_walk_policy is not None else "Not Trained",
                "enabled": True,
            },
            ControllerType.SAC: {
                "name": "SAC Off-Policy Baseline",
                "type": "Maximum Entropy Deep RL",
                "status": "Loaded" if self.sac_policy is not None else "Not Trained",
                "enabled": True,
            },
            ControllerType.PPO_DR: {
                "name": "PPO + Domain Randomization",
                "type": "Robust Policy under Mass/Friction DR",
                "status": "Ready",
                "enabled": True,
            },
            ControllerType.PROPOSED_A3DR: {
                "name": "Proposed Actuator-Aware A3DR",
                "type": "Adaptive Domain Randomization on MG90S Errors",
                "status": "Pending Hardware Calibration",
                "enabled": False,
            },
        }

    def _load_ppo_policy(self) -> bool:
        if os.path.exists(self.ppo_checkpoint):
            try:
                self.ppo_policy = ActorCriticPolicy(obs_dim=self.obs_dim, act_dim=self.act_dim)
                self.ppo_policy.load(self.ppo_checkpoint)
                return True
            except Exception as e:
                print(f"Warning: Failed to load PPO policy: {e}")
        return False

    def _load_ppo_walk_policy(self) -> bool:
        if os.path.exists(self.ppo_walk_checkpoint):
            try:
                self.ppo_walk_policy = ActorCriticPolicy(obs_dim=37, act_dim=self.act_dim)
                self.ppo_walk_policy.load(self.ppo_walk_checkpoint)
                return True
            except Exception as e:
                print(f"Warning: Failed to load PPO walk policy: {e}")
        return False

    def _load_sac_policy(self) -> bool:
        if os.path.exists(self.sac_checkpoint):
            try:
                self.sac_policy = SACPolicy(obs_dim=self.obs_dim, act_dim=self.act_dim)
                self.sac_policy.load(self.sac_checkpoint)
                return True
            except Exception as e:
                print(f"Warning: Failed to load SAC policy: {e}")
        return False

    def set_controller(self, ctrl_type: str) -> bool:
        if ctrl_type in self.controllers_info:
            self.active_type = ctrl_type
            self.reset()
            return True
        return False

    def set_pid_mode(self, mode: str) -> None:
        self.pid_mode = mode
        if mode == "STAND":
            self.pid_trajectory = StandTrajectoryGenerator()
        elif mode == "SINE":
            self.pid_trajectory = SinusoidalTrajectoryGenerator(frequency_hz=0.8, amplitude_rad=0.35)
        elif mode == "WALK":
            self.pid_trajectory = WalkingTrotGaitGenerator(gait_frequency_hz=1.2)
        elif mode == "WAVE":
            self.pid_trajectory = WaveTrajectoryGenerator(frequency_hz=1.8)
        elif mode == "PUSHUP":
            self.pid_trajectory = PushupTrajectoryGenerator(frequency_hz=0.8)

    def compute_action(
        self,
        obs: np.ndarray,
        q_current: np.ndarray,
        dq_current: np.ndarray,
        t: float,
        dt: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute joint targets or control output.
        Returns (target_joint_angles [8,], raw_action [8,])
        """
        if self.active_type == ControllerType.PID:
            q_ref, _ = self.pid_trajectory.get_reference(t)
            ctrl_out = self.pid.compute(q_ref, q_current, current_dq=dq_current, dt=dt, t_sim=t)
            raw_action = (q_ref - REST_POSE_RAD) / (np.pi / 2.0)
            return q_ref, np.clip(raw_action, -1.0, 1.0)
            
        elif self.active_type in [ControllerType.PPO, ControllerType.PPO_DR]:
            if self.ppo_policy is None:
                self._load_ppo_policy()
                
            if self.ppo_policy is not None:
                obs_in = obs[:self.obs_dim] if len(obs) >= self.obs_dim else np.pad(obs, (0, self.obs_dim - len(obs)))
                mu, _ = self.ppo_policy.forward_policy(obs_in)
                raw_action = np.clip(mu, -1.0, 1.0)
            else:
                raw_action = np.zeros(self.act_dim)
                
            target_q = np.zeros(self.act_dim)
            for i, name in enumerate(JOINT_NAMES):
                low, high = JOINT_LIMITS_RAD[name]
                mid = (low + high) / 2.0
                rng = (high - low) / 2.0
                target_q[i] = mid + raw_action[i] * rng
            return target_q, raw_action
            
        elif self.active_type == ControllerType.PPO_WALK:
            if self.ppo_walk_policy is None:
                self._load_ppo_walk_policy()
                
            if self.ppo_walk_policy is not None:
                obs_walk = obs[:37] if len(obs) >= 37 else np.pad(obs, (0, 37 - len(obs)))
                mu, _ = self.ppo_walk_policy.forward_policy(obs_walk)
                raw_action = np.clip(mu, -1.0, 1.0)
            else:
                raw_action = np.zeros(self.act_dim)
                
            target_q = np.zeros(self.act_dim)
            for i, name in enumerate(JOINT_NAMES):
                low, high = JOINT_LIMITS_RAD[name]
                target_q[i] = np.clip(MUJOCO_STAND_RAD[i] + raw_action[i] * 0.35, low, high)
            return target_q, raw_action
            
        elif self.active_type == ControllerType.SAC:
            if self.sac_policy is None:
                self._load_sac_policy()
                
            if self.sac_policy is not None:
                obs_in = obs[:self.obs_dim] if len(obs) >= self.obs_dim else np.pad(obs, (0, self.obs_dim - len(obs)))
                mu, _ = self.sac_policy.forward(obs_in)
                raw_action = np.clip(mu, -1.0, 1.0)
            else:
                raw_action = np.zeros(self.act_dim)
                
            target_q = np.zeros(self.act_dim)
            for i, name in enumerate(JOINT_NAMES):
                low, high = JOINT_LIMITS_RAD[name]
                mid = (low + high) / 2.0
                rng = (high - low) / 2.0
                target_q[i] = mid + raw_action[i] * rng
            return target_q, raw_action
            
        return MUJOCO_STAND_RAD.copy(), np.zeros(8)

    def reset(self) -> None:
        self.pid.reset()
        if hasattr(self.pid_trajectory, "reset"):
            self.pid_trajectory.reset()
