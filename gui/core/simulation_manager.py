"""
Simulation Manager Thread for Sesame Quadruped Digital Twin GUI.

Runs the 50 Hz control loop and 500 Hz MuJoCo physics decimation on a dedicated
worker thread, completely decoupled from the UI rendering thread.
"""

import os
import sys
import time
import threading
from typing import Dict, Optional, Tuple, Any
import numpy as np
import mujoco

from PyQt6.QtCore import QThread, pyqtSignal, QObject, QRecursiveMutex, QMutexLocker
from PyQt6.QtGui import QImage

from robot.parameters import (
    JOINT_NAMES,
    JOINT_LIMITS_RAD,
    STAND_POSE_RAD,
    REST_POSE_RAD,
)

MODEL_XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../simulation/model/sesame.xml"))
from robot.dynamics import SesameDynamics
from gui.core.robot_interface import SimulationRobot
from gui.core.controller_manager import ControllerManager, ControllerType
from gui.core.environment_presets import ENVIRONMENT_PRESETS, EnvironmentPreset


class SimState:
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ESTOP = "ESTOP"


class SimulationManager(QThread):
    """Background worker thread executing the MuJoCo physics and control loop."""

    # Qt Signals
    sig_frame_rendered = pyqtSignal(QImage)
    sig_telemetry_updated = pyqtSignal(dict)
    sig_status_changed = pyqtSignal(str)
    sig_experiment_finished = pyqtSignal(dict)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.mutex = QRecursiveMutex()
        self._is_alive = True
        self.state = SimState.STOPPED
        
        # Load MuJoCo Model and Data
        self.model_path = MODEL_XML_PATH
        if not os.path.isabs(self.model_path):
            self.model_path = os.path.abspath(self.model_path)
            
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize Robot and Controller Manager
        self.robot = SimulationRobot(self.model, self.data, use_actuator_model=True)
        self.controller_manager = ControllerManager(obs_dim=40, act_dim=8)
        self.active_preset = ENVIRONMENT_PRESETS["testing_arena"]
        self.active_preset.apply(self.model)
        
        # Timing & Decimation
        self.physics_dt = self.model.opt.timestep  # 0.002 s (500 Hz)
        self.control_dt = 0.020  # 50 Hz
        self.decimation = int(self.control_dt / self.physics_dt)  # 10 substeps
        self.time_scale = 1.0  # Speed slider
        
        # Single-stepping flag
        self._single_step_requested = False
        
        # Reaching Target & Mocap ID
        self.target_pos = np.array([0.10, 0.0, 0.02], dtype=np.float64)
        self.target_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        self.end_effector_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fl_foot")
        
        # Episode metrics tracking
        self.sim_time = 0.0
        self.step_count = 0
        self.episode_return = 0.0
        self.episode_steps = 0
        self.last_action = np.zeros(8, dtype=np.float32)
        
        # Rendering context
        self.viewport_width = 800
        self.viewport_height = 600
        self.renderer: Optional[mujoco.Renderer] = None
        self.camera = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self._setup_camera()
        
        # Initialize robot to standard standing stance & compute kinematics
        self.robot.reset(STAND_POSE_RAD)
        self._update_mocap_target()
        mujoco.mj_forward(self.model, self.data)
        
        # Performance benchmarking
        self._fps_counter = 0
        self._fps_timer = time.time()
        self.current_fps = 0.0
        self.current_steps_per_sec = 0.0
        self._step_counter = 0

    def _setup_camera(self):
        mujoco.mjv_defaultCamera(self.camera)
        mujoco.mjv_defaultOption(self.opt)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 0.36
        self.camera.elevation = -22.0
        self.camera.azimuth = 135.0
        self.camera.lookat = np.array([0.0, 0.0, 0.05], dtype=np.float64)

    def set_viewport_size(self, width: int, height: int):
        with QMutexLocker(self.mutex):
            self.viewport_width = max(100, width)
            self.viewport_height = max(100, height)
            if self.renderer is not None:
                del self.renderer
                self.renderer = None

    def start_sim(self):
        with QMutexLocker(self.mutex):
            if self.state != SimState.ESTOP:
                self.state = SimState.RUNNING
                self.sig_status_changed.emit(self.state)

    def pause_sim(self):
        with QMutexLocker(self.mutex):
            if self.state == SimState.RUNNING:
                self.state = SimState.PAUSED
                self.sig_status_changed.emit(self.state)

    def stop_sim(self):
        with QMutexLocker(self.mutex):
            self.state = SimState.STOPPED
            self.reset_sim()
            self.sig_status_changed.emit(self.state)

    def step_single(self):
        with QMutexLocker(self.mutex):
            self._single_step_requested = True
            if self.state == SimState.STOPPED:
                self.state = SimState.PAUSED
            self.sig_status_changed.emit(self.state)

    def emergency_stop(self):
        with QMutexLocker(self.mutex):
            self.state = SimState.ESTOP
            self.robot.emergency_stop()
            self.sig_status_changed.emit(self.state)

    def reset_sim(self):
        with QMutexLocker(self.mutex):
            self.robot.reset()
            self.controller_manager.reset()
            self.sim_time = 0.0
            self.step_count = 0
            self.episode_return = 0.0
            self.episode_steps = 0
            self.last_action[:] = 0.0
            
            # Reset target mocap
            self._update_mocap_target()
            mujoco.mj_forward(self.model, self.data)

    def set_controller(self, ctrl_type: str):
        with QMutexLocker(self.mutex):
            self.controller_manager.set_controller(ctrl_type)

    def set_pid_mode(self, mode: str):
        with QMutexLocker(self.mutex):
            self.controller_manager.set_pid_mode(mode)

    def set_environment_preset(self, preset_name: str):
        with QMutexLocker(self.mutex):
            if preset_name in ENVIRONMENT_PRESETS:
                self.active_preset = ENVIRONMENT_PRESETS[preset_name]
                self.active_preset.apply(self.model)

    def set_time_scale(self, scale: float):
        with QMutexLocker(self.mutex):
            self.time_scale = max(0.1, min(5.0, scale))

    def _update_mocap_target(self):
        if self.target_mocap_id != -1 and self.model.nmocap > 0:
            mocap_idx = self.model.body_mocapid[self.target_mocap_id]
            if mocap_idx >= 0:
                base_pos = self.robot.get_base_position()
                self.data.mocap_pos[mocap_idx] = base_pos + self.target_pos

    def _get_observation(self) -> np.ndarray:
        q = self.robot.get_joint_positions()
        dq = self.robot.get_joint_velocities()
        base_euler = self.robot.get_base_euler()
        lin_vel, ang_vel = self.robot.get_base_velocity()
        
        # Normalize q
        q_norm = np.zeros(8, dtype=np.float32)
        for i, name in enumerate(JOINT_NAMES):
            low, high = JOINT_LIMITS_RAD[name]
            q_norm[i] = (2.0 * (q[i] - low) / (high - low)) - 1.0
            
        dq_scaled = (dq / 10.0).astype(np.float32)
        base_pos = self.robot.get_base_position()
        
        feet_pos = self.robot.get_foot_positions()
        feet_rel = []
        for foot_name in ["fl_foot", "fr_foot", "rl_foot", "rr_foot"]:
            feet_rel.extend(feet_pos[foot_name] - base_pos)
            
        target_rel = self.target_pos.astype(np.float32)
        
        obs = np.concatenate([
            q_norm,
            dq_scaled,
            base_euler.astype(np.float32),
            lin_vel.astype(np.float32),
            ang_vel.astype(np.float32),
            np.array(feet_rel, dtype=np.float32),
            target_rel,
        ])
        return obs

    def _compute_reward(self, ee_pos: np.ndarray, world_target: np.ndarray, action: np.ndarray, q: np.ndarray) -> Tuple[float, float]:
        dist = float(np.linalg.norm(ee_pos - world_target))
        r_dist = -10.0 * dist + 5.0 * np.exp(-40.0 * (dist ** 2))
        r_ctrl = -0.01 * float(np.sum(action ** 2))
        r_smooth = -0.02 * float(np.sum((action - self.last_action) ** 2))
        
        # Upright bonus
        base_euler = self.robot.get_base_euler()
        tilt = np.sqrt(base_euler[0] ** 2 + base_euler[1] ** 2)
        r_upright = 2.0 * np.cos(tilt) if tilt < 0.6 else -5.0
        
        total_reward = r_dist + r_ctrl + r_smooth + r_upright
        return total_reward, dist

    def run(self):
        """Main physics & control loop."""
        last_time = time.perf_counter()
        render_timer = time.time()
        
        while self._is_alive:
            t0 = time.perf_counter()
            should_step = False
            
            with QMutexLocker(self.mutex):
                if self.state == SimState.RUNNING:
                    should_step = True
                elif self.state == SimState.PAUSED and self._single_step_requested:
                    should_step = True
                    self._single_step_requested = False
                    
            if should_step:
                with QMutexLocker(self.mutex):
                    q_curr = self.robot.get_joint_positions()
                    dq_curr = self.robot.get_joint_velocities()
                    obs = self._get_observation()
                    
                    # Compute control action
                    target_q, raw_action = self.controller_manager.compute_action(
                        obs, q_curr, dq_curr, self.sim_time, dt=self.control_dt
                    )
                    
                    # Apply targets to robot
                    self.robot.set_joint_targets(target_q, dt=self.control_dt)
                    
                    # Step MuJoCo physics with decimation
                    for _ in range(self.decimation):
                        mujoco.mj_step(self.model, self.data)
                        
                    self.sim_time += self.control_dt
                    self.step_count += 1
                    self._step_counter += 1
                    now_perf = time.perf_counter()
                    if now_perf - self._fps_timer >= 0.5:
                        self.current_steps_per_sec = self._step_counter / (now_perf - self._fps_timer)
                        self._step_counter = 0
                        self._fps_timer = now_perf
                    
                    # Update target marker position
                    self._update_mocap_target()
                    
                    # Compute foot reaching metric
                    ee_pos = self.data.site_xpos[self.end_effector_site_id].copy() if self.end_effector_site_id != -1 else np.zeros(3)
                    base_pos = self.robot.get_base_position()
                    world_target = base_pos + self.target_pos
                    reward, dist_to_target = self._compute_reward(ee_pos, world_target, raw_action, q_curr)
                    self.episode_return += reward
                    self.last_action = raw_action.copy()
                    
                    # Stability & COM
                    com_pos = self.data.subtree_com[1].copy() if self.model.nbody > 1 else np.zeros(3)
                    feet_pos = self.robot.get_foot_positions()
                    contacts_2d = {f: True for f in feet_pos}
                    feet_2d = {f: feet_pos[f][:2] for f in feet_pos}
                    ssm = SesameDynamics.compute_support_polygon_margin(contacts_2d, feet_2d, com_pos[:2])
                    
                    # Telemetry dictionary (emitted at ~25 Hz)
                    if self.step_count % 2 == 0:
                        telemetry = {
                            "sim_time": self.sim_time,
                            "step_count": self.step_count,
                            "episode_steps": self.episode_steps,
                            "state": self.state,
                            "controller": self.controller_manager.active_type,
                            "environment": self.active_preset.display_name,
                            "reward": reward,
                            "episode_return": self.episode_return,
                            "dist_to_target": dist_to_target,
                            "target_xyz": world_target.tolist(),
                            "ee_xyz": ee_pos.tolist(),
                            "base_xyz": base_pos.tolist(),
                            "base_euler_deg": np.degrees(self.robot.get_base_euler()).tolist(),
                            "base_lin_vel": self.robot.get_base_velocity()[0].tolist(),
                            "base_ang_vel": self.robot.get_base_velocity()[1].tolist(),
                            "com_xyz": com_pos.tolist(),
                            "ssm_margin": ssm,
                            "joint_positions_deg": np.degrees(q_curr).tolist(),
                            "joint_targets_deg": np.degrees(target_q).tolist(),
                            "joint_errors_deg": np.degrees(target_q - q_curr).tolist(),
                            "joint_velocities_deg": np.degrees(dq_curr).tolist(),
                            "joint_torques_nm": self.robot.get_joint_torques().tolist(),
                            "fps": self.current_fps,
                            "steps_per_sec": self.current_steps_per_sec,
                            "is_success": dist_to_target < 0.025,
                        }
                    else:
                        telemetry = None
                        
                if telemetry is not None:
                    self.sig_telemetry_updated.emit(telemetry)
                
            # Regulate loop timing
            elapsed = time.perf_counter() - t0
            target_period = (self.control_dt / self.time_scale) if should_step else 0.020
            sleep_time = target_period - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def stop_worker(self):
        self._is_alive = False
        self.wait(1000)
