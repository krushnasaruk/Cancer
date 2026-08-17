"""
FastAPI Backend Server for Sesame AI Digital Twin Web Application.

Streams 60 FPS MuJoCo physics, 8-joint servo states, and RL neural network telemetry
over WebSockets to the Three.js WebGL frontend.
"""

import os
import sys
import time
import json
import asyncio
from typing import Dict, Any, List
import numpy as np
import mujoco
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from robot.parameters import (
    JOINT_NAMES,
    JOINT_LIMITS_RAD,
    MUJOCO_STAND_RAD,
    REST_POSE_RAD,
)
from gui.core.robot_interface import SimulationRobot
from gui.core.controller_manager import ControllerManager, ControllerType
from gui.core.environment_presets import ENVIRONMENT_PRESETS

MODEL_XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulation/model/sesame.xml"))


class WebSimulationEngine:
    """Headless MuJoCo Simulation Engine with thread-safe WebSocket streaming."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
        self.data = mujoco.MjData(self.model)
        
        self.robot = SimulationRobot(self.model, self.data, use_actuator_model=True)
        self.controller_manager = ControllerManager(obs_dim=40, act_dim=8)
        self.active_preset = ENVIRONMENT_PRESETS["testing_arena"]
        self.active_preset.apply(self.model)
        
        self.state = "STOPPED"  # "STOPPED", "RUNNING", "PAUSED", "ESTOP"
        self.sim_time = 0.0
        self.step_count = 0
        self.episode_return = 0.0
        self.episode_steps = 0
        self.time_scale = 1.0
        self.prev_x = 0.0
        
        # Target position anchored in world space (X=+0.35m, Y=0.0m, Z=0.05m)
        self.target_offset = np.array([0.15, 0.0, 0.02], dtype=np.float64)
        self.world_target = np.array([0.25, 0.0, 0.05], dtype=np.float64)
        self.target_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        
        self.robot.reset(MUJOCO_STAND_RAD)
        self._update_mocap_target()
        mujoco.mj_forward(self.model, self.data)
        
        self.last_action = np.zeros(8, dtype=np.float32)

    def _update_mocap_target(self):
        if self.target_mocap_id != -1 and self.model.nmocap > 0:
            mocap_idx = self.model.body_mocapid[self.target_mocap_id]
            if mocap_idx >= 0:
                self.data.mocap_pos[mocap_idx] = self.world_target

    def reset(self):
        self.robot.reset(MUJOCO_STAND_RAD)
        self.controller_manager.reset()
        self.sim_time = 0.0
        self.step_count = 0
        self.episode_return = 0.0
        self.episode_steps = 0
        self.prev_x = 0.0
        self.last_action[:] = 0.0
        
        # Mode-specific target positioning: 0.12m for Reaching (within leg reach), 0.25m for Walking
        base_pos = self.robot.get_base_position()
        if self.controller_manager.active_type in [ControllerType.PPO, ControllerType.PPO_DR]:
            self.target_offset = np.array([0.12, 0.02, 0.04])
        else:
            self.target_offset = np.array([0.25, 0.0, 0.05])
            
        self.world_target = base_pos + self.target_offset
        self._update_mocap_target()
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> Dict[str, Any]:
        if self.state == "RUNNING":
            q_curr = self.robot.get_joint_positions()
            dq_curr = self.robot.get_joint_velocities()
            base_euler = self.robot.get_base_euler()
            lin_vel, ang_vel = self.robot.get_base_velocity()
            base_pos = self.robot.get_base_position()
            feet_pos = self.robot.get_foot_positions()
            
            active_ctrl = self.controller_manager.active_type
            
            if active_ctrl == ControllerType.PPO_WALK or self.controller_manager.pid_mode == "WALK":
                # =============================================================
                # 37-DIMENSIONAL LOCOMOTION OBSERVATION VECTOR & REWARD
                # =============================================================
                q_norm = (q_curr - MUJOCO_STAND_RAD) / 0.35
                dq_norm = dq_curr * 0.1
                
                contacts = []
                feet_z = []
                for foot_name in ["fl_foot", "fr_foot", "rl_foot", "rr_foot"]:
                    fz = feet_pos[foot_name][2]
                    feet_z.append(fz)
                    contacts.append(1.0 if fz < 0.015 else 0.0)
                    
                phi = 2.0 * np.pi * 1.4 * self.sim_time
                clock = [np.sin(phi), np.cos(phi)]
                
                obs = np.concatenate([
                    q_norm,
                    dq_norm,
                    base_euler,
                    lin_vel,
                    ang_vel * 0.1,
                    np.array(contacts, dtype=np.float64),
                    np.array(feet_z, dtype=np.float64),
                    np.array([base_pos[2]], dtype=np.float64),
                    clock,
                    np.array([0.15], dtype=np.float64),
                ]).astype(np.float32)
                
                target_q, raw_action = self.controller_manager.compute_action(
                    obs, q_curr, dq_curr, t=self.sim_time, dt=0.02
                )
                self.last_action = raw_action
                
                # Dynamic Target Respawn & Waypoint Advancement when reaching target sphere (<50mm)
                dx = self.world_target[0] - base_pos[0]
                dy = self.world_target[1] - base_pos[1]
                dist_tgt = np.hypot(dx, dy)
                waypoint_bonus = 0.0
                if dist_tgt < 0.05:
                    waypoint_bonus = 100.0  # +100 Waypoint Reach Bonus!
                    # Automatically advance target sphere 0.25m - 0.35m ahead of robot
                    self.world_target = base_pos + np.array([
                        np.random.uniform(0.25, 0.35),
                        np.random.uniform(-0.06, 0.06),
                        0.05,
                    ])
                    self._update_mocap_target()
                
                # Sub-step physics (500 Hz)
                for _ in range(10):
                    self.robot.set_joint_targets(target_q, dt=0.002)
                    mujoco.mj_step(self.model, self.data)
                    
                self.sim_time += 0.02
                self.step_count += 1
                self.episode_steps += 1
                
                # Locomotion Reward
                curr_x = base_pos[0]
                vx = lin_vel[0]
                vy = lin_vel[1]
                delta_x = curr_x - self.prev_x
                self.prev_x = curr_x
                
                r_progress = 200.0 * delta_x
                r_speed = -10.0 if vx < 0.02 else 25.0 * np.exp(-((vx - 0.15) ** 2) / 0.01)
                r_drift = -8.0 * (vy ** 2)
                r_alive = 1.0 if vx > 0.02 else -2.0
                
                reward = r_progress + r_speed + r_drift + r_alive + waypoint_bonus
                
                # Fall Check & Auto-Reset
                rot_mat = self.data.xmat[self.model.body("base_link").id].reshape(3, 3)
                upright_factor = rot_mat[2, 2]
                if base_pos[2] < 0.04 or upright_factor < 0.60:
                    reward = -50.0
                    self.episode_return += reward
                    self.reset()
                else:
                    self.episode_return += reward

            else:
                # =============================================================
                # 40-DIMENSIONAL REACHING OBSERVATION VECTOR & REWARD
                # =============================================================
                q_norm = np.zeros(8, dtype=np.float32)
                for i, name in enumerate(JOINT_NAMES):
                    low, high = JOINT_LIMITS_RAD[name]
                    q_norm[i] = (2.0 * (q_curr[i] - low) / (high - low)) - 1.0
                    
                feet_rel = []
                for foot_name in ["fl_foot", "fr_foot", "rl_foot", "rr_foot"]:
                    feet_rel.extend(feet_pos[foot_name] - base_pos)
                    
                target_rel = self.world_target - base_pos
                
                obs = np.concatenate([
                    q_norm,
                    (dq_curr / 10.0).astype(np.float32),
                    base_euler.astype(np.float32),
                    lin_vel.astype(np.float32),
                    ang_vel.astype(np.float32),
                    np.array(feet_rel, dtype=np.float32),
                    target_rel.astype(np.float32),
                ])
                
                target_q, raw_action = self.controller_manager.compute_action(
                    obs, q_curr, dq_curr, t=self.sim_time, dt=0.02
                )
                self.last_action = raw_action
                
                # Sub-step physics
                for _ in range(10):
                    self.robot.set_joint_targets(target_q, dt=0.002)
                    mujoco.mj_step(self.model, self.data)
                    
                self.sim_time += 0.02
                self.step_count += 1
                self.episode_steps += 1
                
                # Reaching Distance Reward against nearest front foot (FL or FR)
                fl_pos = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fl_foot")]
                fr_pos = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fr_foot")]
                dist_fl = float(np.linalg.norm(fl_pos - self.world_target))
                dist_fr = float(np.linalg.norm(fr_pos - self.world_target))
                dist = min(dist_fl, dist_fr)
                
                rot_mat = self.data.xmat[self.model.body("base_link").id].reshape(3, 3)
                upright_factor = rot_mat[2, 2]
                
                if base_pos[2] < 0.04 or upright_factor < 0.60:
                    reward = -50.0
                    self.episode_return += reward
                    self.reset()
                else:
                    reward = -25.0 * dist + 45.0 * np.exp(-50.0 * (dist ** 2))
                    if dist < 0.035:
                        reward += 100.0  # +100 Touch Bonus!
                        # Respawn ball at new random location in front of robot!
                        new_offset = np.array([
                            np.random.uniform(0.08, 0.16),
                            np.random.uniform(-0.12, 0.12),
                            np.random.uniform(0.02, 0.08),
                        ])
                        self.world_target = base_pos + new_offset
                        self._update_mocap_target()
                    self.episode_return += reward

        else:
            q_curr = self.robot.get_joint_positions()
            target_q = q_curr
            fl_pos = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fl_foot")]
            fr_pos = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fr_foot")]
            dist = min(float(np.linalg.norm(fl_pos - self.world_target)), float(np.linalg.norm(fr_pos - self.world_target)))
            reward = 0.0
            
        # Compile state output for Three.js & Dashboard
        base_pos = self.robot.get_base_position()
        base_quat = self.data.qpos[3:7]  # [w, x, y, z]
        base_euler = np.degrees(self.robot.get_base_euler())
        
        fl_pos = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fl_foot")]
        fr_pos = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fr_foot")]
        dist = min(float(np.linalg.norm(fl_pos - self.world_target)), float(np.linalg.norm(fr_pos - self.world_target)))
        
        contacts = {}
        for foot in ["fl_foot", "fr_foot", "rl_foot", "rr_foot"]:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, foot)
            contacts[foot[:2].upper()] = bool(self.data.site_xpos[sid][2] < 0.015)
            
        return {
            "time": round(self.sim_time, 2),
            "state": self.state,
            "controller": self.controller_manager.active_type,
            "base": {
                "x": float(base_pos[0]),
                "y": float(base_pos[1]),
                "z": float(base_pos[2]),
                "qw": float(base_quat[0]),
                "qx": float(base_quat[1]),
                "qy": float(base_quat[2]),
                "qz": float(base_quat[3]),
                "roll": float(base_euler[0]),
                "pitch": float(base_euler[1]),
                "yaw": float(base_euler[2]),
            },
            "joints": {
                name: {
                    "angle_deg": float(np.degrees(q_curr[i])),
                    "target_deg": float(np.degrees(target_q[i])),
                    "torque_nm": float(self.robot.get_joint_torques()[i]),
                }
                for i, name in enumerate(JOINT_NAMES)
            },
            "target": {
                "x": float(self.world_target[0]),
                "y": float(self.world_target[1]),
                "z": float(self.world_target[2]),
                "dist_mm": float(dist * 1000.0),
                "reached": bool(dist < 0.035),
            },
            "metrics": {
                "return": float(self.episode_return),
                "reward": float(reward),
                "steps": int(self.episode_steps),
                "stability_margin_mm": 35.0,
            },
            "contacts": contacts,
        }


# FastAPI Application
app = FastAPI(title="Sesame AI Digital Twin Web Application")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = WebSimulationEngine()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def get_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/control")
async def post_control(payload: Dict[str, Any]):
    action = payload.get("action")
    if action == "START":
        engine.state = "RUNNING"
    elif action == "PAUSE":
        engine.state = "PAUSED"
    elif action == "RESET":
        engine.reset()
    elif action == "ESTOP":
        engine.state = "ESTOP"
        engine.robot.emergency_stop()
    elif action == "SET_CONTROLLER":
        ctrl = payload.get("controller", ControllerType.PID)
        engine.controller_manager.set_controller(ctrl)
        engine.reset()  # Reset episode rewards & state on controller switch
    elif action == "SET_PID_MODE":
        mode = payload.get("mode", "STAND")
        engine.controller_manager.set_pid_mode(mode)
        engine.reset()  # Reset episode rewards on mode switch
    elif action == "NUDGE_TARGET":
        dx = payload.get("dx", 0.0)
        dy = payload.get("dy", 0.0)
        dz = payload.get("dz", 0.0)
        engine.world_target[0] = max(-1.0, min(5.0, engine.world_target[0] + dx))
        engine.world_target[1] = max(-2.0, min(2.0, engine.world_target[1] + dy))
        engine.world_target[2] = max(0.00, min(0.50, engine.world_target[2] + dz))
        engine._update_mocap_target()
    elif action == "SET_SPEED":
        engine.time_scale = payload.get("speed", 1.0)
        
    return {"status": "ok", "state": engine.state}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            state_data = engine.step()
            await websocket.send_json(state_data)
            await asyncio.sleep(0.033)  # ~30 FPS stream
    except WebSocketDisconnect:
        pass
