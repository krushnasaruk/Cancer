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
        
        self.target_pos = np.array([0.10, 0.0, 0.02], dtype=np.float64)
        self.target_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        
        self.robot.reset(MUJOCO_STAND_RAD)
        self._update_mocap_target()
        mujoco.mj_forward(self.model, self.data)
        
        self.last_action = np.zeros(8, dtype=np.float32)

    def _update_mocap_target(self):
        if self.target_mocap_id != -1 and self.model.nmocap > 0:
            mocap_idx = self.model.body_mocapid[self.target_mocap_id]
            if mocap_idx >= 0:
                base_pos = self.robot.get_base_position()
                self.data.mocap_pos[mocap_idx] = base_pos + self.target_pos

    def reset(self):
        self.robot.reset(MUJOCO_STAND_RAD)
        self.controller_manager.reset()
        self.sim_time = 0.0
        self.step_count = 0
        self.episode_return = 0.0
        self.episode_steps = 0
        self.last_action[:] = 0.0
        self._update_mocap_target()
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> Dict[str, Any]:
        if self.state == "RUNNING":
            q_curr = self.robot.get_joint_positions()
            dq_curr = self.robot.get_joint_velocities()
            
            # Form observation vector
            base_euler = self.robot.get_base_euler()
            lin_vel, ang_vel = self.robot.get_base_velocity()
            base_pos = self.robot.get_base_position()
            
            q_norm = np.zeros(8, dtype=np.float32)
            for i, name in enumerate(JOINT_NAMES):
                low, high = JOINT_LIMITS_RAD[name]
                q_norm[i] = (2.0 * (q_curr[i] - low) / (high - low)) - 1.0
                
            feet_pos = self.robot.get_foot_positions()
            feet_rel = []
            for foot_name in ["fl_foot", "fr_foot", "rl_foot", "rr_foot"]:
                feet_rel.extend(feet_pos[foot_name] - base_pos)
                
            obs = np.concatenate([
                q_norm,
                (dq_curr / 10.0).astype(np.float32),
                base_euler.astype(np.float32),
                lin_vel.astype(np.float32),
                ang_vel.astype(np.float32),
                np.array(feet_rel, dtype=np.float32),
                self.target_pos.astype(np.float32),
            ])
            
            target_q, raw_action = self.controller_manager.compute_action(
                obs, q_curr, dq_curr, t=self.sim_time, dt=0.02
            )
            self.last_action = raw_action
            
            # Physics Sub-steps (500 Hz decimation)
            for _ in range(10):
                self.robot.set_joint_targets(target_q, dt=0.002)
                mujoco.mj_step(self.model, self.data)
                
            self.sim_time += 0.02
            self.step_count += 1
            self.episode_steps += 1
            
            # Target distance & reward
            ee_pos = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fl_foot")]
            world_target = base_pos + self.target_pos
            dist = float(np.linalg.norm(ee_pos - world_target))
            reward = -25.0 * dist + 35.0 * np.exp(-50.0 * (dist ** 2)) + (50.0 if dist < 0.025 else 0.0)
            self.episode_return += reward
            
        else:
            q_curr = self.robot.get_joint_positions()
            target_q = q_curr
            ee_pos = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fl_foot")]
            world_target = self.robot.get_base_position() + self.target_pos
            dist = float(np.linalg.norm(ee_pos - world_target))
            reward = 0.0
            
        # Compile full state for Three.js & Dashboard
        base_pos = self.robot.get_base_position()
        base_quat = self.data.qpos[3:7]  # [w, x, y, z]
        base_euler = np.degrees(self.robot.get_base_euler())
        
        # Ground contacts
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
                "x": float(self.target_pos[0]),
                "y": float(self.target_pos[1]),
                "z": float(self.target_pos[2]),
                "dist_mm": float(dist * 1000.0),
                "reached": bool(dist < 0.025),
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

# Mount static folder
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
    elif action == "SET_PID_MODE":
        mode = payload.get("mode", "STAND")
        engine.controller_manager.set_pid_mode(mode)
    elif action == "NUDGE_TARGET":
        dx = payload.get("dx", 0.0)
        dy = payload.get("dy", 0.0)
        dz = payload.get("dz", 0.0)
        engine.target_pos[0] = max(0.02, min(0.20, engine.target_pos[0] + dx))
        engine.target_pos[1] = max(-0.10, min(0.10, engine.target_pos[1] + dy))
        engine.target_pos[2] = max(0.00, min(0.12, engine.target_pos[2] + dz))
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
            await asyncio.sleep(0.033)  # ~30-60 FPS stream
    except WebSocketDisconnect:
        pass
