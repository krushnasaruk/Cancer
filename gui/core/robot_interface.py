"""
Robot Interface Abstraction Layer for Sesame Quadruped.

Provides an abstract base class `RobotInterface` and two concrete implementations:
- `SimulationRobot`: Direct integration with MuJoCo MjModel/MjData and SesameActuatorBank.
- `HardwareRobot`: Concrete hardware driver stub for future real ESP32 deployment.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import numpy as np
import mujoco

from robot.parameters import (
    JOINT_NAMES,
    JOINT_LIMITS_RAD,
    MUJOCO_STAND_RAD,
    REST_POSE_RAD,
)
from calibration.actuator_model import SesameActuatorBank


class RobotInterface(ABC):
    """Abstract interface decoupling controllers/GUI from the underlying physical or simulated robot."""

    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Return current joint positions (8,) in radians."""
        pass

    @abstractmethod
    def get_joint_velocities(self) -> np.ndarray:
        """Return current joint velocities (8,) in rad/s."""
        pass

    @abstractmethod
    def get_joint_torques(self) -> np.ndarray:
        """Return applied joint torques (8,) in N*m."""
        pass

    @abstractmethod
    def get_base_position(self) -> np.ndarray:
        """Return base position [x, y, z] in world frame."""
        pass

    @abstractmethod
    def get_base_orientation(self) -> np.ndarray:
        """Return base orientation quaternion [w, x, y, z]."""
        pass

    @abstractmethod
    def get_base_euler(self) -> np.ndarray:
        """Return base orientation Euler angles [roll, pitch, yaw] in radians."""
        pass

    @abstractmethod
    def get_base_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return base linear velocity (3,) and angular velocity (3,)."""
        pass

    @abstractmethod
    def get_foot_positions(self) -> Dict[str, np.ndarray]:
        """Return Cartesian foot positions {name: [x, y, z]} in world frame."""
        pass

    @abstractmethod
    def set_joint_targets(self, targets: np.ndarray) -> None:
        """Send target joint positions (8,) in radians to actuators."""
        pass

    @abstractmethod
    def emergency_stop(self) -> None:
        """Trigger emergency stop: zero targets and lock robot safely."""
        pass

    @abstractmethod
    def reset(self, pose: Optional[np.ndarray] = None) -> None:
        """Reset robot state to initial or specified joint pose."""
        pass

    @abstractmethod
    def is_hardware(self) -> bool:
        """Return True if connected to physical hardware."""
        pass


class SimulationRobot(RobotInterface):
    """Concrete simulation robot wrapping MuJoCo physics and parametric actuator bank."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, use_actuator_model: bool = True):
        self.model = model
        self.data = data
        self.use_actuator_model = use_actuator_model
        
        self.actuators = SesameActuatorBank() if use_actuator_model else None
        self.is_estopped = False
        
        # Cache joint & site IDs
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        self.actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.replace("_joint", "_actuator")) for name in JOINT_NAMES]
        
        self.foot_site_ids = {}
        for site_name in ["fl_foot", "fr_foot", "rl_foot", "rr_foot"]:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid != -1:
                self.foot_site_ids[site_name] = sid
                
        self.last_targets = MUJOCO_STAND_RAD.copy()

    def get_joint_positions(self) -> np.ndarray:
        qpos_indices = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        return np.array([self.data.qpos[idx] for idx in qpos_indices], dtype=np.float64)

    def get_joint_velocities(self) -> np.ndarray:
        qvel_indices = [self.model.jnt_dofadr[jid] for jid in self.joint_ids]
        return np.array([self.data.qvel[idx] for idx in qvel_indices], dtype=np.float64)

    def get_joint_torques(self) -> np.ndarray:
        qvel_indices = [self.model.jnt_dofadr[jid] for jid in self.joint_ids]
        return np.array([self.data.qfrc_actuator[idx] for idx in qvel_indices], dtype=np.float64)

    def get_base_position(self) -> np.ndarray:
        return self.data.qpos[:3].copy()

    def get_base_orientation(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()  # [w, x, y, z]

    def get_base_euler(self) -> np.ndarray:
        quat = self.get_base_orientation()
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float64)

    def get_base_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        lin_vel = self.data.qvel[:3].copy()
        ang_vel = self.data.qvel[3:6].copy()
        return lin_vel, ang_vel

    def get_foot_positions(self) -> Dict[str, np.ndarray]:
        positions = {}
        for name, sid in self.foot_site_ids.items():
            positions[name] = self.data.site_xpos[sid].copy()
        return positions

    def set_joint_targets(self, targets: np.ndarray, dt: float = 0.02) -> None:
        if self.is_estopped:
            return
            
        self.last_targets = np.clip(
            targets,
            [JOINT_LIMITS_RAD[name][0] for name in JOINT_NAMES],
            [JOINT_LIMITS_RAD[name][1] for name in JOINT_NAMES],
        )
        
        if self.use_actuator_model and self.actuators is not None:
            current_q = self.get_joint_positions()
            current_dq = self.get_joint_velocities()
            applied_targets, _ = self.actuators.step(self.last_targets, current_q, current_dq, dt)
        else:
            applied_targets = self.last_targets
            
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = applied_targets[i]

    def emergency_stop(self) -> None:
        self.is_estopped = True
        # Zero control signals
        for aid in self.actuator_ids:
            self.data.ctrl[aid] = 0.0

    def reset(self, pose: Optional[np.ndarray] = None) -> None:
        self.is_estopped = False
        target_pose = pose if pose is not None else MUJOCO_STAND_RAD
        
        # Reset base position (drop from 9 cm)
        self.data.qpos[:3] = np.array([0.0, 0.0, 0.09])
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[:] = 0.0
        
        for i, jname in enumerate(JOINT_NAMES):
            jid = self.joint_ids[i]
            qpos_adr = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_adr] = target_pose[i]
            
        if self.use_actuator_model and self.actuators is not None:
            self.actuators.reset(target_pose)
            
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = target_pose[i]
            
        mujoco.mj_forward(self.model, self.data)

    def is_hardware(self) -> bool:
        return False


class HardwareRobot(RobotInterface):
    """Hardware driver stub for future physical Sesame quadruped ESP32 connection."""

    def __init__(self, port: str = "COM3", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.is_connected = False
        self.is_estopped = False
        self.joint_positions = MUJOCO_STAND_RAD.copy()
        self.joint_velocities = np.zeros(8)
        self.joint_torques = np.zeros(8)

    def get_joint_positions(self) -> np.ndarray:
        return self.joint_positions.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self.joint_velocities.copy()

    def get_joint_torques(self) -> np.ndarray:
        return self.joint_torques.copy()

    def get_base_position(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.05])

    def get_base_orientation(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0])

    def get_base_euler(self) -> np.ndarray:
        return np.zeros(3)

    def get_base_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.zeros(3)

    def get_foot_positions(self) -> Dict[str, np.ndarray]:
        return {
            "fl_foot": np.array([0.036, 0.032, 0.0]),
            "fr_foot": np.array([0.036, -0.032, 0.0]),
            "rl_foot": np.array([-0.036, 0.032, 0.0]),
            "rr_foot": np.array([-0.036, -0.032, 0.0]),
        }

    def set_joint_targets(self, targets: np.ndarray) -> None:
        if self.is_estopped:
            return
        self.joint_positions = targets.copy()

    def emergency_stop(self) -> None:
        self.is_estopped = True

    def reset(self, pose: Optional[np.ndarray] = None) -> None:
        self.is_estopped = False
        self.joint_positions = pose if pose is not None else MUJOCO_STAND_RAD.copy()

    def is_hardware(self) -> bool:
        return True
