"""
Conventional PID Joint Controller for Sesame Quadruped.

Supports:
- Proportional, Integral, and Derivative gains (per joint or bank-wide)
- Anti-windup clamping
- First-order derivative low-pass filtering
- Joint limit enforcement
- Tracking telemetry logging for performance evaluation
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from robot.parameters import JOINT_NAMES, JOINT_LIMITS_RAD


class JointPIDController:
    """Multi-joint PID controller with anti-windup and error telemetry."""

    def __init__(
        self,
        kp: Union[float, np.ndarray] = 4.5,
        ki: Union[float, np.ndarray] = 0.05,
        kd: Union[float, np.ndarray] = 0.12,
        i_limit: float = 0.2,
        d_filter_coeff: float = 0.8,
        num_joints: int = 8,
    ):
        self.num_joints = num_joints
        
        # Gains
        self.kp = np.full(num_joints, kp, dtype=np.float64) if isinstance(kp, (int, float)) else np.asarray(kp, dtype=np.float64)
        self.ki = np.full(num_joints, ki, dtype=np.float64) if isinstance(ki, (int, float)) else np.asarray(ki, dtype=np.float64)
        self.kd = np.full(num_joints, kd, dtype=np.float64) if isinstance(kd, (int, float)) else np.asarray(kd, dtype=np.float64)
        
        self.i_limit = i_limit
        self.d_filter_coeff = d_filter_coeff
        
        # State
        self.integral = np.zeros(num_joints, dtype=np.float64)
        self.prev_error = np.zeros(num_joints, dtype=np.float64)
        self.filtered_d_error = np.zeros(num_joints, dtype=np.float64)
        
        # Telemetry History
        self.history: Dict[str, List[np.ndarray]] = {
            "time": [],
            "target_q": [],
            "current_q": [],
            "error_q": [],
            "output_cmd": [],
        }

    def reset(self):
        """Reset internal integrator and error states."""
        self.integral.fill(0.0)
        self.prev_error.fill(0.0)
        self.filtered_d_error.fill(0.0)
        for key in self.history:
            self.history[key].clear()

    def compute(
        self,
        target_q: np.ndarray,
        current_q: np.ndarray,
        current_dq: Optional[np.ndarray] = None,
        dt: float = 0.002,
        t_sim: float = 0.0,
    ) -> np.ndarray:
        """
        Compute control action for all joints.
        
        Args:
            target_q: 8-element target joint position vector (rad)
            current_q: 8-element measured joint position vector (rad)
            current_dq: Optional measured joint velocity vector (rad/s)
            dt: Timestep (s)
            t_sim: Current simulation timestamp (s)
            
        Returns:
            output_cmd: 8-element clamped command vector (rad or torque depending on actuator mode)
        """
        error = target_q - current_q
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup clamping
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.i_limit, self.i_limit)
        i_term = self.ki * self.integral
        
        # Derivative term
        if current_dq is not None:
            # Use negative measured velocity to avoid setpoint derivative kick
            d_term = -self.kd * current_dq
        else:
            raw_d = (error - self.prev_error) / dt
            self.filtered_d_error = (
                self.d_filter_coeff * self.filtered_d_error + (1.0 - self.d_filter_coeff) * raw_d
            )
            d_term = self.kd * self.filtered_d_error
            
        self.prev_error = error.copy()
        
        # Total output command (for position-mode servo, command is target_q + feedforward correction)
        output_cmd = target_q + (p_term + i_term + d_term) * 0.05
        
        # Enforce physical joint limits
        for idx, j_name in enumerate(JOINT_NAMES):
            low, high = JOINT_LIMITS_RAD[j_name]
            output_cmd[idx] = np.clip(output_cmd[idx], low, high)
            
        # Record Telemetry
        self.history["time"].append(t_sim)
        self.history["target_q"].append(target_q.copy())
        self.history["current_q"].append(current_q.copy())
        self.history["error_q"].append(error.copy())
        self.history["output_cmd"].append(output_cmd.copy())
        
        return output_cmd

    def get_telemetry_arrays(self) -> Dict[str, np.ndarray]:
        """Convert recorded telemetry history into NumPy arrays."""
        return {
            k: np.array(v, dtype=np.float64) for k, v in self.history.items()
        }
