"""
MG90S Actuator Dynamics & Calibration Simulation Model.

STATUS: INITIAL SIMULATION ASSUMPTIONS
Note: The parameters below represent engineering approximations based on manufacturer
specifications and typical micro RC servo behavior. They will be replaced by direct
experimental dynamometer/encoder measurements from physical hardware calibration.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
import numpy as np


@dataclass
class MG90SActuatorConfig:
    """
    Configuration parameters for a single MG90S servo actuator.
    
    All default values are INITIAL SIMULATION ASSUMPTIONS.
    """
    # Name / Joint identifier
    name: str = "mg90s_servo"
    
    # Gearbox and Mechanical Limits
    torque_limit_nm: float = 0.196          # [INITIAL SIMULATION ASSUMPTION] Max stall torque ~2.0 kg*cm @ 5V
    velocity_limit_rad_s: float = 10.47     # [INITIAL SIMULATION ASSUMPTION] No-load speed ~0.10s / 60 deg
    gear_backlash_rad: float = 0.015        # [INITIAL SIMULATION ASSUMPTION] Mechanical gear play (~0.86 deg)
    deadband_rad: float = 0.008             # [INITIAL SIMULATION ASSUMPTION] Controller deadband (~0.46 deg)
    
    # Dynamic Response
    time_constant_s: float = 0.020          # [INITIAL SIMULATION ASSUMPTION] 1st-order response lag (20 ms)
    position_noise_std_rad: float = 0.005   # [INITIAL SIMULATION ASSUMPTION] Sensor / feedback noise (~0.28 deg)
    
    # Angle Operational Range (rad)
    angle_min_rad: float = 0.0              # Min angle (0 deg)
    angle_max_rad: float = np.pi            # Max angle (180 deg)
    
    # Internal Proportional-Derivative Tracking Gains (Simulated Servo Controller)
    kp: float = 4.5                         # Proportional tracking stiffness
    kd: float = 0.12                        # Derivative damping coefficient
    
    # Calibration Curve Polynomial Coefficients (default: identity mapping [0, 1])
    # real_angle = poly_c0 + poly_c1 * cmd_angle + poly_c2 * cmd_angle^2
    calibration_poly: Tuple[float, float, float] = (0.0, 1.0, 0.0)


class MG90SActuatorModel:
    """
    Simulates the non-ideal physical dynamics of a low-cost MG90S servo.
    
    Models:
    1. Command quantization & deadband
    2. 1st-order response delay filter
    3. Angular velocity saturation
    4. Torque saturation and back-EMF derating
    5. Gear backlash and hysteresis
    6. Sensor/potentiometer measurement noise
    """

    def __init__(self, config: Optional[MG90SActuatorConfig] = None, seed: Optional[int] = None):
        self.cfg = config if config is not None else MG90SActuatorConfig()
        self.rng = np.random.default_rng(seed)
        
        # Internal state
        self.filtered_command: float = (self.cfg.angle_min_rad + self.cfg.angle_max_rad) / 2.0
        self.current_position: float = self.filtered_command
        self.current_velocity: float = 0.0
        self.applied_torque: float = 0.0
        self.backlash_state: float = 0.0

    def reset(self, initial_position_rad: float = 1.57079):
        """Reset the internal actuator state."""
        self.filtered_command = float(initial_position_rad)
        self.current_position = float(initial_position_rad)
        self.current_velocity = 0.0
        self.applied_torque = 0.0
        self.backlash_state = 0.0

    def step(
        self,
        command_angle_rad: float,
        measured_joint_pos_rad: float,
        measured_joint_vel_rad_s: float,
        dt: float = 0.002,
    ) -> Tuple[float, float]:
        """
        Step the actuator model forward by timestep dt.
        
        Args:
            command_angle_rad: Target joint angle commanded by policy/controller (rad)
            measured_joint_pos_rad: Actual physical joint position from simulation (rad)
            measured_joint_vel_rad_s: Actual physical joint velocity from simulation (rad/s)
            dt: Simulation timestep (s)
            
        Returns:
            effective_target_rad: Filtered target angle to pass to physics engine (rad)
            computed_torque_nm: Actuator torque output (N*m)
        """
        # 1. Apply calibration polynomial curve
        c0, c1, c2 = self.cfg.calibration_poly
        calibrated_cmd = c0 + c1 * command_angle_rad + c2 * (command_angle_rad ** 2)
        
        # Clamp command to physical limits
        calibrated_cmd = np.clip(calibrated_cmd, self.cfg.angle_min_rad, self.cfg.angle_max_rad)
        
        # 2. Deadband check
        cmd_diff = calibrated_cmd - self.filtered_command
        if abs(cmd_diff) < self.cfg.deadband_rad:
            effective_cmd = self.filtered_command
        else:
            effective_cmd = calibrated_cmd - np.sign(cmd_diff) * self.cfg.deadband_rad
            
        # 3. 1st-order response lag: d(x)/dt = (cmd - x) / tau
        alpha = dt / (self.cfg.time_constant_s + dt)
        self.filtered_command += alpha * (effective_cmd - self.filtered_command)
        
        # 4. Velocity Limiting
        pos_error = self.filtered_command - self.current_position
        max_delta_pos = self.cfg.velocity_limit_rad_s * dt
        delta_pos = np.clip(pos_error, -max_delta_pos, max_delta_pos)
        self.current_position += delta_pos
        self.current_velocity = delta_pos / dt
        
        # 5. Gear Backlash / Hysteresis
        relative_disp = self.current_position - measured_joint_pos_rad
        if abs(relative_disp) > self.cfg.gear_backlash_rad:
            self.backlash_state = relative_disp - np.sign(relative_disp) * self.cfg.gear_backlash_rad
        else:
            self.backlash_state = 0.0
            
        # 6. Noise injection
        if self.cfg.position_noise_std_rad > 0:
            noise = self.rng.normal(0.0, self.cfg.position_noise_std_rad)
            output_pos = self.current_position + noise
        else:
            output_pos = self.current_position

        # 7. Internal Servo PD Controller Torque Computation
        tracking_error = output_pos - measured_joint_pos_rad
        torque = self.cfg.kp * tracking_error - self.cfg.kd * measured_joint_vel_rad_s
        
        # Torque Saturation
        self.applied_torque = float(np.clip(torque, -self.cfg.torque_limit_nm, self.cfg.torque_limit_nm))
        
        return float(output_pos), self.applied_torque


class SesameActuatorBank:
    """Manages all 8 MG90S actuators for the complete quadruped."""

    def __init__(self, seed: Optional[int] = None):
        from robot.parameters import JOINT_NAMES, JOINT_LIMITS_RAD
        
        self.actuators: Dict[str, MG90SActuatorModel] = {}
        for idx, name in enumerate(JOINT_NAMES):
            lim_min, lim_max = JOINT_LIMITS_RAD[name]
            cfg = MG90SActuatorConfig(
                name=name,
                angle_min_rad=lim_min,
                angle_max_rad=lim_max,
            )
            act_seed = (seed + idx) if seed is not None else None
            self.actuators[name] = MG90SActuatorModel(config=cfg, seed=act_seed)

    def reset(self, initial_pose_rad: Optional[np.ndarray] = None):
        """Reset all 8 actuators."""
        from robot.parameters import STAND_POSE_RAD, JOINT_NAMES
        pose = initial_pose_rad if initial_pose_rad is not None else STAND_POSE_RAD
        for idx, name in enumerate(JOINT_NAMES):
            self.actuators[name].reset(pose[idx])

    def step(
        self,
        command_angles_rad: np.ndarray,
        measured_joint_pos_rad: np.ndarray,
        measured_joint_vel_rad_s: np.ndarray,
        dt: float = 0.002,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step all 8 actuators.
        
        Returns:
            effective_commands: 8-element array of filtered target angles (rad)
            torques: 8-element array of output torques (N*m)
        """
        from robot.parameters import JOINT_NAMES
        effective_cmds = np.zeros(8, dtype=np.float64)
        torques = np.zeros(8, dtype=np.float64)
        
        for idx, name in enumerate(JOINT_NAMES):
            eff_cmd, tau = self.actuators[name].step(
                command_angle_rad=command_angles_rad[idx],
                measured_joint_pos_rad=measured_joint_pos_rad[idx],
                measured_joint_vel_rad_s=measured_joint_vel_rad_s[idx],
                dt=dt,
            )
            effective_cmds[idx] = eff_cmd
            torques[idx] = tau
            
        return effective_cmds, torques
