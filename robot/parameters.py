"""
Sesame Robot Parameters and Physical Constants.

All mechanical dimensions, joint ranges, servo channels, and default poses are
defined here directly based on the Sesame repository (dorianborian/sesame-robot)
and clearly categorized by source.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


# ==============================================================================
# 1. LINK DIMENSIONS & STRUCTURAL GEOMETRY [DIRECT REPOSITORY / CAD DATA]
# ==============================================================================

# Base body (Chassis) bounding dimensions in meters
BASE_LENGTH: float = 0.105   # Longitudinal (X axis) ~105 mm
BASE_WIDTH: float = 0.078    # Lateral (Y axis) ~78 mm
BASE_HEIGHT: float = 0.048   # Vertical (Z axis) ~48 mm

# Hip Joint Attachment Offsets relative to Base Center of Mass (m)
# [X (Forward), Y (Left/Right), Z (Up/Down)]
HIP_OFFSETS: Dict[str, np.ndarray] = {
    "FL": np.array([+0.036, +0.032, 0.000], dtype=np.float64),  # Front Left
    "FR": np.array([+0.036, -0.032, 0.000], dtype=np.float64),  # Front Right
    "RL": np.array([-0.036, +0.032, 0.000], dtype=np.float64),  # Rear Left
    "RR": np.array([-0.036, -0.032, 0.000], dtype=np.float64),  # Rear Right
}

# Link kinematic lengths (m)
FEMUR_LENGTH: float = 0.042   # L_femur: Distance between Hip axis and Knee axis (42 mm)
TIBIA_LENGTH: float = 0.046   # L_tibia: Distance between Knee axis and Ground contact tip (46 mm)


# ==============================================================================
# 2. MASS & INERTIAL PROPERTIES [INITIAL SIMULATION ASSUMPTIONS]
# ==============================================================================

MASS_BASE: float = 0.210     # Chassis + ESP32 + Battery + OLED + 4 Hip Servos (kg)
MASS_FEMUR: float = 0.022    # Femur bracket + 1 MG90S Knee Servo + Fasteners (kg)
MASS_TIBIA: float = 0.008    # Tibia lower leg print + contact tip (kg)
TOTAL_MASS: float = MASS_BASE + 4 * (MASS_FEMUR + MASS_TIBIA)  # ~0.330 kg


# ==============================================================================
# 3. JOINT IDENTIFIERS & MAPPINGS [DIRECT REPOSITORY DATA]
# ==============================================================================

# Firmware Servo Identifiers matching movement-sequences.h
# enum ServoName : uint8_t { R1 = 0, R2 = 1, L1 = 2, L2 = 3, R4 = 4, R3 = 5, L3 = 6, L4 = 7 };
FIRMWARE_SERVO_NAMES: List[str] = ["R1", "R2", "L1", "L2", "R4", "R3", "L3", "L4"]

# Canonical joint names in MuJoCo MJCF model
JOINT_NAMES: List[str] = [
    "fr_hip_joint",   # R1 (Index 0)
    "rr_hip_joint",   # R2 (Index 1)
    "fl_hip_joint",   # L1 (Index 2)
    "rl_hip_joint",   # L2 (Index 3)
    "rr_knee_joint",  # R4 (Index 4)
    "fr_knee_joint",  # R3 (Index 5)
    "fl_knee_joint",  # L3 (Index 6)
    "rl_knee_joint",  # L4 (Index 7)
]

# Servo Name to Index mapping dictionary
SERVO_MAPPING: Dict[str, int] = {name: idx for idx, name in enumerate(FIRMWARE_SERVO_NAMES)}


# Leg-grouped joint map
LEG_JOINTS: Dict[str, Tuple[str, str]] = {
    "FL": ("fl_hip_joint", "fl_knee_joint"),  # (L1, L3)
    "FR": ("fr_hip_joint", "fr_knee_joint"),  # (R1, R3)
    "RL": ("rl_hip_joint", "rl_knee_joint"),  # (L2, L4)
    "RR": ("rr_hip_joint", "rr_knee_joint"),  # (R2, R4)
}


# ==============================================================================
# 4. JOINT LIMITS & CALIBRATION [DIRECT REPOSITORY DATA]
# ==============================================================================

# Joint limits in degrees [min_deg, max_deg] from sesame_studio.py
JOINT_LIMITS_DEG: Dict[str, Tuple[float, float]] = {
    "fr_hip_joint": (45.0, 180.0),   # R1
    "rr_hip_joint": (0.0, 135.0),    # R2
    "fl_hip_joint": (0.0, 135.0),    # L1
    "rl_hip_joint": (45.0, 180.0),   # L2
    "rr_knee_joint": (0.0, 180.0),   # R4
    "fr_knee_joint": (0.0, 180.0),   # R3
    "fl_knee_joint": (0.0, 180.0),   # L3
    "rl_knee_joint": (0.0, 180.0),   # L4
}

# Joint limits in radians [min_rad, max_rad]
JOINT_LIMITS_RAD: Dict[str, Tuple[float, float]] = {
    k: (np.deg2rad(v[0]), np.deg2rad(v[1])) for k, v in JOINT_LIMITS_DEG.items()
}


# ==============================================================================
# 5. PRE-PROGRAMMED FIRMWARE POSES [DIRECT REPOSITORY DATA]
# ==============================================================================

# REST POSE: All servos at 90 deg (mechanical calibration midpoint)
REST_POSE_DEG: np.ndarray = np.array([90, 90, 90, 90, 90, 90, 90, 90], dtype=np.float64)
REST_POSE_RAD: np.ndarray = np.deg2rad(REST_POSE_DEG)

# STAND POSE: [R1: 135, R2: 45, L1: 45, L2: 135, R4: 0, R3: 180, L3: 0, L4: 180]
# NOTE: This is the FIRMWARE convention and does NOT produce standing in MuJoCo.
STAND_POSE_DEG: np.ndarray = np.array([135, 45, 45, 135, 0, 180, 0, 180], dtype=np.float64)
STAND_POSE_RAD: np.ndarray = np.deg2rad(STAND_POSE_DEG)

# MUJOCO STAND POSE: The MuJoCo XML ref=90° for all joints means at qpos=90°
# the geometry is at its nominal position (femur straight down, tibia straight down).
# This is the correct standing pose for the simulation.
# Order: [fr_hip, rr_hip, fl_hip, rl_hip, rr_knee, fr_knee, fl_knee, rl_knee]
MUJOCO_STAND_DEG: np.ndarray = np.array([90, 90, 90, 90, 90, 90, 90, 90], dtype=np.float64)
MUJOCO_STAND_RAD: np.ndarray = np.deg2rad(MUJOCO_STAND_DEG)



# ==============================================================================
# 6. ACTUATOR CHARACTERISTICS (MG90S 9G METAL GEAR) [INITIAL ASSUMPTIONS]
# ==============================================================================

@dataclass(frozen=True)
class MG90SSpecs:
    stall_torque_nm: float = 0.196      # 2.0 kg*cm @ 5.0V
    max_velocity_rad_s: float = 10.0    # ~0.10s / 60 deg -> ~10.47 rad/s
    time_constant_s: float = 0.020      # 1st-order response lag (20 ms)
    backlash_rad: float = 0.015         # Gear play (~0.86 deg)
    deadband_rad: float = 0.008         # Deadband (~0.46 deg)
    kp_gain: float = 4.5                # Position controller proportional gain
    kd_gain: float = 0.12               # Derivative damping gain
    voltage_nominal: float = 5.1        # Supply voltage (V)


ACTUATOR_SPECS = MG90SSpecs()
