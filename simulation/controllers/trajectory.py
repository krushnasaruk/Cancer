"""
Trajectory Generators for Sesame Quadruped Experiments.

Implements reference trajectory generation for:
1. Static Poses (STAND, REST)
2. Sinusoidal Tracking (Actuator & PID Benchmark)
3. Quadruped Walking Gait Cycle (Trot Gait)
4. Wave Hand Animation (Expressive 3D Motion)
5. Pushup Animation (Core Kinematics Test)
"""

from typing import Callable, Tuple
import numpy as np

from robot.parameters import (
    MUJOCO_STAND_RAD,
    REST_POSE_RAD,
    JOINT_NAMES,
    JOINT_LIMITS_RAD,
)


class TrajectoryGenerator:
    """Base class for trajectory generators."""
    def get_reference(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (q_target, dq_target) at time t."""
        raise NotImplementedError


class StandTrajectory(TrajectoryGenerator):
    """Hold the MuJoCo standing pose (all joints at 90°).
    
    With ref=90° in the XML, qpos=90° means femur straight down and 
    tibia straight down — the robot stands on all four feet.
    """
    def __init__(self, transition_time: float = 1.0, initial_pose: np.ndarray = REST_POSE_RAD):
        self.t_trans = transition_time
        self.q_init = np.asarray(initial_pose, dtype=np.float64)
        self.q_stand = np.asarray(MUJOCO_STAND_RAD, dtype=np.float64)

    def get_reference(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        if t <= 0.0:
            return self.q_init.copy(), np.zeros(8, dtype=np.float64)
        if t >= self.t_trans:
            return self.q_stand.copy(), np.zeros(8, dtype=np.float64)
        
        # Quintic smooth step
        s = t / self.t_trans
        poly_s = 10 * (s**3) - 15 * (s**4) + 6 * (s**5)
        d_poly_s = (30 * (s**2) - 60 * (s**3) + 30 * (s**4)) / self.t_trans
        
        q = self.q_init + poly_s * (self.q_stand - self.q_init)
        dq = d_poly_s * (self.q_stand - self.q_init)
        return q, dq


class SinusoidalTrajectory(TrajectoryGenerator):
    """
    Multi-joint sinusoidal trajectory for tracking benchmark.
    Tests each joint around nominal stand position with configurable frequency and amplitude.
    """
    def __init__(
        self,
        nominal_pose: np.ndarray = MUJOCO_STAND_RAD,
        amplitude_rad: float = 0.25,
        frequency_hz: float = 1.0,
    ):
        self.nominal = np.asarray(nominal_pose, dtype=np.float64)
        self.amp = amplitude_rad
        self.freq = frequency_hz
        self.omega = 2.0 * np.pi * frequency_hz

    def get_reference(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        # Phase offsets between diagonal leg pairs
        phases = np.array([0.0, np.pi, np.pi, 0.0, np.pi, 0.0, np.pi, 0.0], dtype=np.float64)
        
        q = self.nominal + self.amp * np.sin(self.omega * t + phases)
        dq = self.amp * self.omega * np.cos(self.omega * t + phases)
        
        # Clamp to joint limits
        for idx, j_name in enumerate(JOINT_NAMES):
            low, high = JOINT_LIMITS_RAD[j_name]
            q[idx] = np.clip(q[idx], low, high)
            
        return q, dq


class WalkingGaitTrajectory(TrajectoryGenerator):
    """
    Open-loop periodic diagonal trot gait trajectory for MuJoCo simulation.
    
    JOINT_NAMES order: [fr_hip(0), rr_hip(1), fl_hip(2), rl_hip(3), 
                        rr_knee(4), fr_knee(5), fl_knee(6), rl_knee(7)]
    
    Diagonal pairs:
      Pair 1: FL (hip=2, knee=6) + RR (hip=1, knee=4) 
      Pair 2: FR (hip=0, knee=5) + RL (hip=3, knee=7)
    
    Hip joints swing forward/backward in the sagittal plane (Y-axis rotation).
    Knee joints lift the foot during swing phase.
    """
    def __init__(
        self,
        gait_frequency_hz: float = 1.2,
        hip_swing_amp_rad: float = 0.20,
        knee_lift_amp_rad: float = 0.25,
        ramp_time_s: float = 0.8,
    ):
        self.freq = gait_frequency_hz
        self.omega = 2.0 * np.pi * gait_frequency_hz
        self.hip_amp = hip_swing_amp_rad
        self.knee_amp = knee_lift_amp_rad
        self.ramp_time = ramp_time_s
        self.stand = MUJOCO_STAND_RAD.copy()

    def get_reference(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        phi = self.omega * t
        q = self.stand.copy()
        dq = np.zeros(8, dtype=np.float64)
        
        # Smooth ramp into gait from nominal stand pose
        ramp = min(1.0, max(0.0, t / self.ramp_time))
        
        # Diagonal Pair 1: FL (hip=2, knee=6) & RR (hip=1, knee=4)
        s1 = np.sin(phi)
        c1 = np.cos(phi)
        q[2] += ramp * self.hip_amp * s1   # FL hip swings forward/backward
        q[1] -= ramp * self.hip_amp * s1   # RR hip anti-phase
        # Knee lifts during swing (only positive half of cosine)
        knee_lift_1 = ramp * self.knee_amp * max(0.0, -c1)
        q[6] -= knee_lift_1  # FL knee: decrease angle = lift foot
        q[4] -= knee_lift_1  # RR knee: same
        
        # Diagonal Pair 2: FR (hip=0, knee=5) & RL (hip=3, knee=7) [phase shifted π]
        s2 = np.sin(phi + np.pi)
        c2 = np.cos(phi + np.pi)
        q[0] += ramp * self.hip_amp * s2   # FR hip
        q[3] -= ramp * self.hip_amp * s2   # RL hip anti-phase
        knee_lift_2 = ramp * self.knee_amp * max(0.0, -c2)
        q[5] -= knee_lift_2  # FR knee
        q[7] -= knee_lift_2  # RL knee
        
        # Enforce joint limits
        for idx, j_name in enumerate(JOINT_NAMES):
            low, high = JOINT_LIMITS_RAD[j_name]
            q[idx] = np.clip(q[idx], low, high)
            
        return q, dq


class WaveTrajectory(TrajectoryGenerator):
    """Expressive Sesame Waving animation (lifts Front-Left leg and waves)."""
    def __init__(self, frequency_hz: float = 2.0):
        self.freq = frequency_hz
        self.omega = 2.0 * np.pi * frequency_hz
        self.stand = MUJOCO_STAND_RAD.copy()

    def get_reference(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        q = self.stand.copy()
        # Front Left: Lift hip forward and wave knee
        q[2] = 0.6 + 0.15 * np.sin(self.omega * t)  # FL Hip swing
        q[6] = 1.8 + 0.30 * np.cos(self.omega * t)  # FL Knee wave
        # Stabilize tripod: shift weight slightly to other legs
        q[0] = 1.35  # FR Hip: lean slightly more to support
        q[5] = 1.2   # FR Knee: bend a bit more
        return q, np.zeros(8, dtype=np.float64)


class PushupTrajectory(TrajectoryGenerator):
    """Sesame Pushup workout animation — all knees bend/extend together."""
    def __init__(self, frequency_hz: float = 0.8):
        self.freq = frequency_hz
        self.omega = 2.0 * np.pi * frequency_hz
        self.stand = MUJOCO_STAND_RAD.copy()

    def get_reference(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        cycle = 0.5 * (1.0 - np.cos(self.omega * t))  # 0→1→0 smooth
        q = self.stand.copy()
        # Bend all knees: decrease knee angles to lower body
        q[4:8] -= 0.35 * cycle  # All 4 knees bend down
        # Keep hips stable
        return q, np.zeros(8, dtype=np.float64)
