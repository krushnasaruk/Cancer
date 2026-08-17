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
    """
    def __init__(
        self,
        gait_frequency_hz: float = 1.4,
        hip_swing_amp_rad: float = 0.25,   # ~14.3 deg hip swing
        knee_lift_amp_rad: float = 0.20,   # ~11.5 deg knee lift
        ramp_time_s: float = 0.5,
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
        
        s1 = np.sin(phi)
        c1 = np.cos(phi + 1.68 * np.pi)
        s2 = np.sin(phi + np.pi)
        c2 = np.cos(phi + 2.68 * np.pi)
        
        # Pair 1: FL(2) & RR(1)
        q[2] -= ramp * self.hip_amp * s1
        q[1] -= ramp * self.hip_amp * s1
        knee_1 = ramp * self.knee_amp * max(0.0, c1)
        q[6] -= knee_1
        q[4] -= knee_1
        
        # Pair 2: FR(0) & RL(3)
        q[0] -= ramp * self.hip_amp * s2
        q[3] -= ramp * self.hip_amp * s2
        knee_2 = ramp * self.knee_amp * max(0.0, c2)
        q[5] -= knee_2
        q[7] -= knee_2
        
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


class JumpTrajectory(TrajectoryGenerator):
    """
    Explosive Vertical Jump Trajectory:
    1. Crouch phase (0 -> 0.4s): Bend all 4 knees down
    2. Thrust phase (0.4s -> 0.6s): Explosive extension of hips/knees into mid-air jump
    3. Tuck/Flight phase (0.6s -> 0.9s): Airborne tuck
    4. Landing absorption phase (0.9s -> 1.4s): Bend knees to absorb landing impact
    """
    def __init__(self, period_s: float = 1.6):
        self.period = period_s
        self.stand = MUJOCO_STAND_RAD.copy()

    def get_reference(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        q = self.stand.copy()
        tau = t % self.period
        
        if tau < 0.4:
            # 1. Crouch down
            s = tau / 0.4
            crouch = 0.35 * np.sin(np.pi * s / 2.0)
            q[4:8] -= crouch  # Knees bend down
            q[0:4] += crouch * 0.5  # Hips adjust slightly back
        elif tau < 0.65:
            # 2. Explosive Thrust (Jump launch!)
            s = (tau - 0.4) / 0.25
            thrust = 0.45 * np.sin(np.pi * s)
            q[4:8] += thrust  # Explosive knee extension
            q[0:4] -= thrust * 0.4  # Explosive hip forward swing
        elif tau < 1.0:
            # 3. Flight Phase: Tuck legs
            q[4:8] -= 0.20
            q[0:4] += 0.10
        else:
            # 4. Landing Absorption
            s = (tau - 1.0) / 0.6
            absorb = 0.25 * np.sin(np.pi * s) * np.exp(-2.0 * s)
            q[4:8] -= absorb
            
        # Clamp limits
        for idx, j_name in enumerate(JOINT_NAMES):
            low, high = JOINT_LIMITS_RAD[j_name]
            q[idx] = np.clip(q[idx], low, high)
            
        return q, np.zeros(8, dtype=np.float64)


class HandshakeTrajectory(TrajectoryGenerator):
    """
    Friendly Quadruped Handshake / Paw Wave:
    1. Lift Front-Right (FR) leg high into mid-air
    2. Oscillate FR knee in a friendly handshake rhythm
    3. Maintain 3-leg tripod stance balance
    """
    def __init__(self, frequency_hz: float = 2.5):
        self.freq = frequency_hz
        self.omega = 2.0 * np.pi * frequency_hz
        self.stand = MUJOCO_STAND_RAD.copy()

    def get_reference(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        q = self.stand.copy()
        
        # Tripod balance adjustment (FL, RL, RR support weight)
        q[2] = 1.35  # FL hip support
        q[6] = 1.20  # FL knee support
        q[3] = 1.45  # RL hip support
        
        # Front Right (FR) Handshake / Paw Wave:
        # FR Hip (index 0): Swing leg up forward
        # FR Knee (index 5): Shake up and down
        wave_cycle = np.sin(self.omega * t)
        q[0] = 0.70 - 0.15 * wave_cycle  # FR Hip raised high
        q[5] = 1.90 + 0.35 * wave_cycle  # FR Knee shaking paw
        
        for idx, j_name in enumerate(JOINT_NAMES):
            low, high = JOINT_LIMITS_RAD[j_name]
            q[idx] = np.clip(q[idx], low, high)
            
        return q, np.zeros(8, dtype=np.float64)


class DanceTrajectory(TrajectoryGenerator):
    """
    Quadruped Rhythm Dance Performance:
    1. Side-to-side body roll sway
    2. Alternating diagonal paw tapping in tempo
    """
    def __init__(self, tempo_bpm: float = 120.0):
        self.freq = tempo_bpm / 60.0  # 2.0 Hz
        self.omega = 2.0 * np.pi * self.freq
        self.stand = MUJOCO_STAND_RAD.copy()

    def get_reference(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        q = self.stand.copy()
        phi = self.omega * t
        
        # Rhythmic hip sway (left/right roll)
        sway = 0.20 * np.sin(phi)
        q[0] -= sway  # FR hip
        q[1] -= sway  # RR hip
        q[2] += sway  # FL hip
        q[3] += sway  # RL hip
        
        # Alternating knee taps
        tap_1 = max(0.0, np.sin(2.0 * phi)) * 0.25
        tap_2 = max(0.0, -np.sin(2.0 * phi)) * 0.25
        
        q[6] -= tap_1  # FL knee tap
        q[4] -= tap_1  # RR knee tap
        q[5] -= tap_2  # FR knee tap
        q[7] -= tap_2  # RL knee tap
        
        for idx, j_name in enumerate(JOINT_NAMES):
            low, high = JOINT_LIMITS_RAD[j_name]
            q[idx] = np.clip(q[idx], low, high)
            
        return q, np.zeros(8, dtype=np.float64)


class RunGaitTrajectory(TrajectoryGenerator):
    """
    High-Speed Quadruped Bounding Trot Gait (RUN):
    - Frequency: 2.2 Hz (High cadence)
    - High hip amplitude (0.35 rad) and knee lift (0.30 rad)
    - Achieves >60 cm/s forward velocity
    """
    def __init__(
        self,
        gait_frequency_hz: float = 2.2,
        hip_swing_amp_rad: float = 0.35,
        knee_lift_amp_rad: float = 0.30,
        ramp_time_s: float = 0.3,
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
        ramp = min(1.0, max(0.0, t / self.ramp_time))
        
        s1 = np.sin(phi)
        c1 = np.cos(phi + 1.68 * np.pi)
        s2 = np.sin(phi + np.pi)
        c2 = np.cos(phi + 2.68 * np.pi)
        
        # Pair 1: FL(2) & RR(1)
        q[2] -= ramp * self.hip_amp * s1
        q[1] -= ramp * self.hip_amp * s1
        knee_1 = ramp * self.knee_amp * max(0.0, c1)
        q[6] -= knee_1
        q[4] -= knee_1
        
        # Pair 2: FR(0) & RL(3)
        q[0] -= ramp * self.hip_amp * s2
        q[3] -= ramp * self.hip_amp * s2
        knee_2 = ramp * self.knee_amp * max(0.0, c2)
        q[5] -= knee_2
        q[7] -= knee_2
        
        for idx, j_name in enumerate(JOINT_NAMES):
            low, high = JOINT_LIMITS_RAD[j_name]
            q[idx] = np.clip(q[idx], low, high)
            
        return q, np.zeros(8, dtype=np.float64)
