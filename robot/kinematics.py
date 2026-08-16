"""
Sesame Robot Analytical Kinematics Engine.

Implements Forward Kinematics (FK), Inverse Kinematics (IK), and analytical
Jacobians for all 4 legs (FL, FR, RL, RR) grounded in the Sesame joint hierarchy.
"""

from typing import Dict, Tuple, Union
import numpy as np

from robot.parameters import (
    BASE_LENGTH,
    BASE_WIDTH,
    BASE_HEIGHT,
    HIP_OFFSETS,
    FEMUR_LENGTH,
    TIBIA_LENGTH,
    JOINT_NAMES,
    JOINT_LIMITS_RAD,
    LEG_JOINTS,
    STAND_POSE_RAD,
    REST_POSE_RAD,
)


class SesameKinematics:
    """Analytical Kinematics Solver for the 8-DOF Sesame Quadruped."""

    def __init__(
        self,
        femur_len: float = FEMUR_LENGTH,
        tibia_len: float = TIBIA_LENGTH,
        hip_offsets: Dict[str, np.ndarray] = None,
    ):
        self.L1 = femur_len
        self.L2 = tibia_len
        self.hip_offsets = hip_offsets if hip_offsets is not None else HIP_OFFSETS

    @staticmethod
    def rot_y(theta: float) -> np.ndarray:
        """3x3 Rotation matrix around Y-axis (Pitch)."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c,  0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c]
        ], dtype=np.float64)

    def forward_kinematics_leg(
        self, leg: str, q_hip: float, q_knee: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Forward Kinematics for a single leg.
        
        Args:
            leg: Leg identifier ('FL', 'FR', 'RL', 'RR')
            q_hip: Hip joint angle (rad)
            q_knee: Knee joint angle (rad)
            
        Returns:
            p_knee: 3D coordinates of knee joint in base frame [x, y, z] (m)
            p_foot: 3D coordinates of foot contact point in base frame [x, y, z] (m)
        """
        if leg not in self.hip_offsets:
            raise ValueError(f"Unknown leg identifier: {leg}. Must be one of {list(self.hip_offsets.keys())}")

        p_hip = self.hip_offsets[leg]

        # Knee position relative to hip (sagittal rotation)
        # At q_hip = pi/2 (90 deg), femur points straight down along -Z
        delta_knee_x = -self.L1 * np.cos(q_hip)
        delta_knee_y = 0.0
        delta_knee_z = -self.L1 * np.sin(q_hip)

        p_knee = p_hip + np.array([delta_knee_x, delta_knee_y, delta_knee_z], dtype=np.float64)

        # Foot position relative to knee
        # Cumulative angle of tibia in sagittal plane
        theta_tibia = q_hip + (q_knee - np.pi / 2.0)
        delta_foot_x = -self.L2 * np.cos(theta_tibia)
        delta_foot_y = 0.0
        delta_foot_z = -self.L2 * np.sin(theta_tibia)

        p_foot = p_knee + np.array([delta_foot_x, delta_foot_y, delta_foot_z], dtype=np.float64)

        return p_knee, p_foot

    def forward_kinematics_all(
        self, q: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute Forward Kinematics for all 4 legs given full joint angle vector.
        
        Args:
            q: 8-element joint angle vector (rad) ordered as JOINT_NAMES:
               [FR_hip, RR_hip, FL_hip, RL_hip, RR_knee, FR_knee, FL_knee, RL_knee]
               
        Returns:
            Dictionary mapping leg name -> {'knee': p_knee, 'foot': p_foot}
        """
        if len(q) != 8:
            raise ValueError(f"Expected 8 joint angles, got {len(q)}")

        # Extract angles matching JOINT_NAMES ordering
        # 0: fr_hip, 1: rr_hip, 2: fl_hip, 3: rl_hip, 4: rr_knee, 5: fr_knee, 6: fl_knee, 7: rl_knee
        leg_angles = {
            "FL": (q[2], q[6]),
            "FR": (q[0], q[5]),
            "RL": (q[3], q[7]),
            "RR": (q[1], q[4]),
        }

        results = {}
        for leg, (q_hip, q_knee) in leg_angles.items():
            p_knee, p_foot = self.forward_kinematics_leg(leg, q_hip, q_knee)
            results[leg] = {"knee": p_knee, "foot": p_foot}

        return results

    def get_feet_positions_array(self, q: np.ndarray) -> np.ndarray:
        """
        Return 4x3 array of foot positions in base frame ordered [FL, FR, RL, RR].
        """
        fk = self.forward_kinematics_all(q)
        return np.array([
            fk["FL"]["foot"],
            fk["FR"]["foot"],
            fk["RL"]["foot"],
            fk["RR"]["foot"],
        ], dtype=np.float64)

    def compute_jacobian_leg(self, leg: str, q_hip: float, q_knee: float) -> np.ndarray:
        """
        Compute 3x2 analytical Jacobian for a single leg foot position with respect to [q_hip, q_knee].
        
        Returns:
            J: 3x2 matrix where dp_foot = J @ [dq_hip, dq_knee]^T
        """
        theta_t = q_hip + (q_knee - np.pi / 2.0)
        
        # d(p_foot_x)/dq_hip = L1*sin(q_hip) + L2*sin(theta_t)
        # d(p_foot_x)/dq_knee = L2*sin(theta_t)
        # d(p_foot_z)/dq_hip = -L1*cos(q_hip) - L2*cos(theta_t)
        # d(p_foot_z)/dq_knee = -L2*cos(theta_t)
        
        dx_dhip = self.L1 * np.sin(q_hip) + self.L2 * np.sin(theta_t)
        dx_dknee = self.L2 * np.sin(theta_t)
        
        dy_dhip = 0.0
        dy_dknee = 0.0
        
        dz_dhip = -self.L1 * np.cos(q_hip) - self.L2 * np.cos(theta_t)
        dz_dknee = -self.L2 * np.cos(theta_t)
        
        return np.array([
            [dx_dhip, dx_dknee],
            [dy_dhip, dy_dknee],
            [dz_dhip, dz_dknee],
        ], dtype=np.float64)

    def inverse_kinematics_leg(
        self, leg: str, target_foot_pos: np.ndarray, initial_guess: Tuple[float, float] = (1.57, 1.57)
    ) -> Tuple[float, float, bool]:
        """
        Numerical Inverse Kinematics for a single leg to reach target foot position in base frame.
        
        Args:
            leg: 'FL', 'FR', 'RL', or 'RR'
            target_foot_pos: [x, y, z] target in base frame
            initial_guess: (q_hip_init, q_knee_init)
            
        Returns:
            q_hip, q_knee, success
        """
        q = np.array(initial_guess, dtype=np.float64)
        target = np.asarray(target_foot_pos, dtype=np.float64)
        
        hip_name, knee_name = LEG_JOINTS[leg]
        hip_lim = JOINT_LIMITS_RAD[hip_name]
        knee_lim = JOINT_LIMITS_RAD[knee_name]
        
        for _ in range(50):
            _, current_pos = self.forward_kinematics_leg(leg, q[0], q[1])
            err = target - current_pos
            if np.linalg.norm(err) < 1e-4:
                return float(q[0]), float(q[1]), True
                
            J = self.compute_jacobian_leg(leg, q[0], q[1])
            # Damped least squares
            J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(3))
            dq = J_pinv @ err
            q += np.clip(dq, -0.2, 0.2)
            q[0] = np.clip(q[0], hip_lim[0], hip_lim[1])
            q[1] = np.clip(q[1], knee_lim[0], knee_lim[1])

        _, current_pos = self.forward_kinematics_leg(leg, q[0], q[1])
        success = bool(np.linalg.norm(target - current_pos) < 2e-3)
        return float(q[0]), float(q[1]), success
