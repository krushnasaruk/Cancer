"""
Unit Tests for Sesame Kinematics Engine.
"""

import os
import sys
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from robot.parameters import (
    REST_POSE_RAD,
    STAND_POSE_RAD,
    FEMUR_LENGTH,
    TIBIA_LENGTH,
    HIP_OFFSETS,
    JOINT_NAMES,
)
from robot.kinematics import SesameKinematics


def test_forward_kinematics_rest_pose():
    """At 90 deg (pi/2) for both joints, leg should point straight down."""
    kin = SesameKinematics()
    fk = kin.forward_kinematics_all(REST_POSE_RAD)
    
    for leg, data in fk.items():
        p_hip = HIP_OFFSETS[leg]
        p_knee = data["knee"]
        p_foot = data["foot"]
        
        # Knee should be directly below hip along -Z by L1
        assert np.isclose(p_knee[0], p_hip[0], atol=1e-4), f"{leg} Knee X mismatch"
        assert np.isclose(p_knee[1], p_hip[1], atol=1e-4), f"{leg} Knee Y mismatch"
        assert np.isclose(p_knee[2], p_hip[2] - FEMUR_LENGTH, atol=1e-4), f"{leg} Knee Z mismatch"
        
        # Foot should be directly below knee along -Z by L2
        assert np.isclose(p_foot[0], p_hip[0], atol=1e-4), f"{leg} Foot X mismatch"
        assert np.isclose(p_foot[1], p_hip[1], atol=1e-4), f"{leg} Foot Y mismatch"
        assert np.isclose(p_foot[2], p_hip[2] - (FEMUR_LENGTH + TIBIA_LENGTH), atol=1e-4), f"{leg} Foot Z mismatch"


def test_jacobian_finite_difference():
    """Verify analytical Jacobian matches numerical finite differences."""
    kin = SesameKinematics()
    q_hip = 1.2
    q_knee = 1.0
    eps = 1e-6
    
    for leg in ["FL", "FR", "RL", "RR"]:
        J_analytic = kin.compute_jacobian_leg(leg, q_hip, q_knee)
        
        _, p0 = kin.forward_kinematics_leg(leg, q_hip, q_knee)
        _, p_hip_plus = kin.forward_kinematics_leg(leg, q_hip + eps, q_knee)
        _, p_knee_plus = kin.forward_kinematics_leg(leg, q_hip, q_knee + eps)
        
        J_num_hip = (p_hip_plus - p0) / eps
        J_num_knee = (p_knee_plus - p0) / eps
        
        assert np.allclose(J_analytic[:, 0], J_num_hip, atol=1e-4), f"{leg} Hip Jacobian mismatch"
        assert np.allclose(J_analytic[:, 1], J_num_knee, atol=1e-4), f"{leg} Knee Jacobian mismatch"


def test_inverse_kinematics_roundtrip():
    """Verify IK accurately recovers target position within workspace."""
    kin = SesameKinematics()
    q_hip_true = 1.0
    q_knee_true = 1.4
    
    for leg in ["FL", "FR", "RL", "RR"]:
        _, target_foot = kin.forward_kinematics_leg(leg, q_hip_true, q_knee_true)
        
        q_hip_sol, q_knee_sol, success = kin.inverse_kinematics_leg(
            leg, target_foot, initial_guess=(1.5, 1.5)
        )
        assert success, f"IK failed for leg {leg}"
        
        _, recovered_foot = kin.forward_kinematics_leg(leg, q_hip_sol, q_knee_sol)
        dist = np.linalg.norm(target_foot - recovered_foot)
        assert dist < 1e-3, f"IK target reconstruction error too large: {dist} m"


if __name__ == "__main__":
    test_forward_kinematics_rest_pose()
    test_jacobian_finite_difference()
    test_inverse_kinematics_roundtrip()
    print("ALL KINEMATICS UNIT TESTS PASSED!")
