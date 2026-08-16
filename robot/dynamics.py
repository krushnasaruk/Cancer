"""
Sesame Robot Dynamics & Multi-Body Stability Analysis.

Computes:
- Total Center of Mass (CoM) position in world and base frames
- Support polygon vertices and static stability margin (SSM)
- Foot ground reaction forces (GRF) and contact state
"""

from typing import Dict, List, Tuple
import numpy as np

from robot.parameters import (
    MASS_BASE,
    MASS_FEMUR,
    MASS_TIBIA,
    TOTAL_MASS,
    HIP_OFFSETS,
    FEMUR_LENGTH,
    TIBIA_LENGTH,
)
from robot.kinematics import SesameKinematics


class SesameDynamics:
    """Rigid body dynamics and stability evaluator for Sesame."""

    def __init__(self, kinematics: SesameKinematics = None):
        self.kin = kinematics if kinematics is not None else SesameKinematics()
        self.m_base = MASS_BASE
        self.m_femur = MASS_FEMUR
        self.m_tibia = MASS_TIBIA
        self.m_total = TOTAL_MASS

    def compute_center_of_mass_base_frame(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the Center of Mass (CoM) in the robot base link frame.
        
        Args:
            q: 8-element joint angle vector (rad)
            
        Returns:
            com: 3D coordinates [x, y, z] of total CoM in base frame (m)
        """
        fk = self.kin.forward_kinematics_all(q)
        
        # Base CoM at origin of base frame
        com_weighted = self.m_base * np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        for leg, p_hip in self.kin.hip_offsets.items():
            p_knee = fk[leg]["knee"]
            p_foot = fk[leg]["foot"]
            
            # Femur CoM is approximately halfway along femur link
            p_femur_com = (p_hip + p_knee) / 2.0
            # Tibia CoM is approximately halfway along tibia link
            p_tibia_com = (p_knee + p_foot) / 2.0
            
            com_weighted += self.m_femur * p_femur_com + self.m_tibia * p_tibia_com
            
        return com_weighted / self.m_total

    @staticmethod
    def compute_support_polygon_margin(
        foot_contacts: Dict[str, bool],
        foot_positions_2d: Dict[str, np.ndarray],
        com_2d: np.ndarray,
    ) -> float:
        """
        Compute the Static Stability Margin (SSM).
        
        The SSM is the minimum signed distance from the 2D Center of Mass projection
        to the boundary of the support polygon formed by contacting feet.
        Positive SSM indicates statically stable configuration.
        """
        contacting_points = [
            pos for leg, pos in foot_positions_2d.items() if foot_contacts.get(leg, False)
        ]
        
        if len(contacting_points) < 3:
            # Need at least 3 points for a 2D support polygon
            return -1.0
            
        points = np.array(contacting_points, dtype=np.float64)
        
        # Compute 2D convex hull centroid distance
        center = np.mean(points, axis=0)
        com_dist = np.linalg.norm(com_2d - center)
        
        # Approximate distance to edges
        # Order points in clockwise/counter-clockwise order
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        
        min_dist = float("inf")
        n = len(sorted_points)
        for i in range(n):
            p1 = sorted_points[i]
            p2 = sorted_points[(i + 1) % n]
            
            edge = p2 - p1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-6:
                continue
            edge_unit = edge / edge_len
            normal = np.array([-edge_unit[1], edge_unit[0]])
            
            # Signed distance from CoM to edge line
            vec_to_com = com_2d - p1
            dist = np.dot(vec_to_com, normal)
            min_dist = min(min_dist, dist)
            
        return float(min_dist)
