"""
Environment Presets System for Sesame Quadruped Digital Twin.

Provides modular environment configurations without breaking or altering
the core physical geometry of the robot.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import mujoco


class EnvironmentPreset:
    """Represents a configurable simulation environment preset."""

    def __init__(
        self,
        name: str,
        display_name: str,
        friction_tangential: float = 0.8,
        friction_torsional: float = 0.01,
        friction_rolling: float = 0.001,
        floor_rgba: List[float] = None,
        sky_rgb: List[float] = None,
        target_bounds_x: Tuple[float, float] = (0.01, 0.06),
        target_bounds_y: Tuple[float, float] = (0.01, 0.05),
        target_bounds_z: Tuple[float, float] = (-0.08, -0.02),
        description: str = "",
    ):
        self.name = name
        self.display_name = display_name
        self.friction = np.array([friction_tangential, friction_torsional, friction_rolling], dtype=np.float64)
        self.floor_rgba = floor_rgba if floor_rgba is not None else [0.2, 0.3, 0.4, 1.0]
        self.sky_rgb = sky_rgb if sky_rgb is not None else [0.1, 0.1, 0.15]
        self.target_bounds_x = target_bounds_x
        self.target_bounds_y = target_bounds_y
        self.target_bounds_z = target_bounds_z
        self.description = description

    def apply(self, model: mujoco.MjModel) -> None:
        """Apply friction, materials, and physics parameters to the MuJoCo model."""
        # Update floor friction (geom 0 is typically the floor)
        floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_geom_id != -1:
            model.geom_friction[floor_geom_id] = self.friction.copy()
            
        # Update foot friction for all 4 feet
        for foot_geom_name in ["fl_foot_geom", "fr_foot_geom", "rl_foot_geom", "rr_foot_geom"]:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, foot_geom_name)
            if gid != -1:
                model.geom_friction[gid] = self.friction.copy()


ENVIRONMENT_PRESETS: Dict[str, EnvironmentPreset] = {
    "testing_arena": EnvironmentPreset(
        name="testing_arena",
        display_name="Testing Arena (Standard)",
        friction_tangential=0.8,
        friction_torsional=0.01,
        friction_rolling=0.001,
        floor_rgba=[0.25, 0.25, 0.3, 1.0],
        description="Standard robotics testing arena with flat grid floor and nominal friction.",
    ),
    "laboratory": EnvironmentPreset(
        name="laboratory",
        display_name="Robotics Laboratory",
        friction_tangential=1.0,
        friction_torsional=0.02,
        friction_rolling=0.002,
        floor_rgba=[0.18, 0.22, 0.28, 1.0],
        description="Controlled indoor lab environment with high-grip epoxy flooring.",
    ),
    "office": EnvironmentPreset(
        name="office",
        display_name="Office Carpet",
        friction_tangential=0.55,
        friction_torsional=0.03,
        friction_rolling=0.01,
        floor_rgba=[0.35, 0.32, 0.28, 1.0],
        description="Low-friction commercial carpet setting with mild rolling resistance.",
    ),
    "outdoor": EnvironmentPreset(
        name="outdoor",
        display_name="Outdoor Asphalt",
        friction_tangential=0.95,
        friction_torsional=0.015,
        friction_rolling=0.005,
        floor_rgba=[0.15, 0.15, 0.15, 1.0],
        description="Rough high-friction outdoor pavement surface.",
    ),
    "uneven_terrain": EnvironmentPreset(
        name="uneven_terrain",
        display_name="Uneven Terrain",
        friction_tangential=0.75,
        friction_torsional=0.02,
        friction_rolling=0.008,
        floor_rgba=[0.4, 0.35, 0.25, 1.0],
        description="Variable-traction terrain testing stability and recovery.",
    ),
}
