"""
RL Environment factory and registration helper for Sesame.
"""

import gymnasium as gym
from gymnasium.envs.registration import register

from simulation.environment.sesame_env import SesameEnv


def register_sesame_envs():
    """Register Sesame environments with Gymnasium."""
    try:
        register(
            id="SesameReaching-v0",
            entry_point="simulation.environment.sesame_env:SesameEnv",
            max_episode_steps=500,
        )
    except gym.error.RegistrationError:
        pass  # Already registered


def make_sesame_env(render_mode=None, use_actuator_model=True, **kwargs):
    """Factory method to instantiate a Sesame environment."""
    return SesameEnv(
        render_mode=render_mode,
        use_actuator_model=use_actuator_model,
        **kwargs,
    )
