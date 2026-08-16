"""
Sesame Gymnasium Environment Verification Script.

Executes random action rollout stress tests and validates observation/action space integrity.
"""

import os
import sys
import time
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment.sesame_env import SesameEnv


def verify_environment(num_episodes: int = 5, steps_per_episode: int = 200) -> bool:
    """Run random action rollout stress test."""
    print("=" * 60)
    print("  VERIFYING SESAME GYMNASIUM ENVIRONMENT (RANDOM ACTIONS)")
    print("=" * 60)
    
    env = SesameEnv(use_actuator_model=True)
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space:      {env.action_space}")
    print(f"Control Timestep:  {env.dt:.4f} s ({1.0/env.dt:.1f} Hz)")
    print("-" * 60)
    
    total_steps = 0
    start_time = time.time()
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep * 100)
        
        # Check observation shape and finite values
        assert obs.shape == (env.obs_dim,), f"Obs shape mismatch: {obs.shape} != {(env.obs_dim,)}"
        assert np.isfinite(obs).all(), f"Non-finite values in initial obs: {obs}"
        
        ep_reward = 0.0
        for step in range(steps_per_episode):
            # Sample random uniform action in [-1, 1]
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            total_steps += 1
            
            # Assert finite observations and reward
            if not np.isfinite(obs).all():
                print(f"FAILED: Non-finite observation at ep {ep}, step {step}")
                return False
            if not np.isfinite(reward):
                print(f"FAILED: Non-finite reward at ep {ep}, step {step}")
                return False
                
            if terminated or truncated:
                break
                
        print(f"Episode {ep + 1}/{num_episodes} completed: Steps = {step + 1}, Return = {ep_reward:.2f}, Final Dist = {info['dist_to_target']:.4f} m")

    elapsed = time.time() - start_time
    fps = total_steps / elapsed
    print("-" * 60)
    print(f"SUCCESS: {total_steps} environment steps completed in {elapsed:.2f}s ({fps:.1f} steps/s).")
    print("Environment is numerically stable and ready for RL training.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
