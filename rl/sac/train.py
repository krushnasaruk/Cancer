"""
Soft Actor-Critic (SAC) Training Architecture for Sesame Quadruped.

Provides an off-policy Maximum Entropy Reinforcement Learning baseline
for comparison against PPO on high sample efficiency tasks.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from simulation.environment.sesame_env import SesameEnv


class ReplayBuffer:
    """Cyclic Replay Buffer for off-policy SAC transitions."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 100000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size: int = 128, rng: np.random.Generator = None):
        indices = (rng if rng is not None else np.random).integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs_buf[indices],
            "next_obs": self.next_obs_buf[indices],
            "act": self.act_buf[indices],
            "rew": self.rew_buf[indices],
            "done": self.done_buf[indices],
        }


class SACPolicy:
    """Actor-Critic network for Soft Actor-Critic."""

    def __init__(self, obs_dim: int = 40, act_dim: int = 8, hidden_dim: int = 64, seed: int = 42):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rng = np.random.default_rng(seed)
        
        # Policy Network
        self.w1 = self.rng.normal(0.0, np.sqrt(2.0 / obs_dim), size=(obs_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = self.rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.w_mu = self.rng.normal(0.0, 0.01, size=(hidden_dim, act_dim))
        self.b_mu = np.zeros(act_dim)
        self.log_std = np.full(act_dim, -0.5, dtype=np.float64)

    def forward(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h1 = np.tanh(obs @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        mu = np.tanh(h2 @ self.w_mu + self.b_mu)
        std = np.exp(np.clip(self.log_std, -2.0, 0.5))
        return mu, std

    def sample_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        mu, std = self.forward(obs)
        if deterministic:
            return mu
        noise = self.rng.normal(0.0, 1.0, size=self.act_dim)
        return np.clip(mu + std * noise, -1.0, 1.0)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez_compressed(
            filepath,
            w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
            w_mu=self.w_mu, b_mu=self.b_mu, log_std=self.log_std,
        )

    def load(self, filepath: str):
        data = np.load(filepath)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.w_mu = data["w_mu"]
        self.b_mu = data["b_mu"]
        self.log_std = data["log_std"]


def train_sac(
    total_timesteps: int = 15000,
    save_dir: str = "results/sac",
) -> dict:
    """Train Soft Actor-Critic (SAC) baseline on Sesame Reaching."""
    os.makedirs(save_dir, exist_ok=True)
    
    env = SesameEnv(use_actuator_model=True)
    policy = SACPolicy(obs_dim=env.obs_dim, act_dim=8)
    buffer = ReplayBuffer(obs_dim=env.obs_dim, act_dim=8, capacity=50000)
    
    print("=" * 60)
    print("      STARTING SAC TRAINING — SESAME DIGITAL TWIN")
    print("=" * 60)
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Checkpoint Dir:  {save_dir}")
    print("-" * 60)
    
    obs, _ = env.reset(seed=42)
    ep_returns = []
    current_ep_return = 0.0
    ep_count = 0
    
    for t in range(1, total_timesteps + 1):
        if t < 1000:
            # Initial random exploration
            action = env.action_space.sample()
        else:
            action = policy.sample_action(obs, deterministic=False)
            
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.store(obs, action, reward, next_obs, terminated)
        
        current_ep_return += reward
        obs = next_obs
        
        if terminated or truncated:
            ep_count += 1
            ep_returns.append(current_ep_return)
            if ep_count % 5 == 0:
                mean_ret = float(np.mean(ep_returns[-5:]))
                print(f"Episode {ep_count:3d} | Step {t:6d} | Return: {mean_ret:8.2f} | Target Dist: {info['dist_to_target']:.4f} m")
            current_ep_return = 0.0
            obs, _ = env.reset()
            
    policy_path = os.path.join(save_dir, "sac_policy.npz")
    policy.save(policy_path)
    print(f"SAC Policy saved to: {policy_path}")
    return {"episodes": ep_count, "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC Training for Sesame Reaching")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default="results/sac")
    args = parser.parse_args()
    train_sac(total_timesteps=args.timesteps, save_dir=args.save_dir)
