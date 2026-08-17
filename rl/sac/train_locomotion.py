"""
Soft Actor-Critic (SAC) Locomotion Training Architecture for Sesame Quadruped.

Provides an off-policy Maximum Entropy Reinforcement Learning locomotion trainer
that optimizes joint motor policies for autonomous forward walking and posture stability.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Tuple
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from simulation.environment.sesame_walk_env import SesameWalkEnv
from rl.sac.train import ReplayBuffer, SACPolicy


def train_sac_locomotion(
    total_timesteps: int = 100000,
    steps_per_epoch: int = 2000,
    batch_size: int = 128,
    lr: float = 3e-4,
    gamma: float = 0.99,
    alpha: float = 0.2,  # Entropy temperature
    save_dir: str = "results/sac_walk",
    seed: int = 42,
) -> dict:
    """Train Soft Actor-Critic (SAC) autonomous locomotion policy."""
    os.makedirs(save_dir, exist_ok=True)
    env = SesameWalkEnv(use_actuator_model=True)
    policy = SACPolicy(obs_dim=env.obs_dim, act_dim=8, seed=seed)
    buffer = ReplayBuffer(obs_dim=env.obs_dim, act_dim=8, capacity=100000)
    
    num_epochs = max(1, total_timesteps // steps_per_epoch)
    
    print("=" * 60)
    print("   STARTING SAC LOCOMOTION TRAINING — SESAME ROBOT")
    print("=" * 60)
    print(f"Task:            SAC Off-Policy Quadruped Walking")
    print(f"Total Timesteps: {total_timesteps} ({num_epochs} epochs of {steps_per_epoch} steps)")
    print(f"Entropy Temp:    alpha={alpha}")
    print(f"Checkpoint Dir:  {save_dir}")
    print("-" * 60)
    
    training_log = {
        "epoch": [],
        "mean_episode_return": [],
        "mean_forward_velocity_mps": [],
        "mean_x_displacement_m": [],
        "total_timesteps": [],
    }
    
    t_global = 0
    obs, _ = env.reset(seed=seed)
    current_ep_return = 0.0
    ep_returns = []
    ep_vels = []
    ep_disps = []
    
    for epoch in range(1, num_epochs + 1):
        for step in range(steps_per_epoch):
            # Sample action with exploration noise
            action = policy.sample_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            buffer.store(obs, action, reward, next_obs, done)
            current_ep_return += reward
            t_global += 1
            
            # Off-policy policy update
            if buffer.size >= batch_size:
                batch = buffer.sample_batch(batch_size, rng=policy.rng)
                policy.update(batch, lr=lr)
                
            if done:
                ep_returns.append(current_ep_return)
                ep_vels.append(info.get("forward_vel_mps", 0.0))
                ep_disps.append(info.get("x_displacement_m", 0.0))
                current_ep_return = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs
                
        mean_ret = float(np.mean(ep_returns[-10:])) if ep_returns else 0.0
        mean_vel = float(np.mean(ep_vels[-10:])) * 100.0 if ep_vels else 0.0
        mean_disp = float(np.mean(ep_disps[-10:])) * 100.0 if ep_disps else 0.0
        
        training_log["epoch"].append(epoch)
        training_log["mean_episode_return"].append(mean_ret)
        training_log["mean_forward_velocity_mps"].append(mean_vel / 100.0)
        training_log["mean_x_displacement_m"].append(mean_disp / 100.0)
        training_log["total_timesteps"].append(t_global)
        
        print(f"Epoch {epoch:3d}/{num_epochs} | Steps: {t_global:6d} | Return: {mean_ret:7.1f} | Fwd Speed: {mean_vel:5.1f} cm/s | Dist: {mean_disp:5.1f} cm")

    # Save trained SAC policy checkpoint
    policy_path = os.path.join(save_dir, "sac_walk_policy.npz")
    policy.save(policy_path)
    
    with open(os.path.join(save_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)
        
    print("-" * 60)
    print(f"SAC Walking Locomotion Training Complete! Policy saved: {policy_path}")
    print("=" * 60)
    
    return training_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=60000)
    parser.add_argument("--save_dir", type=str, default="results/sac_walk")
    args = parser.parse_args()
    
    train_sac_locomotion(total_timesteps=args.timesteps, save_dir=args.save_dir)
