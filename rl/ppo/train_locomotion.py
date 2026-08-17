"""
Autonomous Quadruped Locomotion PPO Training Script for Sesame Robot.

Learns:
- Dynamic trot gait coordination
- Forward velocity tracking (+X)
- Posture stabilization and energy-efficient motor outputs
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
from rl.ppo.train import ActorCriticPolicy, AdamOptimizer, compute_gae


def train_locomotion(
    total_timesteps: int = 300000,
    steps_per_epoch: int = 3000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    save_dir: str = "results/ppo_walk",
    seed: int = 42,
) -> dict:
    """Train PPO autonomous walking policy."""
    os.makedirs(save_dir, exist_ok=True)
    env = SesameWalkEnv(use_actuator_model=True)
    policy = ActorCriticPolicy(obs_dim=env.obs_dim, act_dim=8, seed=seed)
    
    pi_opt = AdamOptimizer(policy.get_policy_params(), lr=lr)
    v_opt = AdamOptimizer(policy.get_value_params(), lr=lr * 1.5)
    
    num_epochs = max(1, total_timesteps // steps_per_epoch)
    
    print("=" * 60)
    print("   STARTING PPO LOCOMOTION TRAINING — SESAME ROBOT")
    print("=" * 60)
    print(f"Task:            Autonomous Forward Walking Locomotion")
    print(f"Total Timesteps: {total_timesteps} ({num_epochs} epochs of {steps_per_epoch} steps)")
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
    
    for epoch in range(num_epochs):
        obs_buf = np.zeros((steps_per_epoch, env.obs_dim), dtype=np.float64)
        act_buf = np.zeros((steps_per_epoch, 8), dtype=np.float64)
        rew_buf = np.zeros(steps_per_epoch, dtype=np.float64)
        val_buf = np.zeros(steps_per_epoch, dtype=np.float64)
        logp_buf = np.zeros(steps_per_epoch, dtype=np.float64)
        done_buf = np.zeros(steps_per_epoch, dtype=bool)
        
        for step in range(steps_per_epoch):
            action, log_prob, val = policy.sample_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            obs_buf[step] = obs
            act_buf[step] = action
            rew_buf[step] = reward
            val_buf[step] = val
            logp_buf[step] = log_prob
            done_buf[step] = terminated or truncated
            
            current_ep_return += reward
            t_global += 1
            
            if terminated or truncated:
                ep_returns.append(current_ep_return)
                ep_vels.append(info.get("forward_vel_mps", 0.0))
                ep_disps.append(info.get("x_displacement_m", 0.0))
                current_ep_return = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs
                
        # Compute GAE advantages
        adv_buf, ret_buf = compute_gae(rew_buf, val_buf, done_buf, gamma=gamma, lam=lam)
        adv_norm = (adv_buf - np.mean(adv_buf)) / (np.std(adv_buf) + 1e-8)
        
        # Policy Network Backprop
        h1_pi = np.tanh(obs_buf @ policy.w1_pi + policy.b1_pi)
        h2_pi = np.tanh(h1_pi @ policy.w2_pi + policy.b2_pi)
        mu = np.tanh(h2_pi @ policy.w_mu + policy.b_mu)
        std = np.exp(np.clip(policy.log_std, -2.0, 0.5))
        
        grad_mu = (act_buf - mu) / (std ** 2 + 1e-8) * adv_norm[:, None] * (1.0 - mu ** 2)
        grad_w_mu = (h2_pi.T @ grad_mu) / steps_per_epoch
        grad_b_mu = np.mean(grad_mu, axis=0)
        
        d_h2 = (grad_mu @ policy.w_mu.T) * (1.0 - h2_pi ** 2)
        grad_w2_pi = (h1_pi.T @ d_h2) / steps_per_epoch
        grad_b2_pi = np.mean(d_h2, axis=0)
        
        d_h1 = (d_h2 @ policy.w2_pi.T) * (1.0 - h1_pi ** 2)
        grad_w1_pi = (obs_buf.T @ d_h1) / steps_per_epoch
        grad_b1_pi = np.mean(d_h1, axis=0)
        
        pi_grads = {
            "w_mu": grad_w_mu, "b_mu": grad_b_mu,
            "w2_pi": grad_w2_pi, "b2_pi": grad_b2_pi,
            "w1_pi": grad_w1_pi, "b1_pi": grad_b1_pi,
        }
        pi_opt.step(policy.get_policy_params(), pi_grads)
        
        # Value Network Backprop
        h1_v = np.tanh(obs_buf @ policy.w1_v + policy.b1_v)
        h2_v = np.tanh(h1_v @ policy.w2_v + policy.b2_v)
        v_pred = (h2_v @ policy.w_val + policy.b_val).squeeze(-1)
        v_err = (ret_buf - v_pred)[:, None]
        
        grad_w_val = (h2_v.T @ v_err) / steps_per_epoch
        grad_b_val = np.mean(v_err, axis=0)
        
        d_h2_v = (v_err @ policy.w_val.T) * (1.0 - h2_v ** 2)
        grad_w2_v = (h1_v.T @ d_h2_v) / steps_per_epoch
        grad_b2_v = np.mean(d_h2_v, axis=0)
        
        d_h1_v = (d_h2_v @ policy.w2_v.T) * (1.0 - h1_v ** 2)
        grad_w1_v = (obs_buf.T @ d_h1_v) / steps_per_epoch
        grad_b1_v = np.mean(d_h1_v, axis=0)
        
        v_grads = {
            "w_val": grad_w_val, "b_val": grad_b_val,
            "w2_v": grad_w2_v, "b2_v": grad_b2_v,
            "w1_v": grad_w1_v, "b1_v": grad_b1_v,
        }
        v_opt.step(policy.get_value_params(), v_grads)
        
        mean_ret = float(np.mean(ep_returns[-10:])) if ep_returns else current_ep_return
        mean_v = float(np.mean(ep_vels[-10:])) if ep_vels else 0.0
        mean_d = float(np.mean(ep_disps[-10:])) if ep_disps else 0.0
        
        training_log["epoch"].append(epoch + 1)
        training_log["mean_episode_return"].append(mean_ret)
        training_log["mean_forward_velocity_mps"].append(mean_v)
        training_log["mean_x_displacement_m"].append(mean_d)
        training_log["total_timesteps"].append(t_global)
        
        print(f"Epoch {epoch + 1:3d}/{num_epochs} | Steps: {t_global:6d} | Return: {mean_ret:7.1f} | Fwd Speed: {mean_v * 100:.1f} cm/s | Dist: {mean_d * 100:.1f} cm", flush=True)
        
    policy_path = os.path.join(save_dir, "ppo_walk_policy.npz")
    policy.save(policy_path)
    
    log_path = os.path.join(save_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
        
    print("-" * 60)
    print(f"Walking Locomotion Training Complete! Policy saved: {policy_path}")
    print("=" * 60)
    return training_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Walking Locomotion Training")
    parser.add_argument("--timesteps", type=int, default=300000, help="Total training timesteps")
    parser.add_argument("--steps_per_epoch", type=int, default=3000, help="Steps per rollout epoch")
    parser.add_argument("--save_dir", type=str, default="results/ppo_walk", help="Checkpoint directory")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()
    
    train_locomotion(
        total_timesteps=args.timesteps,
        steps_per_epoch=args.steps_per_epoch,
        save_dir=args.save_dir,
        lr=args.lr,
    )
