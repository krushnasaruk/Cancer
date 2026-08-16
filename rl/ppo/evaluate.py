"""
PPO Policy Evaluation Script for Sesame Reaching Environment.

Evaluates a saved PPO policy checkpoint, computes quantitative benchmarks,
and plots performance metrics.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from simulation.environment.sesame_env import SesameEnv
from rl.ppo.train import ActorCriticPolicy


def evaluate_ppo(
    policy_path: str = "results/ppo/ppo_policy.npz",
    num_episodes: int = 20,
    render: bool = False,
    results_dir: str = "results/ppo",
) -> dict:
    """Evaluate trained PPO policy."""
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"PPO checkpoint not found at: {policy_path}")
        
    env = SesameEnv(render_mode="human" if render else None, use_actuator_model=True)
    policy = ActorCriticPolicy(obs_dim=env.obs_dim, act_dim=8)
    policy.load(policy_path)
    
    returns = []
    final_distances = []
    successes = []
    
    print("=" * 60)
    print("      EVALUATING PPO POLICY — SESAME ROBOT")
    print("=" * 60)
    print(f"Policy: {policy_path}")
    print(f"Episodes: {num_episodes}")
    print("-" * 60)
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        ep_ret = 0.0
        
        while True:
            # Deterministic evaluation using mean action mu
            mu, _ = policy.forward_policy(obs)
            obs, rew, terminated, truncated, info = env.step(mu)
            ep_ret += rew
            
            if render:
                env.render()
                
            if terminated or truncated:
                final_dist = info.get("dist_to_target", 0.0)
                is_success = bool(final_dist < 0.025)
                returns.append(ep_ret)
                final_distances.append(final_dist)
                successes.append(is_success)
                print(f"Episode {ep + 1:2d}: Return = {ep_ret:8.2f} | Final Dist = {final_dist:.4f} m | Success = {is_success}")
                break
                
    success_rate = float(np.mean(successes)) * 100.0
    mean_dist = float(np.mean(final_distances))
    mean_ret = float(np.mean(returns))
    
    eval_metrics = {
        "num_episodes": num_episodes,
        "success_rate_percent": success_rate,
        "mean_final_distance_m": mean_dist,
        "mean_final_distance_mm": mean_dist * 1000.0,
        "mean_episode_return": mean_ret,
        "std_episode_return": float(np.std(returns)),
    }
    
    metrics_path = os.path.join(results_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
        
    print("-" * 60)
    print(f"Success Rate:    {success_rate:.1f}%")
    print(f"Mean Final Dist: {mean_dist * 1000.0:.2f} mm")
    print(f"Mean Return:     {mean_ret:.2f}")
    print("=" * 60)
    
    return eval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO Policy on Sesame")
    parser.add_argument("--policy", type=str, default="results/ppo/ppo_policy.npz")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    
    evaluate_ppo(policy_path=args.policy, num_episodes=args.episodes, render=args.render)
