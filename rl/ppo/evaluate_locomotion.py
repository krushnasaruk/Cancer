"""
Locomotion Evaluation Script for Sesame Quadruped Walking Policy.
"""

import os
import sys
import json
import argparse
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from simulation.environment.sesame_walk_env import SesameWalkEnv
from rl.ppo.train import ActorCriticPolicy


def evaluate_locomotion(
    policy_path: str = "results/ppo_walk/ppo_walk_policy.npz",
    num_episodes: int = 10,
    render: bool = False,
    results_dir: str = "results/ppo_walk",
) -> dict:
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Locomotion checkpoint not found at: {policy_path}")
        
    env = SesameWalkEnv(render_mode="human" if render else None, use_actuator_model=True)
    policy = ActorCriticPolicy(obs_dim=env.obs_dim, act_dim=8)
    policy.load(policy_path)
    
    returns = []
    displacements = []
    velocities = []
    uprights = []
    
    print("=" * 60)
    print("   EVALUATING PPO LOCOMOTION POLICY — SESAME ROBOT")
    print("=" * 60)
    print(f"Policy:   {policy_path}")
    print(f"Episodes: {num_episodes}")
    print("-" * 60)
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        ep_ret = 0.0
        
        while True:
            mu, _ = policy.forward_policy(obs)
            obs, rew, terminated, truncated, info = env.step(mu)
            ep_ret += rew
            
            if render:
                env.render()
                
            if terminated or truncated:
                disp = info.get("x_displacement_m", 0.0)
                vel = info.get("forward_vel_mps", 0.0)
                upright = info.get("upright_factor", 1.0)
                
                returns.append(ep_ret)
                displacements.append(disp)
                velocities.append(vel)
                uprights.append(upright)
                
                print(f"Episode {ep + 1:2d}: Return = {ep_ret:7.1f} | Fwd Dist = {disp * 100:6.1f} cm | Speed = {vel * 100:5.1f} cm/s | Upright = {upright:.2f}")
                break
                
    mean_disp = float(np.mean(displacements))
    mean_vel = float(np.mean(velocities))
    mean_ret = float(np.mean(returns))
    
    metrics = {
        "num_episodes": num_episodes,
        "mean_x_displacement_cm": mean_disp * 100.0,
        "mean_forward_velocity_cm_s": mean_vel * 100.0,
        "mean_episode_return": mean_ret,
        "mean_upright_stability": float(np.mean(uprights)),
    }
    
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, "locomotion_eval_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("-" * 60)
    print(f"Mean Forward Displacement: {mean_disp * 100.0:.1f} cm")
    print(f"Mean Forward Speed:        {mean_vel * 100.0:.1f} cm/s")
    print(f"Mean Episode Return:       {mean_ret:.1f}")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO Locomotion")
    parser.add_argument("--policy", type=str, default="results/ppo_walk/ppo_walk_policy.npz")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    
    evaluate_locomotion(
        policy_path=args.policy,
        num_episodes=args.episodes,
        render=args.render,
    )
