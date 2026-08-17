"""
Unified Training Launcher for Sesame Quadruped RL (Train From Scratch).

Allows you to train PPO Reaching, PPO Walking Locomotion, SAC, or PPO+DR 
completely from scratch for any number of steps (e.g., 2,000,000 steps).

Usage Examples:
---------------
1. Train PPO Reaching from scratch for 2 Million steps:
   python train_custom.py --task reach --steps 2000000

2. Train PPO Walking Locomotion from scratch for 2 Million steps:
   python train_custom.py --task walk --steps 2000000

3. Train SAC Off-Policy Baseline for 1 Million steps:
   python train_custom.py --task sac --steps 1000000
"""

import os
import sys
import argparse
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rl.ppo.train import train_ppo
from rl.ppo.train_locomotion import train_locomotion
from rl.sac.train import train_sac


def main():
    parser = argparse.ArgumentParser(description="Sesame Quadruped RL Training Launcher (Train From Scratch)")
    parser.add_argument(
        "--task",
        type=str,
        default="reach",
        choices=["reach", "walk", "sac"],
        help="Task to train: 'reach' (PPO Reaching), 'walk' (PPO Walking Locomotion), or 'sac' (SAC Baseline)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500000,
        help="Total training timesteps (e.g. 2000000 for 2 Million steps)",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=4000,
        help="Timesteps collected per epoch before policy update",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("   SESAME QUADRUPED RL — CUSTOM TRAINING FROM SCRATCH")
    print("=" * 65)
    print(f" Task Selected:    {args.task.upper()}")
    print(f" Total Timesteps:  {args.steps:,} steps")
    print(f" Steps per Epoch:  {args.steps_per_epoch:,} steps")
    print(f" Learning Rate:    {args.lr}")
    print(f" Random Seed:      {args.seed}")
    print("=" * 65 + "\n")

    if args.task == "reach":
        save_dir = "results/ppo"
        print(f"Starting PPO Reaching Training from scratch... Saving to '{save_dir}'\n")
        train_ppo(
            total_timesteps=args.steps,
            steps_per_epoch=args.steps_per_epoch,
            save_dir=save_dir,
            lr=args.lr,
            seed=args.seed,
        )
    elif args.task == "walk":
        save_dir = "results/ppo_walk"
        print(f"Starting PPO Walking Locomotion Training from scratch... Saving to '{save_dir}'\n")
        train_locomotion(
            total_timesteps=args.steps,
            steps_per_epoch=args.steps_per_epoch,
            save_dir=save_dir,
            lr=args.lr,
            seed=args.seed,
        )
    elif args.task == "sac":
        save_dir = "results/sac"
        print(f"Starting SAC Baseline Training from scratch... Saving to '{save_dir}'\n")
        train_sac(
            total_timesteps=args.steps,
            save_dir=save_dir,
            lr=args.lr,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
