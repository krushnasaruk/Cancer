"""
Master Pipeline Runner for Sesame Robot MuJoCo Digital Twin.

Executes all verification phases sequentially:
1. Analytical Kinematics Unit Tests (FK, IK, Jacobians)
2. MuJoCo Model XML & Actuator Integrity Tests
3. Headless Physics Stability Validation
4. Classical PID Trajectory Tracking Experiment
5. Gymnasium Continuous Control Environment Verification
6. PPO Reinforcement Learning Training & Evaluation
"""

import os
import sys
import time
import subprocess

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_stage(title: str, cmd: list) -> bool:
    print("\n" + "=" * 70)
    print(f"  RUNNING STAGE: {title}")
    print("=" * 70)
    t0 = time.time()
    res = subprocess.run([sys.executable] + cmd)
    elapsed = time.time() - t0
    if res.returncode == 0:
        print(f"\n[OK] STAGE '{title}' COMPLETED SUCCESSFULLY in {elapsed:.2f}s")
        return True
    else:
        print(f"\n[FAIL] STAGE '{title}' FAILED with exit code {res.returncode}")
        return False


def main():
    print("*" * 70)
    print("   SESAME ROBOT DIGITAL TWIN — FULL SYSTEM EXECUTION PIPELINE")
    print("*" * 70)
    
    stages = [
        ("1. Kinematics Unit Tests", ["tests/test_kinematics.py"]),
        ("2. Model XML Integrity", ["tests/test_model.py"]),
        ("3. Headless Physics Stability", ["simulation/visualization/viewer.py", "--headless"]),
        ("4. PID Trajectory Tracking", ["experiments/pid_experiment.py"]),
        ("5. Gymnasium Environment Rollouts", ["scripts/verify_environment.py"]),
        ("6. PPO Reinforcement Learning", ["rl/ppo/train.py", "--timesteps", "4000", "--steps_per_epoch", "1000"]),
        ("7. PPO Policy Evaluation", ["rl/ppo/evaluate.py", "--policy", "results/ppo/ppo_policy.npz", "--episodes", "5"]),
        ("8. GUI Architecture & Worker Tests", ["tests/test_gui.py"]),
    ]
    
    results = {}
    for title, cmd in stages:
        success = run_stage(title, cmd)
        results[title] = "PASSED" if success else "FAILED"
        if not success:
            print("\nPipeline stopped due to error.")
            break
            
    print("\n" + "*" * 70)
    print("                       PIPELINE EXECUTION SUMMARY")
    print("*" * 70)
    for title, status in results.items():
        print(f" {title:<40} : {status}")
    print("*" * 70)


if __name__ == "__main__":
    main()
