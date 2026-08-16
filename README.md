# Sesame AI Digital Twin: Actuator-Aware Sim-to-Real RL Platform

An open, high-fidelity **MuJoCo Digital Twin** and **Sim-to-Real Reinforcement Learning Platform** for the **Sesame Quadruped Robot** ([dorianborian/sesame-robot](https://github.com/dorianborian/sesame-robot)).

---

## 1. Project Objective

The goal of this project is to develop a mathematically grounded, physically accurate digital twin of the low-cost Sesame quadruped robot. The platform serves as the simulation foundation for research into **Actuator-Aware Adaptive Domain Randomization (A3DR)**, bridging the reality gap between simulation and low-cost hobby-grade RC servomotors (MG90S).

---

## 2. Research Motivation

Low-cost quadruped robots built with $5-$10 micro servomotors (such as TowerPro / generic MG90S) present severe sim-to-real transfer obstacles:
- **Gear Backlash & Deadband:** Substantial mechanical play ($\approx 1^\circ$) in plastic/brass gear trains.
- **Torque & Speed Saturation:** Voltage-dependent torque derating and velocity ceilings under quadruped payload.
- **Internal Controller Delays:** PWM duty cycle quantization and analog/digital servo loop lag ($\approx 20\text{ ms}$).
- **Unit Variance:** Significant parameter divergence across mass-manufactured budget servos.

This platform introduces an explicit **parametric actuator dynamics layer** and an adaptive domain randomization pipeline to achieve robust sim-to-real policy transfer without requiring high-end direct-drive or quasi-direct-drive actuators.

---

## 3. Digital Twin Architecture

```
                  ┌───────────────────────────────┐
                  │          REAL ROBOT           │
                  │ (Sesame ESP32 + MG90S Servos) │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │    Servo Characterization     │
                  │   (Bench tests, step-response,│
                  │    torque-speed, latency)     │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │     Actuator Model Module     │
                  │ (calibration/actuator_model.py│
                  │  delay, backlash, saturation) │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │      MuJoCo Digital Twin      │
                  │     (sesame.xml, kinematics,  │
                  │      sesame_env.py Gym Env)   │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │          RL Training          │
                  │   (PPO / SAC with Adaptive    │
                  │     Domain Randomization)     │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │      Trained RL Policy        │
                  │      (Actor Network/ONNX)     │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │    Real Robot Deployment      │
                  │ (WiFi UDP/ESP-NOW / Serial)   │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │   Measured Error & Reality    │
                  │             Gap               │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │      Digital Twin Update      │
                  │  (Adjust randomized bounds    │
                  │   & actuator model params)    │
                  └───────────────────────────────┘
```

---

## 4. Project Structure

```
sesame-ai-digital-twin/
│
├── README.md                           # Master Project Documentation
├── requirements.txt                    # Python Dependencies
├── .gitignore                          # Git Exclusions
│
├── docs/                               # Technical Documentation & Specs
│   ├── sesame_analysis.md              # Deep Repository Analysis & Hardware Specs
│   ├── digital_twin_architecture.md    # Closed-Loop Sim-to-Real Data Flow
│   ├── assumptions.md                  # Simulation Assumptions & Approximations
│   └── research_direction.md           # Sim-to-Real RL Research Methodology
│
├── simulation/                         # MuJoCo Physics & Simulation Core
│   ├── model/
│   │   ├── sesame.xml                  # Multi-Body MJCF Physics Model
│   │   ├── meshes/                     # STL/CAD Mesh Geometry
│   │   └── textures/                   # Environment & Terrain Textures
│   │
│   ├── environment/
│   │   └── sesame_env.py               # Gymnasium Continuous Control Environment
│   │
│   ├── controllers/
│   │   ├── pid.py                      # Multi-Joint PID Baseline Controller
│   │   └── trajectory.py               # Spline & Periodic Gait Generators
│   │
│   └── visualization/
│       └── viewer.py                   # Interactive MuJoCo 3D Viewer & Hotkeys
│
├── robot/                              # Kinematics, Dynamics & Parameters
│   ├── parameters.py                   # Single Source of Truth for Dimensions & Limits
│   ├── kinematics.py                   # Analytical FK, Jacobians, and IK
│   └── dynamics.py                     # Mass, CoM, and Static Stability Margins
│
├── calibration/                        # Actuator Identification & Dynamics
│   └── actuator_model.py               # Parametric MG90S Delay, Backlash & Saturation
│
├── rl/                                 # Reinforcement Learning Baselines
│   ├── environment.py                  # Gym Registration & Environment Factory
│   ├── ppo/
│   │   ├── train.py                    # PPO Training Implementation
│   │   └── evaluate.py                 # PPO Deterministic Evaluation Benchmark
│   └── sac/
│       └── train.py                    # Soft Actor-Critic (SAC) Baseline
│
├── experiments/                        # Reproducible Research Experiments
│   └── pid_experiment.py               # Trajectory Tracking Benchmark & Plotting
│
├── results/                            # Benchmark Metrics & Publication Plots
│   ├── pid/                            # PID Tracking Logs & Plots
│   └── ppo/                            # PPO Checkpoints & Evaluation Logs
│
├── scripts/                            # Operational & Verification Scripts
│   └── verify_environment.py           # Random-Action Stress Test & Environment Checker
│
└── tests/                              # Automated Unit Test Suite
    └── test_kinematics.py              # FK, IK, and Jacobian Precision Tests
```

---

## 5. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/sesame-ai-digital-twin.git
cd sesame-ai-digital-twin

# Install core dependencies
pip install -r requirements.txt
```

---

## 6. Usage Guide

### 6.1 Running the 3D Interactive Simulator
Launch the interactive MuJoCo viewer:
```bash
python simulation/visualization/viewer.py
```
**Interactive Hotkeys:**
- `[1]`: Apply REST Pose (All 8 servos at $90^\circ$)
- `[2]`: Apply STAND Pose (Firmware standard stand)
- `[3]`: Apply Periodic Diagonal Walking Gait
- `[4]`: Apply Wave Pose Animation
- `[R]`: Reset robot pose and height

Run in headless verification mode:
```bash
python simulation/visualization/viewer.py --headless
```

---

### 6.2 Running the PID Trajectory Tracking Benchmark
Execute the classical PID baseline experiment:
```bash
python experiments/pid_experiment.py
```
This runs the full trajectory tracking simulation, logs error statistics, and saves plots:
- `results/pid/joint_tracking_trajectory.png`
- `results/pid/end_effector_trajectory.png`
- `results/pid/tracking_metrics.json`

---

### 6.3 Validating the Gymnasium RL Environment
Run the 1,000-step random action stress test:
```bash
python scripts/verify_environment.py
```

---

### 6.4 Running Kinematics Unit Tests
```bash
python tests/test_kinematics.py
```

---

### 6.5 Training and Evaluating PPO
Train the PPO reaching policy:
```bash
python rl/ppo/train.py --timesteps 20000
```
Evaluate the trained policy checkpoint:
```bash
python rl/ppo/evaluate.py --policy results/ppo/ppo_policy.npz --episodes 10
```

---

## 7. Current Limitations & Assumptions

1. **Mass & Center of Mass (CoM):** Approximated as $330\text{ g}$ total with a balanced chassis layout. Needs knife-edge measurement on physical build with battery installed.
2. **Actuator Model Values:** Backlash ($\pm 0.86^\circ$), delay ($20\text{ ms}$), and torque limits ($0.196\text{ N}\cdot\text{m}$) are initial engineering estimates.
3. **Foot Contact:** Modeled with point-friction capsules. Actual 3D-printed PLA on various surfaces should be calibrated using static/dynamic pull tests.

See [docs/assumptions.md](docs/assumptions.md) for full catalog.

---

## 8. Physical Hardware Sim-to-Real Transfer Workflow

1. Assemble physical Sesame robot per [docs/sesame_analysis.md](docs/sesame_analysis.md).
2. Measure actual MG90S step response and stall torque using an external load cell / encoder.
3. Update polynomial calibration and latency parameters in [calibration/actuator_model.py](calibration/actuator_model.py).
4. Retrain policy with Actuator-Aware Adaptive Domain Randomization (`rl/ppo/train.py`).
5. Export policy weights to C++ header / micro-TFLite or stream motor commands over WiFi UDP / ESP-NOW.
