# Digital Twin Architecture & Sim-to-Real Pipeline

This document details the software architecture, data flow, and closed-loop sim-to-real calibration lifecycle for the Sesame quadruped digital twin.

---

## 1. High-Level Closed-Loop Architecture

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

## 2. Component Stack

1. **Physical Modeling Layer (`simulation/model/sesame.xml`):**
   - Precise multibody kinematics tree with 8 revolute hinge joints.
   - Contact dynamics between 4 foot contact geoms and the ground plane.
   - Exact limit boundaries and geometric dimensions.

2. **Kinematics & Dynamics Layer (`robot/`):**
   - Analytical Forward Kinematics (FK) mapping joint angle vectors $\mathbf{q} \in \mathbb{R}^8$ to 3D Cartesian coordinates of the 4 feet $\mathbf{p}_{\text{feet}} \in \mathbb{R}^{4 \times 3}$.
   - Geometric transformations and differential kinematics (Jacobian computation).

3. **Actuator Calibration Layer (`calibration/actuator_model.py`):**
   - Simulates realistic non-idealities of low-cost MG90S RC servos:
     - Deadband & hysteresis
     - Quantization / resolution limits
     - First-order time lag / delay
     - Velocity & torque saturation
     - Stochastic noise

4. **Reinforcement Learning Environment (`simulation/environment/sesame_env.py`):**
   - Standard Gymnasium interface for continuous control.
   - Observation space: $[\mathbf{q}, \dot{\mathbf{q}}, \mathbf{p}_{\text{feet}}, \mathbf{p}_{\text{target}}, \mathbf{\theta}_{\text{base}}, \mathbf{\omega}_{\text{base}}]$.
   - Action space: Normalized target joint positions $\mathbf{a} \in [-1, 1]^8$.
   - Configurable reward function for reaching, standing balance, and trajectory tracking.

5. **RL Algorithms & Policy Benchmarks (`rl/`):**
   - PPO (Proximal Policy Optimization) and SAC (Soft Actor-Critic).
   - Domain randomization hooks for physics and actuator parameters.
