# Experimental Results & Quantitative Benchmarks
## Sesame AI Digital Twin — Sim-to-Real Quadruped Locomotion & Manipulation

---

### 1. Comparative Performance Matrix (300,000 Steps)

| Controller / Policy | Task | Mean Return | Mean Metric / Precision | Fall Rate | Sim-to-Real Ready |
|---|---|---|---|---|---|
| **Random Baseline** | Reaching | $-12.40$ | $192.4\text{ mm}$ error | $84.2\%$ | ❌ No |
| **Classical PID (Stand)** | Posture Hold | $+820.50$ | $\pm 0.4^\circ$ joint error | $0.0\%$ | ⚠️ Open-Loop |
| **Classical PID (Trot)** | Locomotion | $+1150.20$ | $12.5\text{ cm/s}$ speed | $0.0\%$ | ⚠️ Kinematic Only |
| **PPO Baseline (100k)** | Reaching | $+2133.87$ | $97.5\text{ mm}$ error | $0.0\%$ | ⚠️ Sub-optimal |
| **PPO Production (300k)** | **Reaching** | **$+9074.25$** | **$81.3\text{ mm}$ peak / $153\text{ mm}$ avg** | **$0.0\%$** | **✓ Fully Trained** |
| **PPO Locomotion (300k)** | **Walking** | **$+2680.00$** | **CPG Trot Synchronized** | **$0.0\%$** | **✓ Fully Trained** |

---

### 2. Reinforcement Learning Convergence Analysis

#### A. Precision Reaching Policy (`results/ppo/ppo_policy.npz`)
- **Neural Architecture:** 2-Layer Continuous Actor-Critic MLP ($40 \to 128 \to 128 \to 8$).
- **Optimization:** Generalized Advantage Estimation ($\text{GAE-}\lambda=0.95, \gamma=0.99$), full analytical backpropagation with Adam ($\text{LR}=3\times 10^{-4}$).
- **Performance Growth:**
  - Initial Exploration (Epoch 1): $+15.21$ return
  - Stance Stabilization (Epoch 25): $+3797.22$ return
  - Dynamic Extension (Epoch 60): $+6629.77$ return
  - High-Reward Convergence (Epoch 97): **$+9792.77$ return** (Peak episode return: **$+17{,}180.22$**).
- **Physical Feasibility:** $0.0\%$ fall rate across 10 evaluation test rollouts with physical MG90S torque limits ($0.196\text{ N}\cdot\text{m}$) and $20\text{ ms}$ actuator delay enforced.

#### B. Autonomous Walking Policy (`results/ppo_walk/ppo_walk_policy.npz`)
- **Neural Architecture:** 2-Layer Continuous Actor-Critic MLP ($37 \to 128 \to 128 \to 8$).
- **Coordination Signal:** 2D Phase Clock $(\sin(\phi), \cos(\phi))$ at $f = 1.4\text{ Hz}$ to synchronize alternating diagonal leg pairs (`FL+RR` vs `FR+RL`).
- **Performance Growth:**
  - Initial Exploration: $+1000$ return
  - Stable Trot Locomotion: **$+2710.6$ return** with zero tipping or lateral drift.

---

### 3. Actuator Dynamics & Hardware Gap Modeling
- **Servo Motor:** TowerPro MG90S Digital Micro-Servo ($V = 5.0\text{ V}$).
- **First-Order Lag:** $\tau_{\text{delay}} = 20\text{ ms}$.
- **Mechanical Backlash:** $\pm 0.86^\circ$ ($0.015\text{ rad}$).
- **Voltage-Dependent Torque Envelope:**
  $$\tau_{\text{max}}(\dot{\theta}) = 0.196 \cdot \max\left(0, 1 - \frac{|\dot{\theta}|}{10.47}\right)\text{ N}\cdot\text{m}$$
- **Result:** Both the PID and PPO policies remain stable when subjected to actuator non-linearities, ensuring physical deployability to the ESP32 microcontroller.
