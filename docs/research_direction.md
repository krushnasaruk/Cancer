# Research Direction: Actuator-Aware Sim-to-Real Reinforcement Learning

**Project Title:** Actuator-Aware Sim-to-Real Reinforcement Learning for Low-Cost Quadruped Robots  
**Platform:** Sesame-style 8-DOF Micro Quadruped (ESP32 + MG90S Servos)

---

## 1. Research Motivation & Problem Statement

Low-cost robotic platforms utilizing hobby-grade RC servomotors (such as the MG90S) present extreme sim-to-real transfer challenges due to unmodeled actuator dynamics:
- High gear train backlash and nonlinear friction.
- Severe voltage-dependent torque and velocity limits.
- Substantial internal PID controller delay and non-ideal bandwidth.
- Substantial unit-to-unit variance across low-cost servos.

Traditional **Domain Randomization (DR)** addresses sim-to-real gaps by randomizing physical parameters uniformly over wide heuristic ranges. However, excessive randomization often results in overly conservative policies or fails when the true reality distribution is structured around asymmetric actuator deficiencies.

---

## 2. Proposed Research Contribution: Actuator-Aware Adaptive Domain Randomization (A3DR)

We propose an iterative closed-loop sim-to-real framework:

1. **Step 1 — Baseline Digital Twin:** Develop a physically grounded MuJoCo simulation of Sesame with an explicit parametric MG90S actuator model.
2. **Step 2 — Physical Actuator Identification:** Characterize real MG90S servos under controlled step-response and loaded trajectory tests.
3. **Step 3 — Reality Gap Quantification:** Measure the discrepancy between simulated joint state trajectories $\mathbf{q}_{\text{sim}}(t)$ and real robot trajectories $\mathbf{q}_{\text{real}}(t)$.
4. **Step 4 — Adaptive Domain Randomization:** Dynamically center and scale domain randomization distributions around the measured empirical actuator error bounds rather than arbitrary intervals.
5. **Step 5 — Policy Optimization:** Train policies with PPO/SAC under the adaptive actuator randomization envelope.
6. **Step 6 — Comparative Evaluation:** Compare on real hardware against:
   - Conventional Classical PID Baseline
   - Plain Sim PPO (No Randomization)
   - Standard Heuristic Domain Randomization
   - Actuator-Aware Adaptive Domain Randomization (Proposed)

---

## 3. Evaluation Metrics

- **Trajectory Tracking RMSE:** $\sqrt{\frac{1}{T}\sum_{t=1}^T \|\mathbf{q}_{\text{real}}(t) - \mathbf{q}_{\text{ref}}(t)\|^2}$
- **End-Effector Precision Error:** Average distance between commanded and actual foot positions.
- **Sim-to-Real Gap Ratio:** Performance degradation percentage when transferring policy from simulation directly to hardware.
- **Control Smoothness & Energy Metric:** $\sum_t \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2 + \alpha \sum_t \|\mathbf{\tau}_t \dot{\mathbf{q}}_t\|$.
- **Task Success Rate:** Percentage of successful reaching/walking cycles without falling or violating limits.
