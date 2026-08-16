# Simulation Assumptions & Engineering Approximations

This document catalogs all physical, dynamical, and structural assumptions incorporated into the Sesame MuJoCo Digital Twin. Every entry here is intentionally parameterized in the codebase so physical calibration data can overwrite it.

---

## 1. Structural & Mass Properties

| Component | Assumed Mass | Ground Truth Status | Rationale / Method | Code Location |
|---|---|---|---|---|
| **Base Chassis** | $0.210\text{ kg}$ ($210\text{ g}$) | Approximation | Internal frame + bottom cover + top cover + ESP32 + OLED + 2x 14500 batteries + 4 hip MG90S servos. | `robot/parameters.py` |
| **Femur Link (×4)** | $0.022\text{ kg}$ ($22\text{ g}$) | Approximation | PLA 3D print shell ($8\text{ g}$) + 1 MG90S servo ($13.4\text{ g}$) + M2 fasteners. | `robot/parameters.py` |
| **Tibia Link (×4)** | $0.008\text{ kg}$ ($8\text{ g}$) | Approximation | Solid PLA lower leg shell with rounded contact foot. | `robot/parameters.py` |
| **Total Mass** | $0.330\text{ kg}$ ($330\text{ g}$) | Approximation | Sum of chassis and all 4 leg assemblies. | `robot/parameters.py` |
| **Inertia Tensors** | Diagonal box/cylinder approximations | Approximation | Computed analytically from link bounding dimensions and mass. | `simulation/model/sesame.xml` |

---

## 2. Actuator Assumptions (MG90S Micro Metal Gear Servos)

| Parameter | Initial Assumed Value | Ground Truth Source / Reference | Replaceable In |
|---|---|---|---|
| **Stall Torque ($\tau_{\max}$)** | $0.196\text{ N}\cdot\text{m}$ ($2.0\text{ kg}\cdot\text{cm}$) | MG90S Datasheet at 5.0V | `calibration/actuator_model.py` |
| **No-load Max Speed ($\dot{q}_{\max}$)** | $10.0\text{ rad/s}$ ($573^\circ/\text{s}$) | Datasheet: $0.10\text{ s}/60^\circ$ | `calibration/actuator_model.py` |
| **Gear Backlash / Play** | $\pm 0.015\text{ rad}$ ($\approx 0.86^\circ$) | Estimated metal gear train backlash | `calibration/actuator_model.py` |
| **Response Delay / Lag** | First-order filter $\tau = 0.020\text{ s}$ ($20\text{ ms}$) | PWM frequency & internal motor inductance | `calibration/actuator_model.py` |
| **Position Noise** | Gaussian $\sigma = 0.008\text{ rad}$ ($\approx 0.46^\circ$) | Internal potentiometer sensor noise | `calibration/actuator_model.py` |
| **Joint Damping ($d$)** | $0.005\text{ N}\cdot\text{m}\cdot\text{s/rad}$ | Internal grease & gearbox viscous friction | `simulation/model/sesame.xml` |
| **Joint Armature Inertia** | $0.0001\text{ kg}\cdot\text{m}^2$ | Motor rotor & gear reduction reflection | `simulation/model/sesame.xml` |

---

## 3. Contact & Environment Assumptions

| Parameter | Initial Assumed Value | Rationale | Code Location |
|---|---|---|---|
| **Gravity ($\mathbf{g}$)** | $[0, 0, -9.81]\text{ m/s}^2$ | Standard Earth gravitational acceleration | `simulation/model/sesame.xml` |
| **Foot Ground Friction ($\mu$)** | Tangential: $0.8$, Torsional: $0.01$, Rolling: $0.001$ | PLA point contact on dry flat floor | `simulation/model/sesame.xml` |
| **Contact Solver** | MuJoCo elliptic cone, implicit fast friction | High numerical stability for multi-contact | `simulation/model/sesame.xml` |
| **Simulation Timestep ($\Delta t$)** | $0.002\text{ s}$ ($500\text{ Hz}$ physics, $50\text{ Hz}$ RL policy decimation = 10 sub-steps) | Standard high-fidelity contact resolution | `simulation/environment/sesame_env.py` |
