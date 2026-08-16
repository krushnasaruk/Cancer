# Sesame AI Digital Twin — GUI User Guide & Operational Manual

---

## 1. Quick Launch

Launch the Sesame Control Center GUI from the project root:

```powershell
python run_gui.py
```

---

## 2. Interface Layout Overview

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  [▶ START] [⏸ PAUSE] [⏯ STEP] [↺ RESET] [⏹ STOP] [⛔ EMERGENCY STOP] | Ctrl: [PID ▼]  │
├────────────────────────────────────────────────────────────┬───────────────────────────┤
│                                                            │ 📊 LIVE TELEMETRY         │
│                                                            │ ───────────────────────── │
│                     3D MUJOCO VIEWPORT                     │ State: RUNNING (50 Hz)    │
│            (Interactive Mouse Orbit / Pan / Zoom)          │ UI: 30 FPS | Sim: 350 s/s │
│                                                            │ Base Pos: [0.0, 0.0, 0.05]│
│                                                            │ Orientation: [0°, 0°, 0°] │
│                                                            │ Target Dist: 18.2 mm      │
│                                                            │ Return: +2,166.80         │
├────────────────────────────────────────────────────────────┼───────────────────────────┤
│ ┌────────────────────────────────────────────────────────┐ │                           │
│ │ [📈 PyQtGraph Charts] [🦾 8-Joints] [⚙️ Actuator] [🔬 Res]│ │                           │
│ ├────────────────────────────────────────────────────────┤ │                           │
│ │   Live Multi-Tab Telemetry Curves / Benchmark Table    │ │                           │
│ └────────────────────────────────────────────────────────┘ │                           │
└────────────────────────────────────────────────────────────┴───────────────────────────┘
```

---

## 3. Interactive 3D Viewport Controls

| Mouse Action | Operation | Description |
|---|---|---|
| **Left Click + Drag** | **Orbit Camera** | Rotates camera azimuth and elevation around the robot focus point. |
| **Right Click + Drag** | **Pan Camera** | Shifts the camera focus point $(X, Y)$ in the ground plane. |
| **Mouse Wheel Scroll** | **Zoom In / Out** | Smoothly scales camera distance ($0.15\text{ m}$ to $3.0\text{ m}$). |
| **Double Click** | **Reset Camera** | Snaps the viewport back to nominal isometric perspective ($135^\circ$ azimuth, $-25^\circ$ elevation). |

---

## 4. Operational Workflows

### 4.1 Running Live Simulation
1. Click **`[▶ START]`** to begin physics and control execution.
2. Select your desired controller from the dropdown:
   - **`PID`**: Select `STAND` (posture hold), `SINE` (multi-joint oscillation), or `WALK` (trot gait).
   - **`PPO`**: Runs the trained deep reinforcement learning reaching policy.
   - **`SAC`**: Runs the maximum-entropy off-policy baseline.
   - **`PPO + Domain Randomization`**: Evaluates policy robustness under dynamic payload/friction shifts.
3. Switch environments (Testing Arena, Laboratory, Office, Outdoor, Uneven Terrain).

---

### 4.2 Running Quantitative Research Experiments
1. Click the **`🔬 Research Benchmark Dashboard`** tab at the bottom.
2. Choose the **Controller**, **Environment**, **Domain Randomization (ON/OFF)**, and **Episode Count** ($N=10$).
3. Click **`🚀 RUN BENCHMARK EXPERIMENT`**.
4. The background worker evaluates the complete batch without freezing the GUI.
5. The empirical metrics table updates automatically (Success Rate, Mean/Median Error, RMSE, Maximum Error, Return, FPS).
6. Records are automatically saved with timestamps to `results/gui_experiments/` in both JSON and CSV formats.

---

### 4.3 Monitoring Actuator Health
1. Click the **`⚙️ MG90S Actuator Diagnostics`** tab.
2. Observe live command vs actual angle, mechanical backlash play, velocity ceilings, and torque saturation alarms (`TORQUE SAT` flags red if load exceeds $0.196\text{ N}\cdot\text{m}$).

---

## 5. Keyboard Shortcuts Reference

- **`Ctrl + R`**: Reset Simulation & Pose
- **`Ctrl + E`**: Emergency Stop
- **`Ctrl + 0`**: Reset Camera Viewport
- **`Ctrl + Q`**: Exit Application
