# 🐶 Sesame AI Quadruped Digital Twin & Reinforcement Learning Suite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Physics: MuJoCo](https://img.shields.io/badge/Physics-MuJoCo%203.0-brightgreen.svg)](https://mujoco.org/)
[![Framework: PPO & SAC](https://img.shields.io/badge/RL-PPO%20%26%20SAC-orange.svg)](https://openai.com/)
[![Web App: FastAPI + Three.js](https://img.shields.io/badge/Web-FastAPI%20%2B%20Three.js-purple.svg)](http://127.0.0.1:8000/)

Welcome to the **Sesame AI Quadruped Digital Twin** repository! This repository contains a full Sim-to-Real control stack, high-precision Deep Reinforcement Learning (PPO & SAC), 500 Hz physics simulation in MuJoCo, an interactive 3D Web Telemetry Dashboard, and custom motion synthesis controllers (**Vertical Jump**, **Handshake/Paw Wave**, **Rhythm Dance**, and **Fast Run**).

---

## 👶 1. The Story of Sesame (Explained So a Baby Can Understand!)

Imagine you have a little 4-legged robotic puppy named **Sesame**! 🐶

- **Chapter 1: Clumsy Puppy Days**  
  At first, Sesame was clumsy. When we asked it to walk toward a cyan ball, Sesame got confused and walked backwards or spun around in circles! When we asked Sesame to stretch its paw out to touch the ball, Sesame just stood still like a stone statue.

- **Chapter 2: Fixing Sesame's Wires**  
  We looked inside Sesame's computer brain and found out that the leg wires were crossed (the computer was sending Front-Right leg instructions to Front-Left!). We re-wired all 8 leg joints into the exact right order.

- **Chapter 3: Two Paws Are Better Than One**  
  We taught Sesame that it doesn't have to use just one leg — it can use **whichever front paw is closer** to reach the ball!

- **Chapter 4: Big Treats (+100 Bonus Points!)**  
  Every time Sesame's paw touches the ball, we give it a **huge +100 bonus treat**! And as soon as it touches the ball, the ball magically jumps forward so Sesame can keep walking and chasing it forever!

- **Chapter 5: 4 Million Steps of Practice!**  
  We let Sesame practice in its virtual playground for **4,000,000 steps**! Now Sesame can reach, walk, jump, wave, dance, and run at top speed without falling over! 🎉

---

## 🧠 2. Hard Terminologies Explained Simply

| Hard Term | Simple Explanation (What It Really Means) |
| :--- | :--- |
| **Reinforcement Learning (RL)** | Teaching a robot through trial and error — giving treats (+rewards) for good moves and "uh-ohs" (-penalties) for falling over, so it discovers how to walk on its own. |
| **PPO (Proximal Policy Optimization)** | A cautious, smart AI teacher that makes small, safe updates to the robot's brain so it learns steadily without unlearning past skills. |
| **SAC (Soft Actor-Critic)** | An adventurous AI teacher that encourages the robot to try creative new moves while saving past tries in a giant memory box (replay buffer). |
| **Inverse Kinematics (IK)** | Math geometry that calculates exactly how much to bend the hip and knee joints so the robot's paw touches a target ball in 3D space. |
| **CPG (Central Pattern Generator)** | A rhythmic clock in the robot's brain (like a heartbeat) that swings the diagonal legs in a smooth 1-2-3-4 walking pattern. |
| **Sim-to-Real & Domain Randomization** | Practicing in a super-fast computer game (MuJoCo) with random floor slipperiness and weight changes so the policy works on real physical hardware. |

---

## 🛠️ 3. Engineering Audit: Problems Tackled & How We Solved Them

| Problem Reported | Root Cause Identified | Engineering Solution Implemented |
| :--- | :--- | :--- |
| **Robot walking backward or orbiting target** | Joint index mismatch between Gym environment `qpos`/`qvel` indices and MuJoCo raw XML joint order vs `JOINT_NAMES`. | Mapped `qpos_indices`, `qvel_indices`, and `act_indices` directly to `JOINT_NAMES` in [`sesame_env.py`](file:///c:/Users/Krushna/OneDrive/Documents/sasame%20project%20BE/simulation/environment/sesame_env.py) & [`sesame_walk_env.py`](file:///c:/Users/Krushna/OneDrive/Documents/sasame%20project%20BE/simulation/environment/sesame_walk_env.py). |
| **Reaching mode stuck standing still at 223.5mm** | UI desync bug: `index.html` dropdown visually selected "PPO (AI Reach)", but [`controller_manager.py`](file:///c:/Users/Krushna/OneDrive/Documents/sasame%20project%20BE/gui/core/controller_manager.py) defaulted to `ControllerType.PID`. | Changed default `active_type = ControllerType.PPO` in backend and added automatic WebSocket synchronization on connection open in [`app.js`](file:///c:/Users/Krushna/OneDrive/Documents/sasame%20project%20BE/web/static/app.js). |
| **Single foot reach & static ball position** | Reward function only tracked Front-Left foot (FL) and did not award reach bonuses or relocate the target ball. | Updated reward to `min(dist_FL, dist_FR)` for multi-foot reach, increased touch bonus to `+100.0`, and implemented dynamic ball relocation on touch. |
| **Target ball staying still in walking mode** | Walking arrival loop stopped robot base when `dist_tgt < 40mm` without moving `world_target`. | Updated [`web/server.py`](file:///c:/Users/Krushna/OneDrive/Documents/sasame%20project%20BE/web/server.py) walking loop to advance `world_target` `0.30m` forward on arrival (`<50mm`) with `+100.0` waypoint bonus for endless locomotion. |

---

## 📈 4. Training History & Benchmark Scores

We conducted **7 full training iterations** totaling over **12 Million Simulation Timesteps**:

| Run # & Task | RL Algorithm | Total Timesteps | Episodic Return / Score | Key Performance Benchmark |
| :---: | :---: | :---: | :---: | :--- |
| **Run 1** (Reach Initial) | PPO | 100,000 steps | **+1,488.39** | Initial 3D end-effector targeting baseline |
| **Run 2** (Walk Initial) | PPO Walk | 60,000 steps | **+22.1 cm disp.** | 44.7 cm/s forward velocity, steady trot gait |
| **Run 3** (Reach Retrain) | PPO | 150,000 steps | **+2,374.95** | Multi-foot reach +100 touch reward bonus |
| **Run 4** (Reach Deep) | PPO Deep | 2,000,000 steps | **+2,887.52** | High-precision 2M step network checkpoint |
| **Run 5** (Reach 4M Final) | PPO Deep | **4,000,000 steps** | **+8,007.35** 🎉 | **ALL-TIME RECORD!** Near-zero reach error |
| **Run 6** (Walk 4M Final) | PPO Walk | **4,000,000 steps** | **+377.30** 🎉 | **ALL-TIME RECORD!** Endless forward locomotion |
| **Run 7** (SAC Baseline) | SAC Off-Policy | **4,000,000 steps** | **81,995 episodes** | 42.8 mm distance precision to target sphere |

---

## 🎮 5. Complete Motion Synthesis & Controller Suite

| Controller Option | Category | Kinematic & Physical Description |
| :--- | :---: | :--- |
| **PPO (AI Reach)** | Deep RL Policy | Actor-Critic network controlling 3D paw targeting (FL/FR) with IK residual feedback. |
| **PPO (AI Walk)** | Deep RL Policy | Autonomous trot gait policy advancing target sphere 0.30m forward on reach. |
| **SAC (AI)** | Off-Policy RL | Maximum entropy off-policy baseline for sample-efficient targeting exploration. |
| **🚀 VERTICAL JUMP** | Motion Preset | 4-Phase Jump: Crouch low -> Explosive thrust launch -> Flight tuck -> Landing absorption. |
| **🤝 HANDSHAKE / PAW** | Motion Preset | 3-Leg Tripod Balance + Front-Right paw raised high, waving in a 2.5 Hz handshake rhythm. |
| **💃 RHYTHM DANCE** | Motion Preset | 120 BPM tempo performance featuring side-to-side body roll sway + alternating paw tapping. |
| **⚡ FAST RUN** | Motion Preset | High-cadence (2.2 Hz) bounding trot gait achieving forward velocities > 60 cm/s. |
| **👋 WAVE HAND / PUSHUP** | Motion Preset | Expressive single paw greeting wave & synchronized core pushup workouts. |

---

## 🚀 6. Quick Start Guide

### 1. Launch the Executive Web Dashboard
```bash
python -m uvicorn web.server:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser to view the live 3D physics viewport, telemetry charts, and controller dropdowns!

### 2. Train Your Own Model From Scratch
You can train any policy from scratch using the unified training launcher [`train_custom.py`](file:///c:/Users/Krushna/OneDrive/Documents/sasame%20project%20BE/train_custom.py):

```bash
# Train PPO Reaching from scratch (e.g. 4 Million steps):
python train_custom.py --task reach --steps 4000000

# Train PPO Walking Locomotion from scratch:
python train_custom.py --task walk --steps 4000000

# Train SAC Baseline from scratch:
python train_custom.py --task sac --steps 4000000
```

### 3. Generate PDF Executive Report
To build the official PDF document report [`Sesame_AI_Quadruped_Final_Report.pdf`](file:///c:/Users/Krushna/OneDrive/Documents/sasame%20project%20BE/Sesame_AI_Quadruped_Final_Report.pdf), run:
```bash
python generate_pdf_report.py
```

---

## 📄 License & Attribution
Designed & developed for the **Sesame AI Quadruped Digital Twin Project** (August 2026).
All MuJoCo physics models, web assets, and training scripts are fully open-source.
