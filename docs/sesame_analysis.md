# Sesame Robot Repository Technical Analysis & Specifications

**Primary Source Reference:** [dorianborian/sesame-robot](https://github.com/dorianborian/sesame-robot)  
**Document Status:** Complete Phase 1 Analysis  
**Audience:** Lead Robotics Simulation Engineer / Sim-to-Real RL Research Team

---

## 1. Classification Methodology

To ensure engineering rigor and prevent simulated artifacts from propagating into sim-to-real reinforcement learning, all parameters and specifications in this document are strictly classified under three tiers:

- `[DIRECT REPOSITORY DATA]`: Verified directly from code, CAD files, circuit schematics, build manuals, or hardware BOMs in the official Sesame repository.
- `[INITIAL SIMULATION ASSUMPTION]`: Physically motivated engineering approximations used where exact measurements are not explicitly specified in the repository.
- `[PENDING PHYSICAL BENCH MEASUREMENT]`: Specific real-world parameters that must be experimentally measured on the physical hardware platform prior to physical sim-to-real transfer.

---

## 2. Comprehensive System Breakdown

### 2.1 Robot Body Components
- **Chassis & Enclosure:**
  - `[DIRECT REPOSITORY DATA]` **Internal Frame (`Internal-Frame-v121.stl`)**: Central structural skeleton hosting the 4 hip servos, battery compartment, and distro board mounts.
  - `[DIRECT REPOSITORY DATA]` **Bottom Cover (`Bottom-Cover-v121.stl`)**: Base plate securing the bottom of the internal frame with M2 screws.
  - `[DIRECT REPOSITORY DATA]` **Top Cover (`Top-Cover-Enclosed-v117.stl`)**: Houses the SSD1306 OLED screen, power switch, magnetic accessory mounts, and wire routing channels.
- **Limb Assemblies (4 Legs: FL, FR, RL, RR):**
  - `[DIRECT REPOSITORY DATA]` **Femur Links (Upper Leg / Hip Horn Mount)**:
    - Left Front Femur: `L1-v117.stl`
    - Right Front Femur: `R1-v117.stl`
    - Left Rear Femur: `L2-v117.stl`
    - Right Rear Femur: `R2-v117.stl`
    - Function: Bolted to the hip servo horn on one end; rigidly encloses the knee MG90S servo on the other.
  - `[DIRECT REPOSITORY DATA]` **Tibia / Foot Links (Lower Leg)**:
    - Left Front Tibia/Foot: `L3-v117.stl`
    - Right Front Tibia/Foot: `R3-v117.stl`
    - Left Rear Tibia/Foot: `L4-v117.stl`
    - Right Rear Tibia/Foot: `R4-v117.stl`
    - Function: Bolted directly to the knee servo horn and makes point contact with the terrain.

---

### 2.2 Actuated Joints & Kinematic Configuration
- `[DIRECT REPOSITORY DATA]` **Total Actuated Joints:** 8 Degrees of Freedom (2 DOFs per leg × 4 legs).
- `[DIRECT REPOSITORY DATA]` **Joint Kinematic Topology:** Planar 2-DOF sagittal pitch linkages per quadrant.
- `[DIRECT REPOSITORY DATA]` **Joint Types:** All 8 joints are single-axis continuous Revolute (Hinge) joints.
- `[DIRECT REPOSITORY DATA]` **Joint Names & Servo Channel Mapping:**

| Leg Quadrant | Joint Name | Joint Code | Firmware Servo Identifier | ESP32 S2-Mini Pin | Distro V3 Pin | Motion Type |
|---|---|---|---|---|---|---|
| **Front Left (FL)** | `fl_hip_joint` | FL_HIP | `L1` (Index 2) | GPIO 4 | GPIO 6 | Sagittal Hip Pitch |
| **Front Left (FL)** | `fl_knee_joint` | FL_KNEE | `L3` (Index 6) | GPIO 13 | GPIO 12 | Sagittal Knee Pitch |
| **Front Right (FR)** | `fr_hip_joint` | FR_HIP | `R1` (Index 0) | GPIO 1 | GPIO 4 | Sagittal Hip Pitch |
| **Front Right (FR)** | `fr_knee_joint` | FR_KNEE | `R3` (Index 5) | GPIO 10 | GPIO 11 | Sagittal Knee Pitch |
| **Rear Left (RL)** | `rl_hip_joint` | RL_HIP | `L2` (Index 3) | GPIO 6 | GPIO 7 | Sagittal Hip Pitch |
| **Rear Left (RL)** | `rl_knee_joint` | RL_KNEE | `L4` (Index 7) | GPIO 14 | GPIO 13 | Sagittal Knee Pitch |
| **Rear Right (RR)** | `rr_hip_joint` | RR_HIP | `R2` (Index 1) | GPIO 2 | GPIO 5 | Sagittal Hip Pitch |
| **Rear Right (RR)** | `rr_knee_joint` | RR_KNEE | `R4` (Index 4) | GPIO 8 | GPIO 10 | Sagittal Knee Pitch |

---

### 2.3 Joint Axes, Neutral Poses, and Limits

- `[DIRECT REPOSITORY DATA]` **Joint Axes of Rotation:**
  - All 8 joints rotate about the lateral horizontal axis: **Y-axis** $\mathbf{a} = [0, 1, 0]^T$.
- `[DIRECT REPOSITORY DATA]` **Reference Calibration & Assembly Angles:**
  - Standard REST pose in firmware: All 8 servos at $90^\circ$ (mechanical calibration midpoint).
  - Standard STAND pose in firmware:
    - `R1`: $135^\circ$, `R2`: $45^\circ$, `L1`: $45^\circ$, `L2`: $135^\circ$
    - `R4`: $0^\circ$, `R3`: $180^\circ$, `L3`: $0^\circ$, `L4`: $180^\circ$
- `[DIRECT REPOSITORY DATA]` **Firmware Valid Joint Angle Limits (from `sesame_studio.py` & `movement-sequences.h`):**
  - `R1`, `L2` (Hip Diagonals): $[45^\circ, 180^\circ]$ $\rightarrow$ $[0.785, 3.1415]\text{ rad}$
  - `R2`, `L1` (Hip Diagonals): $[0^\circ, 135^\circ]$ $\rightarrow$ $[0.0, 2.356]\text{ rad}$
  - `R3`, `R4`, `L3`, `L4` (All Knees): $[0^\circ, 180^\circ]$ $\rightarrow$ $[0.0, 3.1415]\text{ rad}$

---

### 2.4 Servo Locations & Mechanical Placement
- `[DIRECT REPOSITORY DATA]` **Hip Servos (4 Units):**
  - Embedded horizontally inside the 4 corners of the internal chassis frame (`Internal-Frame-v121.stl`).
  - Output spline shafts face outwards laterally along the $\pm Y$ axis.
- `[DIRECT REPOSITORY DATA]` **Knee Servos (4 Units):**
  - Embedded directly inside the upper femur shells (`L1`, `R1`, `L2`, `R2`).
  - Output spline shafts align laterally with the knee pitch axis to actuate the lower leg tibia/foot shells (`L3`, `R3`, `L4`, `R4`).

---

### 2.5 Link Hierarchy
The kinematic tree of Sesame is structured as follows:

```
world
  └── base_link (chassis frame, distro board, battery, top/bottom covers, OLED)
        ├── fl_femur (Front Left Hip Link) [Joint: fl_hip_joint]
        │     └── fl_tibia (Front Left Lower Leg & Foot) [Joint: fl_knee_joint]
        ├── fr_femur (Front Right Hip Link) [Joint: fr_hip_joint]
        │     └── fr_tibia (Front Right Lower Leg & Foot) [Joint: fr_knee_joint]
        ├── rl_femur (Rear Left Hip Link) [Joint: rl_hip_joint]
        │     └── rl_tibia (Rear Left Lower Leg & Foot) [Joint: rl_knee_joint]
        └── rr_femur (Rear Right Hip Link) [Joint: rr_hip_joint]
              └── rr_tibia (Rear Right Lower Leg & Foot) [Joint: rr_knee_joint]
```

---

### 2.6 Link Dimensions & Geometric Properties
- `[DIRECT REPOSITORY DATA]` **From STEP/STL Analysis:**
  - Base Chassis Bounding Box: $\approx 105 \text{ mm (length)} \times 78 \text{ mm (width)} \times 48 \text{ mm (height)}$.
  - Hip Joint Lateral Spacing (Track Width): $\approx 64 \text{ mm}$ ($Y = \pm 0.032 \text{ m}$).
  - Hip Joint Longitudinal Spacing (Wheelbase): $\approx 72 \text{ mm}$ ($X = \pm 0.036 \text{ m}$).
  - Femur Link Length ($L_{\text{femur}}$ between Hip and Knee axes): $\approx 42.0 \text{ mm}$ ($0.042 \text{ m}$).
  - Tibia Link Length ($L_{\text{tibia}}$ between Knee axis and Foot ground contact): $\approx 46.0 \text{ mm}$ ($0.046 \text{ m}$).
- `[INITIAL SIMULATION ASSUMPTION]` **Mass & Inertia Estimates:**
  - Chassis (including ESP32, OLED, battery, 4 hip servos): $m_{\text{base}} = 0.210 \text{ kg}$.
  - Femur Link (PLA shell + 1 MG90S knee servo + wiring): $m_{\text{femur}} = 0.022 \text{ kg}$.
  - Tibia Link (PLA shell + foot tip): $m_{\text{tibia}} = 0.008 \text{ kg}$.
  - Total Robot Mass: $M_{\text{total}} \approx 0.210 + 4 \times (0.022 + 0.008) = 0.330 \text{ kg}$ ($330 \text{ g}$).

---

### 2.7 Actuator Specifications (TowerPro / Generic MG90S Micro Metal Gear)
- `[DIRECT REPOSITORY DATA]` **BOM Model:** MG90S 9g Micro Metal Gear Servo (180° rotation).
- `[DIRECT REPOSITORY DATA]` **PWM Pulse Width Specification (from `sesame-motor-tester.ino`):**
  - Minimum Pulse Width: $732 \mu\text{s}$
  - Maximum Pulse Width: $2929 \mu\text{s}$
- `[DIRECT REPOSITORY DATA]` **Operating Voltage:** $5.0\text{V} - 6.0\text{V}$ (driven via 5.1V buck converter).
- `[DIRECT REPOSITORY DATA]` **Manufacturer Datasheet Ratings:**
  - Stall Torque: $1.8 \text{ kg}\cdot\text{cm}$ ($0.177 \text{ N}\cdot\text{m}$) at 4.8V, $2.2 \text{ kg}\cdot\text{cm}$ ($0.216 \text{ N}\cdot\text{m}$) at 6.0V.
  - Operating Speed: $0.11 \text{ s}/60^\circ$ ($9.52 \text{ rad/s}$) at 4.8V, $0.08 \text{ s}/60^\circ$ ($13.09 \text{ rad/s}$) at 6.0V.
  - Weight: $13.4 \text{ g}$ per unit.
  - Dead bandwidth: $5 \mu\text{s}$.

---

### 2.8 Existing Electronics, Control & Firmware Architecture
- `[DIRECT REPOSITORY DATA]` **Microcontroller:** ESP32-S2 Mini (or ESP32-WROOM-32E on Distro V1/V2/V3).
- `[DIRECT REPOSITORY DATA]` **Power Subsystem:** 2× 14500 Li-Ion cells in series ($7.4\text{V}$ nominal) regulated to $5.1\text{V} / 3\text{A}$ through a DC-DC buck converter.
- `[DIRECT REPOSITORY DATA]` **Display:** 0.96" I2C SSD1306 128×64 OLED screen ($0\text{x}3\text{C}$).
- `[DIRECT REPOSITORY DATA]` **Firmware Interface:**
  - HTTP REST WebServer & Captive Portal (`/api/command`, `/command`, `/status`, `/settings`).
  - Serial CLI debug interface (115200 baud).
  - Time-stepped keyframe gait execution with linear/cubic motor step delays.

---

### 2.9 Existing Simulation, URDF & CAD Assets
- `[DIRECT REPOSITORY DATA]` **Existing URDF/MJCF/Gazebo Files:** None exist in the upstream repository.
- `[DIRECT REPOSITORY DATA]` **CAD Files Available:**
  - Native Fusion 360 project: `hardware/cad/Sesame-ESP32-v122.f3z`
  - Universal STEP model: `hardware/cad/Sesame-ESP32-v122.step`
  - 3D Printable STL files in `hardware/printing/stl/`.

---

## 3. Physical Parameters Needing Real Hardware Measurement

Prior to executing physical sim-to-real transfer on the actual 3D-printed Sesame robot, the following parameters must be measured on the bench:

1. `[PENDING PHYSICAL BENCH MEASUREMENT]` **Exact Center of Mass (CoM):** Precision knife-edge / 3-scale measurement of the assembled robot chassis with battery installed.
2. `[PENDING PHYSICAL BENCH MEASUREMENT]` **Foot-Ground Friction Coefficients:** Static ($\mu_s$) and dynamic ($\mu_k$) friction coefficients between printed PLA feet (or TPU boots) and test terrain surfaces (wood, carpet, tile).
3. `[PENDING PHYSICAL BENCH MEASUREMENT]` **MG90S Loaded Velocity & Torque Derating Curve:** Torque-speed curve and thermal degradation under continuous quadruped standing/walking loads.
4. `[PENDING PHYSICAL BENCH MEASUREMENT]` **Gear Backlash & Deadband Angle:** Measured angular play in the metal gear trains of the 8 installed MG90S servos under external loads.
5. `[PENDING PHYSICAL BENCH MEASUREMENT]` **End-to-End Latency:** Total control loop latency from Python command issuance over WiFi / Serial / ESP-NOW to servo position change.
