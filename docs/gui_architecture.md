# Sesame AI Digital Twin — GUI Architecture & Technical Design

## 1. Overview & Core Principles

The **Sesame AI Digital Twin — Control Center** is a professional desktop robotics workstation built using **Python, PyQt6, MuJoCo, and pyqtgraph**. 

It is designed strictly as a **frontend/controller and telemetry monitoring layer** over the verified physics, kinematics, actuator, and RL baseline modules without embedding physics logic in the UI or mutating the underlying simulation code.

---

## 2. Decoupled Threading & Execution Architecture

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         MAIN GUI THREAD (Qt)                            │
  │                                                                         │
  │  ┌─────────────────────────┐         ┌───────────────────────────────┐  │
  │  │   3D Viewport Widget    │         │     Live Telemetry Panel      │  │
  │  │(MuJoCo Renderer @30 FPS)│         │ (State, CoM, Velocities, Dist)│  │
  │  └────────────▲────────────┘         └──────────────▲────────────────┘  │
  │               │                                     │                   │
  │  ┌────────────┴────────────┐         ┌──────────────┴────────────────┐  │
  │  │  Live PyQtGraph Charts  │         │   Research Benchmark Panel    │  │
  │  │ (Reward, Error, Angles) │         │ (Empirical Metric Aggregator) │  │
  │  └────────────▲────────────┘         └──────────────▲────────────────┘  │
  └───────────────┼─────────────────────────────────────┼───────────────────┘
                  │                                     │ Qt Signals / Slots
                  │      sig_telemetry_updated(dict)    │
                  └──────────────────┬──────────────────┘
                                     │
  ┌──────────────────────────────────▼──────────────────────────────────────┐
  │                 SIMULATION MANAGER WORKER THREAD (QThread)              │
  │                                                                         │
  │  ┌───────────────────────────────────────────────────────────────────┐  │
  │  │ 50 Hz Control Loop (dt = 0.020s)                                  │  │
  │  │   1. Retrieve Observation Vector (40-dim)                         │  │
  │  │   2. ControllerManager.compute_action(obs, q, dq, t)              │  │
  │  │   3. Apply Clamped Setpoints to RobotInterface                    │  │
  │  └───────────────────────────────┬───────────────────────────────────┘  │
  │                                  │                                      │
  │  ┌───────────────────────────────▼───────────────────────────────────┐  │
  │  │ 500 Hz MuJoCo Physics Decimation (10 substeps of dt = 0.002s)     │  │
  │  │   - mj_step(model, data)                                          │  │
  │  │   - Contact resolution & actuator torque saturation               │  │
  │  └───────────────────────────────────────────────────────────────────┘  │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
  ┌──────────────────────────────────▼──────────────────────────────────────┐
  │                    ROBOT INTERFACE ABSTRACTION LAYER                    │
  │                                                                         │
  │  ┌────────────────────────────────┐   ┌──────────────────────────────┐  │
  │  │        SimulationRobot         │   │        HardwareRobot         │  │
  │  │ (MuJoCo MjData + ActuatorBank) │   │ (Serial/WiFi ESP32 Driver)   │  │
  │  └────────────────────────────────┘   └──────────────────────────────┘  │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Robot Interface Abstraction Layer

The system defines an abstract base class [`RobotInterface`](file:///c:/Users/Krushna/OneDrive/Documents/sasame%20project%20BE/gui/core/robot_interface.py) that standardizes interaction across both simulation and physical hardware:

```python
class RobotInterface(ABC):
    @abstractmethod
    def get_joint_positions(self) -> np.ndarray: ...
    @abstractmethod
    def get_joint_velocities(self) -> np.ndarray: ...
    @abstractmethod
    def get_joint_torques(self) -> np.ndarray: ...
    @abstractmethod
    def get_base_position(self) -> np.ndarray: ...
    @abstractmethod
    def get_base_euler(self) -> np.ndarray: ...
    @abstractmethod
    def set_joint_targets(self, targets: np.ndarray) -> None: ...
    @abstractmethod
    def emergency_stop(self) -> None: ...
    @abstractmethod
    def reset(self, pose: Optional[np.ndarray] = None) -> None: ...
```

### Implementations:
1. **`SimulationRobot`**: Direct binding with MuJoCo `MjModel` and `MjData` with real-time parametric MG90S non-idealities injection (delay, deadband, backlash, velocity/torque saturation, sensor noise).
2. **`HardwareRobot`**: Concrete driver stub prepared for future WiFi/Serial packet streaming to the ESP32-S2 micro-controller without requiring any changes to the UI or controllers.

---

## 4. Controller Manager & Runtime Switching

The [`ControllerManager`](file:///c:/Users/Krushna/OneDrive/Documents/sasame%20project%20BE/gui/core/controller_manager.py) enables instant switching between classical and reinforcement learning control strategies:

| Controller ID | Type | Description | State |
|---|---|---|---|
| `PID` | Classical Feedback | Anti-windup joint PID tracking Stand, Sine, or Trot gaits | **Active** |
| `PPO` | On-Policy Deep RL | Trained continuous actor-critic reaching policy | **Active** |
| `SAC` | Off-Policy Deep RL | Maximum-entropy off-policy baseline | **Active** |
| `PPO + DR` | Robust Deep RL | Evaluated under mass/friction domain randomization | **Active** |
| `Proposed A3DR` | Research Method | Actuator-Aware Adaptive Domain Randomization | *Pending HW Calibration* |

---

## 5. Viewport Rendering & Performance Optimization

To prevent UI stutter and maintain $50\text{ Hz}$ physics stability:
- **No OpenGL in worker threads:** Physics steps continuously without blocking on graphics synchronization.
- **Main-Thread Rendering (`ViewportWidget`):** Uses `mujoco.Renderer` triggered by a $30\text{ FPS}$ `QTimer` on the main GUI thread, acquiring a recursive mutex lock on `sim_manager.data` only during scene composition.
- **Selective PyQtGraph Updates:** The multi-tab plotting system only updates curves for the currently visible tab.

---

## 6. Safety & Emergency Stop Implementation

When the user presses **`[⛔ EMERGENCY STOP]`** (or `Ctrl+E`):
1. State transitions to `ESTOP`.
2. All actuator control setpoints are immediately zeroed ($\tau = 0, \mathbf{u} = \mathbf{0}$).
3. Controller loop execution halts.
4. UI lockouts prevent sending motion commands until an explicit `[RESET]` is performed.
