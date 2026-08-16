"""
Telemetry Panel Widget for Real-Time State & Metric Monitoring.
"""

from typing import Optional, Dict, Any
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QFrame,
    QGroupBox,
    QScrollArea,
)


class MetricCard(QFrame):
    """Clean telemetry metric card with key-value pairs."""

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("CardFrame")
        self.setStyleSheet("""
            QFrame#CardFrame {
                background-color: #181b24;
                border: 1px solid #282c3a;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 6, 8, 6)
        self.layout.setSpacing(4)
        
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("color: #00d2ff; font-weight: 700; font-size: 11px; text-transform: uppercase;")
        self.layout.addWidget(self.lbl_title)
        
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 4, 0, 0)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(3)
        self.layout.addLayout(self.grid)
        
        self.rows: Dict[str, QLabel] = {}

    def add_metric(self, key: str, label: str, default_val: str = "--"):
        row = len(self.rows)
        lbl_k = QLabel(label)
        lbl_k.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 500;")
        
        lbl_v = QLabel(default_val)
        lbl_v.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: 700; font-family: monospace;")
        lbl_v.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.grid.addWidget(lbl_k, row, 0)
        self.grid.addWidget(lbl_v, row, 1)
        self.rows[key] = lbl_v

    def update_metric(self, key: str, value_str: str):
        if key in self.rows:
            self.rows[key].setText(value_str)


class TelemetryPanel(QWidget):
    """Right-hand side live telemetry dashboard."""

    def __init__(self, sim_manager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.sim_manager = sim_manager
        self.setMinimumWidth(260)
        self._init_ui()
        self.sim_manager.sig_telemetry_updated.connect(self.on_telemetry_updated)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(8)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background-color: transparent;")
        
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(8)
        
        # 1. System & Performance Card
        self.card_sys = MetricCard("System & Rates")
        self.card_sys.add_metric("state", "Sim State:", "READY")
        self.card_sys.add_metric("time", "Sim Time:", "0.00 s")
        self.card_sys.add_metric("fps", "UI Rendering FPS:", "30.0 FPS")
        self.card_sys.add_metric("physics_hz", "Physics Rate:", "500 Hz")
        self.card_sys.add_metric("control_hz", "Control Rate:", "50 Hz")
        self.card_sys.add_metric("steps_sec", "Throughput:", "0 steps/s")
        container_layout.addWidget(self.card_sys)
        
        # 2. Robot Base Telemetry Card
        self.card_base = MetricCard("Base Kinematics")
        self.card_base.add_metric("pos", "Base Pos (X, Y, Z):", "[0.00, 0.00, 0.06]")
        self.card_base.add_metric("euler", "Orientation (R, P, Y):", "[+0.0°, +0.0°, +0.0°]")
        self.card_base.add_metric("lin_vel", "Linear Vel (m/s):", "[+0.00, +0.00, +0.00]")
        self.card_base.add_metric("ang_vel", "Angular Vel (°/s):", "[+0.0, +0.0, +0.0]")
        self.card_base.add_metric("com", "Center of Mass:", "[0.00, 0.00, 0.06]")
        self.card_base.add_metric("ssm", "Static Stability:", "+35.0 mm")
        container_layout.addWidget(self.card_base)
        
        # 3. Target Reaching Task Card
        self.card_target = MetricCard("Reaching Task (FL Foot)")
        self.card_target.add_metric("target_pos", "Target Pos (m):", "[+0.10, +0.00, +0.02]")
        self.card_target.add_metric("ee_pos", "Foot Pos (m):", "[+0.04, +0.04, +0.00]")
        self.card_target.add_metric("dist", "Foot-Target Dist:", "78.5 mm")
        self.card_target.add_metric("status", "Task Status:", "IN TRANSIT")
        container_layout.addWidget(self.card_target)
        
        # 4. Reinforcement Learning Performance Card
        self.card_rl = MetricCard("Reward & Return")
        self.card_rl.add_metric("reward", "Step Reward:", "+0.00")
        self.card_rl.add_metric("ep_return", "Episode Return:", "+0.00")
        self.card_rl.add_metric("ep_steps", "Episode Steps:", "0")
        container_layout.addWidget(self.card_rl)
        
        container_layout.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def on_telemetry_updated(self, data: dict):
        # Update System
        self.card_sys.update_metric("state", data.get("state", "--"))
        self.card_sys.update_metric("time", f"{data.get('sim_time', 0.0):.2f} s")
        self.card_sys.update_metric("fps", f"{data.get('fps', 0.0):.1f} FPS")
        self.card_sys.update_metric("physics_hz", "500 Hz")
        self.card_sys.update_metric("control_hz", "50 Hz")
        self.card_sys.update_metric("steps_sec", f"{data.get('steps_per_sec', 0.0):.0f} steps/s")
        
        # Update Base
        bx, by, bz = data.get("base_xyz", [0, 0, 0])
        self.card_base.update_metric("pos", f"[{bx:.2f}, {by:.2f}, {bz:.2f}]")
        
        r, p, y = data.get("base_euler_deg", [0, 0, 0])
        self.card_base.update_metric("euler", f"[{r:+.1f}°, {p:+.1f}°, {y:+.1f}°]")
        
        vx, vy, vz = data.get("base_lin_vel", [0, 0, 0])
        self.card_base.update_metric("lin_vel", f"[{vx:+.2f}, {vy:+.2f}, {vz:+.2f}]")
        
        wx, wy, wz = data.get("base_ang_vel", [0, 0, 0])
        self.card_base.update_metric("ang_vel", f"[{wx:+.1f}, {wy:+.1f}, {wz:+.1f}]")
        
        cx, cy, cz = data.get("com_xyz", [0, 0, 0])
        self.card_base.update_metric("com", f"[{cx:.2f}, {cy:.2f}, {cz:.2f}]")
        
        ssm = data.get("ssm_margin", 0.0)
        self.card_base.update_metric("ssm", f"{ssm * 1000.0:+.1f} mm")
        
        # Update Target Reaching
        tx, ty, tz = data.get("target_xyz", [0, 0, 0])
        self.card_target.update_metric("target_pos", f"[{tx:.2f}, {ty:.2f}, {tz:.2f}]")
        
        ex, ey, ez = data.get("ee_xyz", [0, 0, 0])
        self.card_target.update_metric("ee_pos", f"[{ex:.2f}, {ey:.2f}, {ez:.2f}]")
        
        dist = data.get("dist_to_target", 0.0)
        self.card_target.update_metric("dist", f"{dist * 1000.0:.1f} mm")
        
        is_success = data.get("is_success", False)
        status_text = "TARGET REACHED" if is_success else "IN TRANSIT"
        self.card_target.update_metric("status", status_text)
        
        # Update RL
        self.card_rl.update_metric("reward", f"{data.get('reward', 0.0):+.2f}")
        self.card_rl.update_metric("ep_return", f"{data.get('episode_return', 0.0):+.2f}")
        self.card_rl.update_metric("ep_steps", f"{data.get('episode_steps', 0)}")
