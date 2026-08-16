"""
Joint Monitoring Panel Widget.

Displays all 8 actuated joints with real-time target, actual angle, tracking error,
velocity, applied torque, and graphical range bars.
"""

from typing import Optional, Dict, List
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QProgressBar,
    QFrame,
    QScrollArea,
)

from robot.parameters import (
    JOINT_NAMES,
    JOINT_LIMITS_RAD,
    SERVO_MAPPING,
)


class JointCard(QFrame):
    """Card displaying live status of a single actuated joint."""

    def __init__(self, joint_name: str, index: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.joint_name = joint_name
        self.index = index
        
        self.min_rad, self.max_rad = JOINT_LIMITS_RAD[joint_name]
        self.min_deg = np.degrees(self.min_rad)
        self.max_deg = np.degrees(self.max_rad)
        
        # Look up servo label (e.g. "R1", "L3")
        self.servo_label = ""
        for s_name, s_idx in SERVO_MAPPING.items():
            if s_idx == index:
                self.servo_label = s_name
                break
                
        self.setObjectName("CardFrame")
        self.setStyleSheet("""
            QFrame#CardFrame {
                background-color: #181b24;
                border: 1px solid #282c3a;
                border-radius: 8px;
                padding: 6px;
            }
        """)
        
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(3)
        
        # Header: Joint Name and Servo Tag
        header_layout = QHBoxLayout()
        readable_name = self.joint_name.replace("_joint", "").upper().replace("_", " ")
        lbl_name = QLabel(f"{readable_name} ({self.servo_label})")
        lbl_name.setStyleSheet("color: #00d2ff; font-weight: 700; font-size: 11px;")
        
        self.lbl_error = QLabel("Δ 0.0°")
        self.lbl_error.setStyleSheet("color: #10b981; font-weight: 700; font-size: 11px; font-family: monospace;")
        
        header_layout.addWidget(lbl_name)
        header_layout.addStretch()
        header_layout.addWidget(self.lbl_error)
        layout.addLayout(header_layout)
        
        # Metrics Grid: Current, Target, Vel, Torque
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        
        from robot.parameters import STAND_POSE_RAD
        stand_deg = float(np.degrees(STAND_POSE_RAD[self.index]))
        
        lbl_c_tag = QLabel("Cur:")
        lbl_c_tag.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self.lbl_curr = QLabel(f"{stand_deg:.1f}°")
        self.lbl_curr.setStyleSheet("color: #ffffff; font-weight: 700; font-size: 10px; font-family: monospace;")
        
        lbl_t_tag = QLabel("Tgt:")
        lbl_t_tag.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self.lbl_tgt = QLabel(f"{stand_deg:.1f}°")
        self.lbl_tgt.setStyleSheet("color: #e2e8f0; font-weight: 600; font-size: 10px; font-family: monospace;")
        
        lbl_v_tag = QLabel("Vel:")
        lbl_v_tag.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self.lbl_vel = QLabel("0.0°/s")
        self.lbl_vel.setStyleSheet("color: #cbd5e1; font-size: 10px; font-family: monospace;")
        
        grid.addWidget(lbl_c_tag, 0, 0)
        grid.addWidget(self.lbl_curr, 0, 1)
        grid.addWidget(lbl_t_tag, 0, 2)
        grid.addWidget(self.lbl_tgt, 0, 3)
        grid.addWidget(lbl_v_tag, 0, 4)
        grid.addWidget(self.lbl_vel, 0, 5)
        layout.addLayout(grid)
        
        # Graphical Range Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(int(self.min_deg), int(self.max_deg))
        self.progress_bar.setValue(int(self.min_deg))
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #232734;
                border: 1px solid #32384a;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #00d2ff;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

    def update_joint(self, curr_deg: float, tgt_deg: float, err_deg: float, vel_deg: float, torque_nm: float):
        self.lbl_curr.setText(f"{curr_deg:.1f}°")
        self.lbl_tgt.setText(f"{tgt_deg:.1f}°")
        self.lbl_vel.setText(f"{vel_deg:+.0f}°/s")
        
        # Format Error with color coding
        abs_err = abs(err_deg)
        if abs_err < 5.0:
            self.lbl_error.setStyleSheet("color: #10b981; font-weight: 700; font-size: 11px; font-family: monospace;")
        elif abs_err < 15.0:
            self.lbl_error.setStyleSheet("color: #f59e0b; font-weight: 700; font-size: 11px; font-family: monospace;")
        else:
            self.lbl_error.setStyleSheet("color: #ef4444; font-weight: 700; font-size: 11px; font-family: monospace;")
        self.lbl_error.setText(f"Δ {err_deg:+.1f}°")
        
        # Update Range Bar
        clamped_val = int(np.clip(curr_deg, self.min_deg, self.max_deg))
        self.progress_bar.setValue(clamped_val)


class JointPanel(QWidget):
    """Panel hosting all 8 joint cards arranged in a responsive grid."""

    def __init__(self, sim_manager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.sim_manager = sim_manager
        self._init_ui()
        self.sim_manager.sig_telemetry_updated.connect(self.on_telemetry_updated)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background-color: transparent;")
        
        container = QWidget()
        self.grid_layout = QGridLayout(container)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(6)
        
        self.joint_cards: List[JointCard] = []
        for i, jname in enumerate(JOINT_NAMES):
            card = JointCard(jname, i)
            row = i // 2
            col = i % 2
            self.grid_layout.addWidget(card, row, col)
            self.joint_cards.append(card)
            
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def on_telemetry_updated(self, data: dict):
        pos_list = data.get("joint_positions_deg", [0] * 8)
        tgt_list = data.get("joint_targets_deg", [0] * 8)
        err_list = data.get("joint_errors_deg", [0] * 8)
        vel_list = data.get("joint_velocities_deg", [0] * 8)
        tor_list = data.get("joint_torques_nm", [0] * 8)
        
        for i in range(min(8, len(self.joint_cards))):
            self.joint_cards[i].update_joint(
                pos_list[i], tgt_list[i], err_list[i], vel_list[i], tor_list[i]
            )
