"""
Actuator Diagnostics Widget for MG90S Servo Modeling.

Visualizes parametric actuator non-idealities: Delay, Deadband, Backlash,
Torque/Velocity Saturation, and Sensor Noise.
"""

from typing import Optional, List
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFrame,
    QGroupBox,
)

from robot.parameters import JOINT_NAMES, SERVO_MAPPING


class ActuatorPanel(QWidget):
    """Actuator dynamics diagnostics dashboard."""

    def __init__(self, sim_manager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.sim_manager = sim_manager
        self._init_ui()
        self.sim_manager.sig_telemetry_updated.connect(self.on_telemetry_updated)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)
        
        # 1. Summary Information Card
        info_frame = QFrame()
        info_frame.setObjectName("CardFrame")
        info_frame.setStyleSheet("""
            QFrame#CardFrame {
                background-color: #181b24;
                border: 1px solid #282c3a;
                border-radius: 8px;
                padding: 6px;
            }
        """)
        info_layout = QGridLayout(info_frame)
        info_layout.setContentsMargins(8, 4, 8, 4)
        info_layout.setHorizontalSpacing(14)
        
        lbl_mod = QLabel("Actuator Model:")
        lbl_mod.setStyleSheet("color: #00d2ff; font-weight: 700; font-size: 11px;")
        val_mod = QLabel("TowerPro MG90S (Parametric Non-Linear)")
        val_mod.setStyleSheet("color: #ffffff; font-weight: 600; font-size: 11px;")
        
        lbl_del = QLabel("Modeled Delay:")
        lbl_del.setStyleSheet("color: #94a3b8; font-size: 11px;")
        val_del = QLabel("20 ms (1st-order Lag)")
        val_del.setStyleSheet("color: #e2e8f0; font-family: monospace; font-size: 11px;")
        
        lbl_bak = QLabel("Gear Backlash:")
        lbl_bak.setStyleSheet("color: #94a3b8; font-size: 11px;")
        val_bak = QLabel("±0.86° (Mechanical Play)")
        val_bak.setStyleSheet("color: #e2e8f0; font-family: monospace; font-size: 11px;")
        
        lbl_sat = QLabel("Torque / Speed:")
        lbl_sat.setStyleSheet("color: #94a3b8; font-size: 11px;")
        val_sat = QLabel("0.196 N·m / 600°/s")
        val_sat.setStyleSheet("color: #e2e8f0; font-family: monospace; font-size: 11px;")
        
        info_layout.addWidget(lbl_mod, 0, 0)
        info_layout.addWidget(val_mod, 0, 1)
        info_layout.addWidget(lbl_del, 0, 2)
        info_layout.addWidget(val_del, 0, 3)
        info_layout.addWidget(lbl_bak, 1, 0)
        info_layout.addWidget(val_bak, 1, 1)
        info_layout.addWidget(lbl_sat, 1, 2)
        info_layout.addWidget(val_sat, 1, 3)
        
        main_layout.addWidget(info_frame)
        
        # 2. Live Diagnostics Table
        self.table = QTableWidget(8, 7)
        self.table.setHorizontalHeaderLabels([
            "Servo", "Joint Name", "Target (°)", "Actual (°)", "Vel (°/s)", "Torque (N·m)", "Status"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #141720;
                border: 1px solid #282c3a;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #181b24;
                color: #00d2ff;
                font-weight: 700;
                font-size: 11px;
                padding: 4px;
            }
            QTableWidget::item {
                padding: 2px 4px;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        
        from robot.parameters import STAND_POSE_RAD
        for i, jname in enumerate(JOINT_NAMES):
            servo_id = ""
            for s_name, s_idx in SERVO_MAPPING.items():
                if s_idx == i:
                    servo_id = s_name
                    break
            self.table.setItem(i, 0, QTableWidgetItem(servo_id))
            self.table.setItem(i, 1, QTableWidgetItem(jname.replace("_joint", "").upper()))
            
            stand_deg = float(np.degrees(STAND_POSE_RAD[i]))
            t_item = QTableWidgetItem(f"{stand_deg:.1f}°")
            t_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 2, t_item)
            
            p_item = QTableWidgetItem(f"{stand_deg:.1f}°")
            p_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 3, p_item)
            
            v_item = QTableWidgetItem("0°/s")
            v_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 4, v_item)
            
            tau_item = QTableWidgetItem("+0.000")
            tau_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 5, tau_item)
            
            stat_item = QTableWidgetItem("NOMINAL")
            stat_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            stat_item.setForeground(Qt.GlobalColor.green)
            self.table.setItem(i, 6, stat_item)
                
        main_layout.addWidget(self.table)

    def on_telemetry_updated(self, data: dict):
        pos_list = data.get("joint_positions_deg", [0] * 8)
        tgt_list = data.get("joint_targets_deg", [0] * 8)
        vel_list = data.get("joint_velocities_deg", [0] * 8)
        tor_list = data.get("joint_torques_nm", [0] * 8)
        
        for i in range(8):
            p = pos_list[i] if i < len(pos_list) else 0.0
            t = tgt_list[i] if i < len(tgt_list) else 0.0
            v = vel_list[i] if i < len(vel_list) else 0.0
            tau = tor_list[i] if i < len(tor_list) else 0.0
            
            self.table.item(i, 2).setText(f"{t:.1f}°")
            self.table.item(i, 3).setText(f"{p:.1f}°")
            self.table.item(i, 4).setText(f"{v:+.0f}°/s")
            self.table.item(i, 5).setText(f"{tau:+.3f}")
            
            # Status / Saturation Indicator
            if abs(tau) >= 0.190:
                self.table.item(i, 6).setText("TORQUE SAT")
                self.table.item(i, 6).setForeground(Qt.GlobalColor.red)
            elif abs(v) >= 550.0:
                self.table.item(i, 6).setText("VEL SAT")
                self.table.item(i, 6).setForeground(Qt.GlobalColor.yellow)
            else:
                self.table.item(i, 6).setText("NOMINAL")
                self.table.item(i, 6).setForeground(Qt.GlobalColor.green)
