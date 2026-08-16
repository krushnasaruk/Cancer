"""
Live PyQtGraph Multi-Tab Plotting Widget.

Provides real-time graphs for:
1. Reward vs Time (Step & Episode returns)
2. Foot-Target Distance vs Time (with reaching threshold line)
3. Joint Tracking Errors (8-curve real-time traces)
4. Target vs Actual Joint Angles
5. Foot Cartesian Trajectory
"""

from typing import Optional, List
from collections import deque
import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QComboBox,
    QLabel,
)

from robot.parameters import JOINT_NAMES

# Configure pyqtgraph styling
pg.setConfigOption("background", "#141720")
pg.setConfigOption("foreground", "#e2e8f0")
pg.setConfigOptions(antialias=True)


class PlotsWidget(QWidget):
    """Multi-tab real-time plotting dashboard."""

    def __init__(self, sim_manager, parent: Optional[QWidget] = None, max_points: int = 300):
        # Support both (sim_manager, parent, max_points) and (sim_manager, max_points, parent)
        if isinstance(parent, int):
            max_points, parent = parent, None
        super().__init__(parent)
        self.sim_manager = sim_manager
        self.max_points = int(max_points)
        
        # Data buffers
        self.time_buf = deque(maxlen=self.max_points)
        self.reward_buf = deque(maxlen=self.max_points)
        self.ep_return_buf = deque(maxlen=self.max_points)
        self.dist_buf = deque(maxlen=self.max_points)
        self.joint_err_buf = [deque(maxlen=self.max_points) for _ in range(8)]
        self.joint_pos_buf = [deque(maxlen=self.max_points) for _ in range(8)]
        self.joint_tgt_buf = [deque(maxlen=self.max_points) for _ in range(8)]
        self.foot_x_buf = deque(maxlen=self.max_points)
        self.foot_z_buf = deque(maxlen=self.max_points)
        
        self.selected_joint_idx = 0
        self._init_ui()
        self.sim_manager.sig_telemetry_updated.connect(self.on_telemetry_updated)
        
        # 15 FPS Plot Refresh Timer on Main GUI Thread
        self.plot_timer = QTimer(self)
        self.plot_timer.setInterval(66)  # ~15 FPS
        self.plot_timer.timeout.connect(self.update_active_plot)
        self.plot_timer.start()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        self.tabs = QTabWidget()
        self.tabs.setObjectName("PlotsTabs")
        
        # Tab 1: Reward & Episode Return
        self.plot_reward = pg.PlotWidget(title="Step Reward & Cumulative Return")
        self.plot_reward.showGrid(x=True, y=True, alpha=0.3)
        self.plot_reward.setLabel("bottom", "Time (s)")
        self.plot_reward.setLabel("left", "Value")
        self.plot_reward.addLegend(offset=(10, 10))
        self.curve_reward = self.plot_reward.plot(pen=pg.mkPen("#00d2ff", width=2), name="Step Reward")
        self.curve_ep_return = self.plot_reward.plot(pen=pg.mkPen("#00f5a0", width=2), name="Episode Return")
        self.tabs.addTab(self.plot_reward, "Reward vs Time")
        
        # Tab 2: Foot-Target Distance
        self.plot_dist = pg.PlotWidget(title="FL Foot to Target Distance (Goal Threshold = 25 mm)")
        self.plot_dist.showGrid(x=True, y=True, alpha=0.3)
        self.plot_dist.setLabel("bottom", "Time (s)")
        self.plot_dist.setLabel("left", "Distance (mm)")
        self.curve_dist = self.plot_dist.plot(pen=pg.mkPen("#f59e0b", width=2), name="Distance (mm)")
        self.line_thresh = pg.InfiniteLine(pos=25.0, angle=0, pen=pg.mkPen("#ef4444", width=1, style=Qt.PenStyle.DashLine))
        self.plot_dist.addItem(self.line_thresh)
        self.tabs.addTab(self.plot_dist, "Target Distance")
        
        # Tab 3: Joint Tracking Errors
        self.plot_joint_err = pg.PlotWidget(title="8-Joint Tracking Errors (deg)")
        self.plot_joint_err.showGrid(x=True, y=True, alpha=0.3)
        self.plot_joint_err.setLabel("bottom", "Time (s)")
        self.plot_joint_err.setLabel("left", "Error (deg)")
        self.plot_joint_err.addLegend(offset=(10, 10))
        colors = ["#00d2ff", "#00f5a0", "#f59e0b", "#ef4444", "#a855f7", "#ec4899", "#3b82f6", "#10b981"]
        self.curves_err = []
        for i in range(8):
            pen = pg.mkPen(colors[i % len(colors)], width=1.5)
            c = self.plot_joint_err.plot(pen=pen, name=f"J{i+1}")
            self.curves_err.append(c)
        self.tabs.addTab(self.plot_joint_err, "Joint Errors")
        
        # Tab 4: Target vs Actual Angle (with Joint selector)
        joint_widget = QWidget()
        joint_layout = QVBoxLayout(joint_widget)
        joint_layout.setContentsMargins(0, 0, 0, 0)
        
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Select Joint:"))
        self.joint_combo = QComboBox()
        self.joint_combo.addItems([
            "1: FR Hip (R1)", "2: RR Hip (R2)", "3: FL Hip (L1)", "4: RL Hip (L2)",
            "5: RR Knee (R4)", "6: FR Knee (R3)", "7: FL Knee (L3)", "8: RL Knee (L4)",
        ])
        self.joint_combo.currentIndexChanged.connect(self._on_joint_combo_changed)
        top_bar.addWidget(self.joint_combo)
        top_bar.addStretch()
        joint_layout.addLayout(top_bar)
        
        self.plot_angle = pg.PlotWidget(title="Joint Target vs Actual Angle")
        self.plot_angle.showGrid(x=True, y=True, alpha=0.3)
        self.plot_angle.setLabel("bottom", "Time (s)")
        self.plot_angle.setLabel("left", "Angle (deg)")
        self.plot_angle.addLegend(offset=(10, 10))
        self.curve_actual = self.plot_angle.plot(pen=pg.mkPen("#00d2ff", width=2), name="Actual Angle")
        self.curve_target = self.plot_angle.plot(pen=pg.mkPen("#f59e0b", width=2, style=Qt.PenStyle.DashLine), name="Target Angle")
        joint_layout.addWidget(self.plot_angle)
        self.tabs.addTab(joint_widget, "Target vs Actual Angle")
        
        # Tab 5: Foot Trajectory (X vs Z)
        self.plot_traj = pg.PlotWidget(title="FL Foot Trajectory (Sagittal X-Z Plane)")
        self.plot_traj.showGrid(x=True, y=True, alpha=0.3)
        self.plot_traj.setLabel("bottom", "X Position (m)")
        self.plot_traj.setLabel("left", "Z Position (m)")
        self.curve_traj = self.plot_traj.plot(pen=pg.mkPen("#00f5a0", width=2), symbol="o", symbolSize=4, symbolBrush="#00d2ff")
        self.tabs.addTab(self.plot_traj, "Foot Trajectory")
        
        layout.addWidget(self.tabs)

    def _on_joint_combo_changed(self, idx: int):
        self.selected_joint_idx = max(0, min(7, idx))

    def on_telemetry_updated(self, data: dict):
        # Fast append without redrawing
        t = data.get("sim_time", 0.0)
        self.time_buf.append(t)
        self.reward_buf.append(data.get("reward", 0.0))
        self.ep_return_buf.append(data.get("episode_return", 0.0))
        self.dist_buf.append(data.get("dist_to_target", 0.0) * 1000.0)
        
        errs = data.get("joint_errors_deg", [0] * 8)
        pos = data.get("joint_positions_deg", [0] * 8)
        tgt = data.get("joint_targets_deg", [0] * 8)
        for i in range(8):
            if i < len(errs):
                self.joint_err_buf[i].append(errs[i])
                self.joint_pos_buf[i].append(pos[i])
                self.joint_tgt_buf[i].append(tgt[i])
                
        ee = data.get("ee_xyz", [0, 0, 0])
        self.foot_x_buf.append(ee[0])
        self.foot_z_buf.append(ee[2])

    def update_active_plot(self):
        # Only update the currently active tab at ~15 FPS
        if not self.isVisible() or len(self.time_buf) < 2:
            return
            
        current_tab = self.tabs.currentIndex()
        t_arr = np.array(self.time_buf)
        
        if current_tab == 0:
            self.curve_reward.setData(t_arr, np.array(self.reward_buf))
            self.curve_ep_return.setData(t_arr, np.array(self.ep_return_buf))
        elif current_tab == 1:
            self.curve_dist.setData(t_arr, np.array(self.dist_buf))
        elif current_tab == 2:
            for i in range(8):
                if len(self.joint_err_buf[i]) == len(t_arr):
                    self.curves_err[i].setData(t_arr, np.array(self.joint_err_buf[i]))
        elif current_tab == 3:
            j_idx = self.selected_joint_idx
            if len(self.joint_pos_buf[j_idx]) == len(t_arr):
                self.curve_actual.setData(t_arr, np.array(self.joint_pos_buf[j_idx]))
                self.curve_target.setData(t_arr, np.array(self.joint_tgt_buf[j_idx]))
        elif current_tab == 4:
            if len(self.foot_x_buf) == len(self.foot_z_buf) and len(self.foot_x_buf) > 0:
                self.curve_traj.setData(np.array(self.foot_x_buf), np.array(self.foot_z_buf))
