"""
Executive KPI Header Cards (Finnova-inspired design with micro-visualizations).

Matches the 4 KPI cards from the reference dashboard with custom icon badges,
mini bar charts, sparkline trajectories, and controller pills.
"""

from typing import Optional
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QFrame,
    QProgressBar,
)


class FinnovaKpiCard(QFrame):
    """Modern rounded glass KPI card matching Finnova executive design."""

    def __init__(
        self,
        title: str,
        initial_value: str,
        subtitle: str,
        icon: str = "📊",
        icon_bg: str = "#ede9fe",
        icon_color: str = "#6366f1",
        trend: str = "",
        trend_positive: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("KpiCard")
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(14, 12, 14, 12)
        self.layout.setSpacing(6)
        
        # Header Row: Title + Circular Icon Badge
        top_row = QHBoxLayout()
        lbl_title = QLabel(title)
        lbl_title.setObjectName("kpi_title")
        
        lbl_icon = QLabel(icon)
        lbl_icon.setObjectName("kpi_icon")
        lbl_icon.setStyleSheet(f"""
            background-color: {icon_bg};
            color: {icon_color};
            border-radius: 12px;
            padding: 3px 7px;
            font-size: 11px;
            font-weight: 800;
        """)
        
        top_row.addWidget(lbl_title)
        top_row.addStretch()
        top_row.addWidget(lbl_icon)
        self.layout.addLayout(top_row)
        
        # Big Metric Value
        self.lbl_val = QLabel(initial_value)
        self.lbl_val.setObjectName("kpi_value")
        self.layout.addWidget(self.lbl_val)
        
        # Footer Row: Subtitle + Trend Chip
        self.bottom_row = QHBoxLayout()
        self.lbl_sub = QLabel(subtitle)
        self.lbl_sub.setObjectName("kpi_sub")
        
        self.lbl_trend = QLabel(trend)
        trend_color = "#10b981" if trend_positive else "#ef4444"
        self.lbl_trend.setStyleSheet(f"color: {trend_color}; font-size: 11px; font-weight: 700;")
        
        self.bottom_row.addWidget(self.lbl_sub)
        self.bottom_row.addStretch()
        if trend:
            self.bottom_row.addWidget(self.lbl_trend)
        self.layout.addLayout(self.bottom_row)

    def set_value(self, val_str: str, sub_str: Optional[str] = None, trend_str: Optional[str] = None):
        self.lbl_val.setText(val_str)
        if sub_str is not None:
            self.lbl_sub.setText(sub_str)
        if trend_str is not None:
            self.lbl_trend.setText(trend_str)


class ExecutiveKpiRow(QWidget):
    """Row of 4 KPI cards matching the Finnova executive dashboard header."""

    def __init__(self, sim_manager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.sim_manager = sim_manager
        self._init_ui()
        self.sim_manager.sig_telemetry_updated.connect(self._on_telemetry)

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Card 1: Base Kinematics (Red/Orange theme matching Card 1 in reference)
        self.card1 = FinnovaKpiCard(
            title="Base Kinematics",
            initial_value="Z: 65.0 mm",
            subtitle="● Nominal Upright",
            icon="!",
            icon_bg="#fee2e2",
            icon_color="#ef4444",
            trend="↑ 0.0° Balance",
            trend_positive=True,
        )
        
        # Card 2: FL Target Precision (Purple theme matching Card 2 in reference)
        self.card2 = FinnovaKpiCard(
            title="FL Reaching Target",
            initial_value="81.3 mm",
            subtitle="Target [0.10, 0.00, 0.02]",
            icon="📅",
            icon_bg="#ede9fe",
            icon_color="#6366f1",
            trend="↑ Peak Precision",
            trend_positive=True,
        )
        
        # Card 3: Actuator Load & Lag (Cyan theme matching Card 3 in reference)
        self.card3 = FinnovaKpiCard(
            title="Actuator Load & Lag",
            initial_value="0.042 N·m",
            subtitle="20 ms lag | ±0.86° backlash",
            icon="⏱",
            icon_bg="#cffafe",
            icon_color="#06b6d4",
            trend="MG90S Normal",
            trend_positive=True,
        )
        
        # Card 4: AI Episode Return (Green theme matching Card 4 in reference)
        self.card4 = FinnovaKpiCard(
            title="AI Episode Return",
            initial_value="+9,074",
            subtitle="PPO 300k Production Checkpoint",
            icon="🔒",
            icon_bg="#dcfce7",
            icon_color="#10b981",
            trend="↑ +425% vs Baseline",
            trend_positive=True,
        )
        
        layout.addWidget(self.card1)
        layout.addWidget(self.card2)
        layout.addWidget(self.card3)
        layout.addWidget(self.card4)

    def _on_telemetry(self, data: dict):
        bx, by, bz = data.get("base_xyz", [0, 0, 0.065])
        r, p, y = data.get("base_euler_deg", [0, 0, 0])
        self.card1.set_value(f"Z: {bz*1000.0:.1f} mm", f"Roll: {r:+.0f}° | Pitch: {p:+.0f}°")
        
        dist = data.get("dist_to_target", 0.081) * 1000.0
        status = "✓ REACHED" if data.get("is_success", False) else "In Transit"
        self.card2.set_value(f"{dist:.1f} mm", f"Status: {status}")
        
        tor_list = data.get("joint_torques_nm", [0.042] * 8)
        max_tor = max(abs(t) for t in tor_list) if tor_list else 0.042
        self.card3.set_value(f"{max_tor:.3f} N·m", "MG90S Servo Bank")
        
        ep_ret = data.get("episode_return", 9074.0)
        self.card4.set_value(f"{ep_ret:+.0f}", f"Step: {data.get('episode_steps', 0)} | 30 FPS")
