"""
Main Application Window for Sesame AI Digital Twin — Control Center.

Features a spacious, breathable, Viewport-First architecture:
- Top: Clean, unclipped Navigation Bar
- Left / Center: Massive high-resolution 3D Robot Twin Viewport (~75% screen)
- Right: Clean Tabbed Control & Analytics Sidebar (Telemetry, Servos, Charts, Benchmarks)
"""

import os
import sys
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QTabWidget,
    QStatusBar,
    QMenuBar,
    QMenu,
    QMessageBox,
)
from PyQt6.QtGui import QAction

from gui.core.simulation_manager import SimulationManager
from gui.widgets.control_bar import ControlBarWidget
from gui.widgets.viewport_widget import ViewportWidget
from gui.widgets.telemetry_panel import TelemetryPanel
from gui.widgets.joint_panel import JointPanel
from gui.widgets.actuator_panel import ActuatorPanel
from gui.widgets.plots_widget import PlotsWidget
from gui.widgets.research_dashboard import ResearchDashboard


class SesameControlCenterWindow(QMainWindow):
    """Main window of the Sesame AI Digital Twin Control Center."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Sesame AI Digital Twin - Sim-to-Real Control Center")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 720)
        
        # 1. Initialize Simulation Manager Thread
        self.sim_manager = SimulationManager(self)
        
        # 2. Build UI Layout
        self._init_ui()
        self._init_menu_and_status()
        self._load_stylesheet()
        
        # 3. Start Simulation Worker Thread
        self.sim_manager.start()

    def _init_ui(self):
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(12, 12, 12, 12)
        central_layout.setSpacing(10)
        
        # 1. Top: Clean Spacious Control Bar
        self.control_bar = ControlBarWidget(self.sim_manager, self)
        self.control_bar.sig_theme_changed.connect(self._apply_theme)
        central_layout.addWidget(self.control_bar)
        
        # 2. Main Horizontal Splitter (Left: Big Viewport, Right: Clean Tabbed Sidebar)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(8)
        
        # Left: Massive 3D Viewport Hero Stage
        self.viewport = ViewportWidget(self.sim_manager, self)
        main_splitter.addWidget(self.viewport)
        
        # Right: Tabbed Analytics & Control Sidebar
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setObjectName("SidebarTabs")
        
        self.telemetry_panel = TelemetryPanel(self.sim_manager, self)
        self.joint_panel = JointPanel(self.sim_manager, self)
        self.plots_widget = PlotsWidget(self.sim_manager, self)
        self.actuator_panel = ActuatorPanel(self.sim_manager, self)
        self.research_dash = ResearchDashboard(self)
        
        self.sidebar_tabs.addTab(self.telemetry_panel, "📊 Telemetry")
        self.sidebar_tabs.addTab(self.joint_panel, "🦾 8-Joints")
        self.sidebar_tabs.addTab(self.plots_widget, "📈 Charts")
        self.sidebar_tabs.addTab(self.actuator_panel, "⚙️ Actuators")
        self.sidebar_tabs.addTab(self.research_dash, "🔬 Benchmarks")
        
        main_splitter.addWidget(self.sidebar_tabs)
        
        # Sizing: 72% for Viewport (1050px), 28% for Sidebar (390px)
        main_splitter.setSizes([1050, 390])
        
        central_layout.addWidget(main_splitter)
        self.setCentralWidget(central_widget)

    def _init_menu_and_status(self):
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        act_reset = QAction("↺ Reset Simulation", self)
        act_reset.setShortcut("Ctrl+R")
        act_reset.triggered.connect(self.sim_manager.reset_sim)
        file_menu.addAction(act_reset)
        
        act_estop = QAction("⛔ Emergency Stop", self)
        act_estop.setShortcut("Ctrl+E")
        act_estop.triggered.connect(self.sim_manager.emergency_stop)
        file_menu.addAction(act_estop)
        
        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)
        
        # View Menu
        view_menu = menubar.addMenu("&View")
        act_iso = QAction("🌐 Isometric View", self)
        act_iso.triggered.connect(self.viewport.set_isometric_view)
        view_menu.addAction(act_iso)
        
        act_front = QAction("👁️ Front View", self)
        act_front.triggered.connect(self.viewport.set_front_view)
        view_menu.addAction(act_front)
        
        act_side = QAction("📐 Side View", self)
        act_side.triggered.connect(self.viewport.set_side_view)
        view_menu.addAction(act_side)
        
        act_top = QAction("🔝 Top View", self)
        act_top.triggered.connect(self.viewport.set_top_view)
        view_menu.addAction(act_top)
        
        # Help Menu
        help_menu = menubar.addMenu("&Help")
        act_about = QAction("About Sesame Twin...", self)
        act_about.triggered.connect(self._show_about_dialog)
        help_menu.addAction(act_about)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Sesame AI Digital Twin Ready | Sim-to-Real Multi-Task Mode Active")
        self.sim_manager.sig_telemetry_updated.connect(self._update_status_bar)

    def _update_status_bar(self, data: dict):
        fps = self.sim_manager.current_fps
        ctrl = self.sim_manager.controller_manager.active_type
        bz = data.get("base_xyz", [0, 0, 0.065])[2] * 1000.0
        dist = data.get("dist_to_target", 0.0) * 1000.0
        msg = f"Status: {self.sim_manager.state}  |  Controller: {ctrl}  |  FPS: {fps:.0f}  |  Base Z: {bz:.1f} mm  |  Target Dist: {dist:.1f} mm"
        self.status_bar.showMessage(msg)

    def _load_stylesheet(self):
        self._apply_theme("🌙 Dark Mode")

    def _apply_theme(self, theme_name: str):
        theme_file_map = {
            "🌙 Dark Mode": "dark_theme.qss",
            "☀️ Light Mode": "light_theme.qss",
        }
        filename = theme_file_map.get(theme_name, "dark_theme.qss")
        qss_path = os.path.join(os.path.dirname(__file__), "styles", filename)
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    def _show_about_dialog(self):
        QMessageBox.about(
            self,
            "About Sesame Digital Twin",
            "<h3>Sesame Quadruped AI Digital Twin</h3>"
            "<p><b>B.E. Final Year Research Project</b></p>"
            "<p>Sim-to-Real Digital Twin with 3D CAD Meshes, Non-Linear Actuator Dynamics, "
            "and Deep Reinforcement Learning (PPO / SAC).</p>"
            "<p>Official Architecture: <b>dorianborian / sesame-robot</b></p>",
        )

    def closeEvent(self, event):
        self.sim_manager.stop_worker()
        event.accept()


def main():
    """Launch Sesame Control Center Application."""
    app = QApplication(sys.argv)
    window = SesameControlCenterWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
