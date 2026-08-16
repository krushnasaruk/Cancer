"""
Clean & Breathable Top Navigation Bar for Sesame Digital Twin GUI.

Provides clear, unclipped controls with generous spacing:
- Transport Deck: [ ▶ Run ] [ ⏸ Pause ] [ ↺ Reset ] [ ⛔ E-STOP ]
- Controller & Mode Selectors
- Environment & Theme Toggles
"""

from typing import Optional
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QSlider,
    QFrame,
)

from gui.core.controller_manager import ControllerType


class ControlBarWidget(QFrame):
    """Spacious, breathable modern header navigation bar."""

    sig_theme_changed = pyqtSignal(str)

    def __init__(self, sim_manager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.sim_manager = sim_manager
        self.setObjectName("ControlBarFrame")
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 8, 16, 8)
        main_layout.setSpacing(16)
        
        # 1. Logo & App Title
        brand_box = QHBoxLayout()
        brand_box.setSpacing(8)
        lbl_logo = QLabel("🤖")
        lbl_logo.setStyleSheet("font-size: 20px;")
        
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        lbl_title = QLabel("SESAME DIGITAL TWIN")
        lbl_title.setObjectName("AppTitle")
        lbl_title.setStyleSheet("font-weight: 800; font-size: 13px; letter-spacing: 0.8px;")
        
        lbl_sub = QLabel("Sim-to-Real Control Center")
        lbl_sub.setObjectName("AppSubtitle")
        lbl_sub.setStyleSheet("font-size: 10px; color: #64748b; font-weight: 500;")
        
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        brand_box.addWidget(lbl_logo)
        brand_box.addLayout(title_box)
        main_layout.addLayout(brand_box)
        
        main_layout.addSpacing(10)
        
        # 2. Spacious Transport Buttons (No overlapping frames)
        transport_box = QHBoxLayout()
        transport_box.setSpacing(8)
        
        self.btn_start = QPushButton("▶  Run")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setMinimumWidth(80)
        self.btn_start.setFixedHeight(34)
        
        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setObjectName("btn_pause")
        self.btn_pause.setMinimumWidth(80)
        self.btn_pause.setFixedHeight(34)
        
        self.btn_reset = QPushButton("↺  Reset")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.setMinimumWidth(80)
        self.btn_reset.setFixedHeight(34)
        
        self.btn_estop = QPushButton("⛔  E-STOP")
        self.btn_estop.setObjectName("btn_estop")
        self.btn_estop.setMinimumWidth(85)
        self.btn_estop.setFixedHeight(34)
        
        transport_box.addWidget(self.btn_start)
        transport_box.addWidget(self.btn_pause)
        transport_box.addWidget(self.btn_reset)
        transport_box.addWidget(self.btn_estop)
        main_layout.addLayout(transport_box)
        
        main_layout.addSpacing(10)
        
        # 3. Controller Selector
        ctrl_box = QHBoxLayout()
        ctrl_box.setSpacing(6)
        lbl_ctrl = QLabel("Controller:")
        lbl_ctrl.setStyleSheet("font-weight: 700; font-size: 11px;")
        
        self.combo_controller = QComboBox()
        self.combo_controller.addItem(ControllerType.PID)
        self.combo_controller.addItem(ControllerType.PPO)
        self.combo_controller.addItem(ControllerType.PPO_WALK)
        self.combo_controller.addItem(ControllerType.SAC)
        self.combo_controller.addItem(ControllerType.PPO_DR)
        self.combo_controller.setMinimumWidth(130)
        self.combo_controller.setFixedHeight(34)
        
        ctrl_box.addWidget(lbl_ctrl)
        ctrl_box.addWidget(self.combo_controller)
        main_layout.addLayout(ctrl_box)
        
        # PID Gait Sub-Mode
        self.gait_box = QHBoxLayout()
        self.gait_box.setSpacing(6)
        self.lbl_gait = QLabel("Gait:")
        self.lbl_gait.setStyleSheet("font-weight: 700; font-size: 11px;")
        self.combo_pid_mode = QComboBox()
        self.combo_pid_mode.addItems(["STAND", "WALK", "SINE"])
        self.combo_pid_mode.setMinimumWidth(85)
        self.combo_pid_mode.setFixedHeight(34)
        self.gait_box.addWidget(self.lbl_gait)
        self.gait_box.addWidget(self.combo_pid_mode)
        main_layout.addLayout(self.gait_box)
        
        main_layout.addStretch()
        
        # 4. Right Controls: Environment, Speed & Theme
        right_box = QHBoxLayout()
        right_box.setSpacing(12)
        
        # Environment
        lbl_env = QLabel("Env:")
        lbl_env.setStyleSheet("font-weight: 700; font-size: 11px;")
        self.combo_env = QComboBox()
        self.combo_env.addItems(["Arena (Std)", "Robotics Lab", "Carpet", "Asphalt"])
        self.combo_env.setMinimumWidth(110)
        self.combo_env.setFixedHeight(34)
        right_box.addWidget(lbl_env)
        right_box.addWidget(self.combo_env)
        
        # Speed Slider
        lbl_spd = QLabel("Speed:")
        lbl_spd.setStyleSheet("font-weight: 700; font-size: 11px;")
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setRange(1, 30)
        self.slider_speed.setValue(10)
        self.slider_speed.setFixedWidth(70)
        self.lbl_speed_val = QLabel("1.0x")
        self.lbl_speed_val.setStyleSheet("font-weight: 700; font-size: 11px; min-width: 28px;")
        right_box.addWidget(lbl_spd)
        right_box.addWidget(self.slider_speed)
        right_box.addWidget(self.lbl_speed_val)
        
        # Theme Selector
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(["🌙 Dark Mode", "☀️ Light Mode"])
        self.combo_theme.setMinimumWidth(115)
        self.combo_theme.setFixedHeight(34)
        right_box.addWidget(self.combo_theme)
        
        main_layout.addLayout(right_box)

    def _connect_signals(self):
        self.btn_start.clicked.connect(self.sim_manager.start_sim)
        self.btn_pause.clicked.connect(self.sim_manager.pause_sim)
        self.btn_reset.clicked.connect(self.sim_manager.reset_sim)
        self.btn_estop.clicked.connect(self.sim_manager.emergency_stop)
        
        self.combo_controller.currentTextChanged.connect(self._on_controller_changed)
        self.combo_pid_mode.currentTextChanged.connect(self._on_pid_mode_changed)
        self.combo_env.currentTextChanged.connect(self._on_env_changed)
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        self.combo_theme.currentTextChanged.connect(self.sig_theme_changed.emit)
        
        self.sim_manager.sig_status_changed.connect(self._on_state_changed)

    def _on_controller_changed(self, text: str):
        self.sim_manager.set_controller(text)
        is_pid = (text == ControllerType.PID)
        self.lbl_gait.setVisible(is_pid)
        self.combo_pid_mode.setVisible(is_pid)

    def _on_pid_mode_changed(self, mode: str):
        self.sim_manager.set_pid_mode(mode)

    def _on_env_changed(self, env_name: str):
        preset_map = {
            "Arena (Std)": "testing_arena",
            "Robotics Lab": "robotics_lab",
            "Carpet": "carpet_surface",
            "Asphalt": "asphalt_outdoor",
        }
        target_preset = preset_map.get(env_name, "testing_arena")
        self.sim_manager.set_environment_preset(target_preset)

    def _on_speed_changed(self, val: int):
        speed_factor = val / 10.0
        self.lbl_speed_val.setText(f"{speed_factor:.1f}x")
        self.sim_manager.set_time_scale(speed_factor)

    def _on_state_changed(self, state: str):
        if state == "RUNNING":
            self.btn_start.setEnabled(False)
            self.btn_pause.setEnabled(True)
        elif state == "PAUSED":
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)
        elif state in ["STOPPED", "ESTOP"]:
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)
