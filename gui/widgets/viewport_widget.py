"""
Interactive 3D Viewport Widget for MuJoCo Rendering.

Renders offscreen RGB frames to high-DPI QImage on the main UI thread with full mouse camera navigation
(Orbit, Pan, Zoom, Reset), Quick Camera Presets, Interactive Target Gizmo, 4-Foot Contact LEDs, and active telemetry HUD.
"""

from typing import Optional
import time
import numpy as np
import mujoco

from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QSize, QTimer, QMutexLocker
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFrame,
)
from PyQt6.QtGui import (
    QPainter,
    QImage,
    QPixmap,
    QColor,
    QFont,
    QPen,
    QBrush,
    QMouseEvent,
    QWheelEvent,
)


class ViewportWidget(QWidget):
    """Custom widget rendering MuJoCo offscreen buffers with interactive mouse controls & camera HUD."""

    def __init__(self, sim_manager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.sim_manager = sim_manager
        self.setMinimumSize(450, 320)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        
        self.current_image: Optional[QImage] = None
        self.last_mouse_pos = QPoint()
        self.is_dragging_left = False
        self.is_dragging_right = False
        self.renderer: Optional[mujoco.Renderer] = None
        self.frame_count = 0
        self.last_fps_time = time.perf_counter()
        self.follow_robot = False
        
        # Live contact states
        self.foot_contacts = {"FL": True, "FR": True, "RL": True, "RR": True}
        self.dist_to_target_mm = 0.0
        
        # 30 FPS Render Timer on Main GUI Thread
        self.render_timer = QTimer(self)
        self.render_timer.setInterval(33)  # ~30 FPS
        self.render_timer.timeout.connect(self.render_frame)
        self.render_timer.start()
        
        # Connect telemetry for contacts HUD
        self.sim_manager.sig_telemetry_updated.connect(self._on_telemetry)
        
        # Layout for overlay buttons
        self._init_overlay_toolbar()

    def _init_overlay_toolbar(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top row: Left spacer, Right camera preset bar
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        
        # Glassmorphism Camera Tool Strip
        cam_frame = QFrame()
        cam_frame.setObjectName("CamToolStrip")
        cam_frame.setStyleSheet("""
            QFrame#CamToolStrip {
                background-color: rgba(12, 16, 24, 0.88);
                border: 1px solid rgba(32, 43, 62, 0.9);
                border-radius: 8px;
                padding: 2px;
            }
            QPushButton.cam_btn {
                background-color: transparent;
                color: #94a3b8;
                border: none;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 700;
                border-radius: 4px;
            }
            QPushButton.cam_btn:hover {
                background-color: rgba(0, 240, 255, 0.2);
                color: #00f0ff;
            }
            QPushButton.cam_btn:checked {
                background-color: #00f0ff;
                color: #080a0f;
            }
        """)
        cam_layout = QHBoxLayout(cam_frame)
        cam_layout.setContentsMargins(4, 2, 4, 2)
        cam_layout.setSpacing(3)
        
        btn_iso = QPushButton("🌐 Isometric")
        btn_iso.setProperty("class", "cam_btn")
        btn_iso.clicked.connect(self.set_isometric_view)
        
        btn_front = QPushButton("👁️ Front")
        btn_front.setProperty("class", "cam_btn")
        btn_front.clicked.connect(self.set_front_view)
        
        btn_side = QPushButton("📐 Side")
        btn_side.setProperty("class", "cam_btn")
        btn_side.clicked.connect(self.set_side_view)
        
        btn_top = QPushButton("🔝 Top")
        btn_top.setProperty("class", "cam_btn")
        btn_top.clicked.connect(self.set_top_view)
        
        self.btn_follow = QPushButton("🎯 Follow CoM")
        self.btn_follow.setCheckable(True)
        self.btn_follow.setProperty("class", "cam_btn")
        self.btn_follow.toggled.connect(self._toggle_follow)
        
        cam_layout.addWidget(btn_iso)
        cam_layout.addWidget(btn_front)
        cam_layout.addWidget(btn_side)
        cam_layout.addWidget(btn_top)
        cam_layout.addWidget(self.btn_follow)
        
        top_bar.addWidget(cam_frame)
        main_layout.addLayout(top_bar)
        
        main_layout.addStretch()
        
        # Bottom row: Interactive Target Nudge Toolbar
        bottom_bar = QHBoxLayout()
        
        nudge_frame = QFrame()
        nudge_frame.setObjectName("NudgeFrame")
        nudge_frame.setStyleSheet("""
            QFrame#NudgeFrame {
                background-color: rgba(12, 16, 24, 0.88);
                border: 1px solid rgba(32, 43, 62, 0.9);
                border-radius: 8px;
                padding: 2px 6px;
            }
            QPushButton.nudge_btn {
                background-color: #141a26;
                color: #e2e8f0;
                border: 1px solid #222b3d;
                padding: 3px 7px;
                font-size: 10px;
                font-weight: 700;
                font-family: monospace;
                border-radius: 4px;
            }
            QPushButton.nudge_btn:hover {
                background-color: #00f0ff;
                color: #080a0f;
            }
        """)
        nudge_layout = QHBoxLayout(nudge_frame)
        nudge_layout.setContentsMargins(4, 2, 4, 2)
        nudge_layout.setSpacing(4)
        
        lbl_target_tag = QLabel("🎯 Target:")
        lbl_target_tag.setStyleSheet("color: #00f0ff; font-weight: 700; font-size: 10px;")
        nudge_layout.addWidget(lbl_target_tag)
        
        btn_xp = QPushButton("+X")
        btn_xp.setProperty("class", "nudge_btn")
        btn_xp.clicked.connect(lambda: self._nudge_target(0.015, 0.0, 0.0))
        
        btn_xm = QPushButton("-X")
        btn_xm.setProperty("class", "nudge_btn")
        btn_xm.clicked.connect(lambda: self._nudge_target(-0.015, 0.0, 0.0))
        
        btn_yp = QPushButton("+Y")
        btn_yp.setProperty("class", "nudge_btn")
        btn_yp.clicked.connect(lambda: self._nudge_target(0.0, 0.015, 0.0))
        
        btn_ym = QPushButton("-Y")
        btn_ym.setProperty("class", "nudge_btn")
        btn_ym.clicked.connect(lambda: self._nudge_target(0.0, -0.015, 0.0))
        
        btn_zp = QPushButton("+Z")
        btn_zp.setProperty("class", "nudge_btn")
        btn_zp.clicked.connect(lambda: self._nudge_target(0.0, 0.0, 0.010))
        
        btn_zm = QPushButton("-Z")
        btn_zm.setProperty("class", "nudge_btn")
        btn_zm.clicked.connect(lambda: self._nudge_target(0.0, 0.0, -0.010))
        
        nudge_layout.addWidget(btn_xp)
        nudge_layout.addWidget(btn_xm)
        nudge_layout.addWidget(btn_yp)
        nudge_layout.addWidget(btn_ym)
        nudge_layout.addWidget(btn_zp)
        nudge_layout.addWidget(btn_zm)
        
        bottom_bar.addWidget(nudge_frame)
        bottom_bar.addStretch()
        main_layout.addLayout(bottom_bar)

    def _nudge_target(self, dx: float, dy: float, dz: float):
        pos = self.sim_manager.target_pos
        pos[0] = max(0.02, min(0.20, pos[0] + dx))
        pos[1] = max(-0.10, min(0.10, pos[1] + dy))
        pos[2] = max(0.00, min(0.12, pos[2] + dz))
        self.sim_manager._update_mocap_target()

    def set_isometric_view(self):
        cam = self.sim_manager.camera
        cam.distance = 0.38
        cam.azimuth = 135.0
        cam.elevation = -24.0
        cam.lookat = np.array([0.0, 0.0, 0.05])

    def set_front_view(self):
        cam = self.sim_manager.camera
        cam.distance = 0.36
        cam.azimuth = 90.0
        cam.elevation = -5.0
        cam.lookat = np.array([0.0, 0.0, 0.05])

    def set_side_view(self):
        cam = self.sim_manager.camera
        cam.distance = 0.36
        cam.azimuth = 0.0
        cam.elevation = -5.0
        cam.lookat = np.array([0.0, 0.0, 0.05])

    def set_top_view(self):
        cam = self.sim_manager.camera
        cam.distance = 0.42
        cam.azimuth = 90.0
        cam.elevation = -89.0
        cam.lookat = np.array([0.0, 0.0, 0.05])

    def _toggle_follow(self, checked: bool):
        self.follow_robot = checked

    def _on_telemetry(self, data: dict):
        self.dist_to_target_mm = data.get("dist_to_target", 0.0) * 1000.0
        contacts = data.get("contacts", {})
        if contacts:
            self.foot_contacts = contacts
        else:
            self.foot_contacts = {"FL": True, "FR": True, "RL": True, "RR": True}

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = max(100, event.size().width())
        h = max(100, event.size().height())
        if self.renderer is not None:
            del self.renderer
            self.renderer = None

    def render_frame(self):
        try:
            w = max(100, self.width())
            h = max(100, self.height())
            
            with QMutexLocker(self.sim_manager.mutex):
                if self.renderer is None:
                    self.renderer = mujoco.Renderer(self.sim_manager.model, width=w, height=h)
                    
                if self.follow_robot:
                    base_pos = self.sim_manager.robot.get_base_position()
                    self.sim_manager.camera.lookat = base_pos.copy()
                    
                self.renderer.update_scene(
                    self.sim_manager.data,
                    camera=self.sim_manager.camera,
                    scene_option=self.sim_manager.opt,
                )
                rgb = self.renderer.render()
                
            ch_h, ch_w, c = rgb.shape
            bytes_per_line = c * ch_w
            self.current_image = QImage(rgb.data, ch_w, ch_h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            
            # Real FPS counter
            self.frame_count += 1
            now = time.perf_counter()
            if now - self.last_fps_time >= 0.5:
                self.sim_manager.current_fps = self.frame_count / (now - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = now
                
            self.update()
        except Exception:
            pass

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.current_image is not None and not self.current_image.isNull():
            scaled_img = self.current_image.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled_img.width()) // 2
            y = (self.height() - scaled_img.height()) // 2
            painter.drawImage(x, y, scaled_img)
        else:
            painter.fillRect(self.rect(), QColor(8, 10, 15))
            painter.setPen(QColor(100, 116, 139))
            painter.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Initializing Sesame Visual Twin...")
            
        # Draw HUD Overlays
        self._draw_hud(painter)
        painter.end()

    def _draw_hud(self, painter: QPainter):
        # 1. Controller & State Glass Badge (Top Left)
        ctrl_name = self.sim_manager.controller_manager.active_type
        state = self.sim_manager.state
        
        painter.setPen(QPen(QColor(32, 43, 62, 180), 1))
        painter.setBrush(QColor(12, 16, 24, 220))
        painter.drawRoundedRect(14, 14, 210, 52, 8, 8)
        
        painter.setPen(QColor(0, 240, 255))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(24, 33, f"🤖 CTRL: {ctrl_name[:16]}")
        
        state_color = QColor(0, 255, 157) if state == "RUNNING" else (QColor(245, 158, 11) if state == "PAUSED" else QColor(239, 68, 68))
        painter.setPen(state_color)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(24, 52, f"● {state}  |  {self.sim_manager.current_fps:.0f} FPS")
        
        # 2. 4-Foot Ground Contact LEDs (Bottom Right HUD)
        box_w, box_h = 180, 44
        bx = self.width() - box_w - 14
        by = self.height() - box_h - 14
        
        painter.setPen(QPen(QColor(32, 43, 62, 180), 1))
        painter.setBrush(QColor(12, 16, 24, 220))
        painter.drawRoundedRect(bx, by, box_w, box_h, 8, 8)
        
        painter.setPen(QColor(148, 163, 184))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(bx + 10, by + 15, "GROUND CONTACTS")
        
        legs = [("FL", 0), ("FR", 1), ("RL", 2), ("RR", 3)]
        for leg_name, i in legs:
            lx = bx + 10 + i * 40
            ly = by + 21
            is_contact = self.foot_contacts.get(leg_name, True)
            
            led_color = QColor(0, 255, 157) if is_contact else QColor(51, 65, 85)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(led_color)
            painter.drawRoundedRect(lx, ly, 34, 16, 4, 4)
            
            painter.setPen(QColor(8, 10, 15) if is_contact else QColor(148, 163, 184))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(lx + 8, ly + 12, leg_name)

    def mousePressEvent(self, event: QMouseEvent):
        self.last_mouse_pos = event.pos()
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging_left = True
        elif event.button() == Qt.MouseButton.RightButton:
            self.is_dragging_right = True

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging_left = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.is_dragging_right = False

    def mouseMoveEvent(self, event: QMouseEvent):
        dx = event.pos().x() - self.last_mouse_pos.x()
        dy = event.pos().y() - self.last_mouse_pos.y()
        self.last_mouse_pos = event.pos()
        
        cam = self.sim_manager.camera
        if self.is_dragging_left:
            cam.azimuth += dx * 0.4
            cam.elevation = max(-89.0, min(89.0, cam.elevation + dy * 0.4))
        elif self.is_dragging_right:
            cam.lookat[0] -= dx * 0.0006
            cam.lookat[1] += dy * 0.0006

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        cam = self.sim_manager.camera
        cam.distance = max(0.12, min(3.0, cam.distance * (1.0 - delta * 0.001)))

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self.set_isometric_view()
