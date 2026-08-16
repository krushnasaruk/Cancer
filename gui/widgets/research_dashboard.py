"""
Research Dashboard Widget for Empirical Benchmarking and Data Logging.

Orchestrates multi-episode quantitative experiments, computes true empirical metrics
(Success Rate, Mean/Median Error, RMSE, Return), and auto-records timestamped logs to
`results/gui_experiments/`.
"""

import os
import sys
import time
import json
import csv
from datetime import datetime
from typing import Optional, Dict, List
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFrame,
    QGroupBox,
    QScrollArea,
)

from simulation.environment.sesame_env import SesameEnv
from gui.core.controller_manager import ControllerType, ControllerManager
from gui.core.environment_presets import ENVIRONMENT_PRESETS


class ExperimentWorker(QThread):
    """Background worker executing the batch experiment without freezing GUI."""

    sig_progress = pyqtSignal(int, int)  # current_ep, total_eps
    sig_finished = pyqtSignal(dict)      # summary_metrics

    def __init__(
        self,
        controller_name: str,
        env_name: str,
        domain_randomization: bool,
        use_actuator_model: bool,
        num_episodes: int = 10,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.controller_name = controller_name
        self.env_name = env_name
        self.domain_randomization = domain_randomization
        self.use_actuator_model = use_actuator_model
        self.num_episodes = num_episodes
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        env = SesameEnv(use_actuator_model=self.use_actuator_model)
        ctrl_manager = ControllerManager()
        ctrl_manager.set_controller(self.controller_name)
        
        # Apply environment preset
        if self.env_name in ENVIRONMENT_PRESETS:
            ENVIRONMENT_PRESETS[self.env_name].apply(env.model)
            
        episode_records = []
        returns = []
        final_dists = []
        successes = []
        step_counts = []
        
        t_start = time.time()
        
        for ep in range(self.num_episodes):
            if self._is_cancelled:
                break
                
            obs, _ = env.reset(seed=5000 + ep)
            ep_return = 0.0
            ep_step = 0
            
            while True:
                q_curr = env.robot.get_joint_positions() if hasattr(env, "robot") else env.data.qpos[7:15]
                dq_curr = env.robot.get_joint_velocities() if hasattr(env, "robot") else env.data.qvel[6:14]
                
                target_q, raw_act = ctrl_manager.compute_action(
                    obs, q_curr, dq_curr, ep_step * 0.02, dt=0.02
                )
                
                obs, rew, terminated, truncated, info = env.step(raw_act)
                ep_return += rew
                ep_step += 1
                
                if terminated or truncated or ep_step >= 200:
                    dist = info.get("dist_to_target", 0.0)
                    is_succ = bool(dist < 0.025)
                    
                    returns.append(ep_return)
                    final_dists.append(dist)
                    successes.append(is_succ)
                    step_counts.append(ep_step)
                    
                    episode_records.append({
                        "episode": ep + 1,
                        "return": float(ep_return),
                        "final_dist_m": float(dist),
                        "final_dist_mm": float(dist * 1000.0),
                        "steps": ep_step,
                        "success": is_succ,
                    })
                    break
                    
            self.sig_progress.emit(ep + 1, self.num_episodes)
            
        elapsed = time.time() - t_start
        total_steps = sum(step_counts)
        fps = total_steps / max(0.001, elapsed)
        
        dists_mm = np.array(final_dists) * 1000.0 if final_dists else np.zeros(1)
        
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "controller": self.controller_name,
                "environment": self.env_name,
                "domain_randomization": self.domain_randomization,
                "use_actuator_model": self.use_actuator_model,
                "num_episodes": self.num_episodes,
            },
            "metrics": {
                "success_rate_pct": float(np.mean(successes) * 100.0) if successes else 0.0,
                "mean_final_dist_mm": float(np.mean(dists_mm)),
                "mean_final_error_mm": float(np.mean(dists_mm)),
                "median_final_dist_mm": float(np.median(dists_mm)),
                "rmse_final_dist_mm": float(np.sqrt(np.mean(dists_mm ** 2))),
                "rmse_error_mm": float(np.sqrt(np.mean(dists_mm ** 2))),
                "max_final_dist_mm": float(np.max(dists_mm)),
                "mean_episode_return": float(np.mean(returns)) if returns else 0.0,
                "std_episode_return": float(np.std(returns)) if returns else 0.0,
                "mean_episode_length": float(np.mean(step_counts)) if step_counts else 0.0,
                "total_sim_steps": total_steps,
                "elapsed_time_s": float(elapsed),
                "sim_throughput_steps_per_s": float(fps),
            },
            "episodes": episode_records,
        }
        
        self._save_results(summary)
        self.sig_finished.emit(summary)

    def _save_results(self, summary: dict):
        save_dir = "results/gui_experiments"
        os.makedirs(save_dir, exist_ok=True)
        
        t_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        ctrl_tag = summary["configuration"]["controller"].lower().replace(" ", "_").replace("+", "_")
        
        # 1. Save JSON
        json_path = os.path.join(save_dir, f"exp_{t_tag}_{ctrl_tag}.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        # 2. Save CSV
        csv_path = os.path.join(save_dir, f"exp_{t_tag}_{ctrl_tag}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Return", "Final_Dist_mm", "Steps", "Success"])
            for ep in summary["episodes"]:
                writer.writerow([ep["episode"], f"{ep['return']:.2f}", f"{ep['final_dist_mm']:.2f}", ep["steps"], ep["success"]])


class ResearchDashboard(QWidget):
    """Research Dashboard UI for experimental configuration and live analysis."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.worker: Optional[ExperimentWorker] = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)
        
        # Top Container Frame (Compact Configuration Strip)
        cfg_frame = QFrame()
        cfg_frame.setObjectName("CardFrame")
        cfg_frame.setStyleSheet("""
            QFrame#CardFrame {
                background-color: #121622;
                border: 1px solid #202738;
                border-radius: 8px;
                padding: 6px;
            }
        """)
        cfg_layout = QVBoxLayout(cfg_frame)
        cfg_layout.setContentsMargins(6, 4, 6, 4)
        cfg_layout.setSpacing(4)
        
        # Row 1: Dropdowns & Options
        r1 = QHBoxLayout()
        r1.setSpacing(8)
        
        lbl_c = QLabel("Controller:")
        lbl_c.setStyleSheet("color: #94a3b8; font-weight: 700; font-size: 11px;")
        self.combo_ctrl = QComboBox()
        self.combo_ctrl.addItem(ControllerType.PID)
        self.combo_ctrl.addItem(ControllerType.PPO)
        self.combo_ctrl.addItem(ControllerType.SAC)
        self.combo_ctrl.addItem(ControllerType.PPO_DR)
        self.combo_ctrl.setMinimumWidth(90)
        
        lbl_e = QLabel("Env:")
        lbl_e.setStyleSheet("color: #94a3b8; font-weight: 700; font-size: 11px;")
        self.combo_env = QComboBox()
        for key, preset in ENVIRONMENT_PRESETS.items():
            self.combo_env.addItem(preset.display_name[:16], key)
        self.combo_env.setMinimumWidth(110)
        
        self.chk_dr = QCheckBox("Domain Rand")
        self.chk_dr.setStyleSheet("color: #cbd5e1; font-weight: 600; font-size: 11px;")
        
        self.chk_act = QCheckBox("MG90S Dynamics")
        self.chk_act.setChecked(True)
        self.chk_act.setStyleSheet("color: #cbd5e1; font-weight: 600; font-size: 11px;")
        
        lbl_n = QLabel("Eps:")
        lbl_n.setStyleSheet("color: #94a3b8; font-weight: 700; font-size: 11px;")
        self.spin_episodes = QSpinBox()
        self.spin_episodes.setRange(1, 100)
        self.spin_episodes.setValue(10)
        self.spin_episodes.setFixedWidth(55)
        self.spin_episodes.setStyleSheet("background-color: #1a202c; color: #ffffff; padding: 2px;")
        
        self.btn_run = QPushButton("🚀 Run Benchmark")
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                border: 1px solid #38bdf8;
                font-size: 11px;
                font-weight: 700;
                padding: 4px 12px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #0369a1; }
        """)
        self.btn_run.clicked.connect(self._on_run_clicked)
        
        r1.addWidget(lbl_c)
        r1.addWidget(self.combo_ctrl)
        r1.addWidget(lbl_e)
        r1.addWidget(self.combo_env)
        r1.addWidget(self.chk_dr)
        r1.addWidget(self.chk_act)
        r1.addWidget(lbl_n)
        r1.addWidget(self.spin_episodes)
        r1.addWidget(self.btn_run)
        r1.addStretch()
        cfg_layout.addLayout(r1)
        
        # Row 2: Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready for experiment")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a202c;
                border: 1px solid #283046;
                border-radius: 4px;
                font-size: 10px;
                color: #e2e8f0;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #00d2ff;
                border-radius: 3px;
            }
        """)
        cfg_layout.addWidget(self.progress_bar)
        main_layout.addWidget(cfg_frame)
        
        # 2. Empirical Results Table
        self.table_results = QTableWidget(8, 2)
        self.table_results.setHorizontalHeaderLabels(["Benchmark Metric", "Empirical Value"])
        self.table_results.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_results.verticalHeader().setVisible(False)
        self.table_results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_results.setStyleSheet("""
            QTableWidget {
                background-color: #10131d;
                border: 1px solid #202738;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #141824;
                color: #00d2ff;
                font-weight: 700;
                font-size: 11px;
                padding: 4px;
                border-bottom: 1px solid #202738;
            }
            QTableWidget::item {
                padding: 3px 6px;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        
        metric_labels = [
            "Task Success Rate",
            "Mean Final Distance (Error)",
            "Median Final Error",
            "Root-Mean-Square Error (RMSE)",
            "Maximum Error",
            "Mean Episode Return",
            "Mean Episode Length",
            "Simulation Throughput",
        ]
        
        for r, label in enumerate(metric_labels):
            self.table_results.setItem(r, 0, QTableWidgetItem(label))
            val_item = QTableWidgetItem("--")
            val_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_results.setItem(r, 1, val_item)
            
        main_layout.addWidget(self.table_results)

    def _on_run_clicked(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.btn_run.setText("🚀 Run Benchmark")
            self.progress_bar.setFormat("Experiment Cancelled")
            return
            
        ctrl = self.combo_ctrl.currentText()
        env_key = self.combo_env.currentData() or "standard_arena"
        dr = self.chk_dr.isChecked()
        act = self.chk_act.isChecked()
        num_eps = self.spin_episodes.value()
        
        self.btn_run.setText("⏹ Stop Benchmark")
        self.btn_run.setStyleSheet("background-color: #dc2626; color: #ffffff; border-radius: 5px; font-weight: 700;")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"Running Episode 1 / {num_eps}...")
        
        self.worker = ExperimentWorker(
            controller_name=ctrl,
            env_name=env_key,
            domain_randomization=dr,
            use_actuator_model=act,
            num_episodes=num_eps,
            parent=self,
        )
        self.worker.sig_progress.connect(self._on_worker_progress)
        self.worker.sig_finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_worker_progress(self, current: int, total: int):
        pct = int(current / total * 100)
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"Evaluating Episode {current} / {total} ({pct}%)")

    def _on_worker_finished(self, summary: dict):
        self.btn_run.setText("🚀 Run Benchmark")
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                border: 1px solid #38bdf8;
                font-size: 11px;
                font-weight: 700;
                padding: 4px 12px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #0369a1; }
        """)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("✓ Experiment Finished & Logged to results/gui_experiments/")
        
        m = summary["metrics"]
        
        # Populate Table
        results = [
            f"{m['success_rate_pct']:.1f}%",
            f"{m['mean_final_dist_mm']:.2f} mm",
            f"{m['median_final_dist_mm']:.2f} mm",
            f"{m['rmse_final_dist_mm']:.2f} mm",
            f"{m['max_final_dist_mm']:.2f} mm",
            f"{m['mean_episode_return']:.2f} ± {m['std_episode_return']:.2f}",
            f"{m['mean_episode_length']:.1f} steps",
            f"{m['sim_throughput_steps_per_s']:.0f} steps/s",
        ]
        
        for r, val_str in enumerate(results):
            item = QTableWidgetItem(val_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if r == 0:
                item.setForeground(Qt.GlobalColor.green if m['success_rate_pct'] > 50 else Qt.GlobalColor.yellow)
            self.table_results.setItem(r, 1, item)
