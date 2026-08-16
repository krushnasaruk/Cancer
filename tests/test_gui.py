"""
Headless Automated Unit & Integration Tests for Sesame Control Center GUI.
"""

import os
import sys
import time
import json
from PyQt6.QtWidgets import QApplication

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gui.core.robot_interface import SimulationRobot, HardwareRobot
from gui.core.controller_manager import ControllerManager, ControllerType
from gui.core.environment_presets import ENVIRONMENT_PRESETS
from gui.core.simulation_manager import SimulationManager, SimState
from gui.widgets.research_dashboard import ExperimentWorker


def test_robot_interface():
    import mujoco
    MODEL_XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulation/model/sesame.xml"))
    
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)
    sim_robot = SimulationRobot(model, data, use_actuator_model=True)
    
    q = sim_robot.get_joint_positions()
    assert len(q) == 8, f"Expected 8 joint positions, got {len(q)}"
    
    dq = sim_robot.get_joint_velocities()
    assert len(dq) == 8, f"Expected 8 joint velocities, got {len(dq)}"
    
    pos = sim_robot.get_base_position()
    assert len(pos) == 3, f"Expected 3D base position, got {len(pos)}"
    
    euler = sim_robot.get_base_euler()
    assert len(euler) == 3, f"Expected 3D Euler angles, got {len(euler)}"
    
    feet = sim_robot.get_foot_positions()
    assert len(feet) == 4, f"Expected 4 feet positions, got {len(feet)}"
    
    hw_robot = HardwareRobot()
    assert hw_robot.is_hardware() is True
    print("RobotInterface tests passed!")


def test_controller_manager():
    cm = ControllerManager()
    assert cm.set_controller(ControllerType.PID) is True
    assert cm.set_controller(ControllerType.PPO) is True
    assert cm.set_controller(ControllerType.SAC) is True
    assert cm.set_controller(ControllerType.PPO_DR) is True
    print("ControllerManager tests passed!")


def test_environment_presets():
    import mujoco
    MODEL_XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulation/model/sesame.xml"))
    
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    for name, preset in ENVIRONMENT_PRESETS.items():
        preset.apply(model)
    print("EnvironmentPresets tests passed!")


def test_simulation_manager_lifecycle():
    app = QApplication.instance() or QApplication(sys.argv)
    sim = SimulationManager()
    sim.start()
    for _ in range(5):
        app.processEvents()
        time.sleep(0.1)
    
    sim.start_sim()
    for _ in range(5):
        app.processEvents()
        time.sleep(0.1)
    assert sim.state == SimState.RUNNING
    
    sim.pause_sim()
    for _ in range(2):
        app.processEvents()
        time.sleep(0.1)
    assert sim.state == SimState.PAUSED
    
    sim.step_single()
    app.processEvents()
    time.sleep(0.1)
    
    sim.reset_sim()
    app.processEvents()
    time.sleep(0.1)
    
    sim.emergency_stop()
    app.processEvents()
    time.sleep(0.1)
    assert sim.state == SimState.ESTOP
    
    sim.stop_worker()
    for _ in range(5):
        app.processEvents()
        time.sleep(0.05)
    print("SimulationManager lifecycle tests passed!")


def test_experiment_worker():
    app = QApplication.instance() or QApplication(sys.argv)
    worker = ExperimentWorker(
        controller_name=ControllerType.PID,
        env_name="testing_arena",
        domain_randomization=False,
        use_actuator_model=True,
        num_episodes=2,
    )
    
    results = {}
    worker.sig_finished.connect(lambda summary: results.setdefault("summary", summary))
    worker.start()
    worker.wait(10000)
    for _ in range(5):
        app.processEvents()
        time.sleep(0.02)
        
    assert "summary" in results, "ExperimentWorker did not complete within timeout"
    metrics = results["summary"]["metrics"]
    assert "mean_final_error_mm" in metrics
    assert "rmse_error_mm" in metrics
    print("ExperimentWorker tests passed!")


if __name__ == "__main__":
    test_robot_interface()
    test_controller_manager()
    test_environment_presets()
    test_simulation_manager_lifecycle()
    test_experiment_worker()
    print("\nALL GUI CORE INTEGRATION TESTS PASSED SUCCESSFULLY!")
