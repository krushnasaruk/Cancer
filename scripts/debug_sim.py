import sys, time, os
from PyQt6.QtWidgets import QApplication
sys.path.insert(0, ".")
from gui.core.simulation_manager import SimulationManager, SimState

app = QApplication.instance() or QApplication(sys.argv)
print("1. Creating SimulationManager...", flush=True)
sim = SimulationManager()
print("2. Starting worker thread...", flush=True)
sim.start()
time.sleep(0.3)
print("3. Starting sim...", flush=True)
sim.start_sim()
time.sleep(0.3)
print("4. Pausing sim...", flush=True)
sim.pause_sim()
time.sleep(0.3)
print("5. Stopping worker...", flush=True)
sim.stop_worker()
print("6. Done!", flush=True)
