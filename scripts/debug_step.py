import sys, traceback
sys.path.insert(0, ".")
from gui.core.simulation_manager import SimulationManager
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)
sim = SimulationManager()

try:
    print("Testing get_observation...")
    obs = sim._get_observation()
    print("Obs shape:", obs.shape)
    
    print("Testing compute_action...")
    q_curr = sim.robot.get_joint_positions()
    dq_curr = sim.robot.get_joint_velocities()
    tgt_q, raw_act = sim.controller_manager.compute_action(obs, q_curr, dq_curr, 0.0, 0.02)
    print("Target Q:", tgt_q)
    
    print("Testing set_joint_targets...")
    sim.robot.set_joint_targets(tgt_q, 0.02)
    
    print("Testing mj_step...")
    import mujoco
    for _ in range(10):
        mujoco.mj_step(sim.model, sim.data)
        
    print("Testing update mocap...")
    sim._update_mocap_target()
    
    print("Testing reward...")
    ee_pos = sim.data.site_xpos[sim.end_effector_site_id].copy()
    base_pos = sim.robot.get_base_position()
    w_tgt = base_pos + sim.target_pos
    rew, dist = sim._compute_reward(ee_pos, w_tgt, raw_act, q_curr)
    print("Reward:", rew, "Dist:", dist)
    
    print("Testing SSM...")
    from robot.dynamics import SesameDynamics
    com_pos = sim.data.subtree_com[1].copy()
    feet_pos = sim.robot.get_foot_positions()
    contacts_2d = {f: True for f in feet_pos}
    feet_2d = {f: feet_pos[f][:2] for f in feet_pos}
    ssm = SesameDynamics.compute_support_polygon_margin(contacts_2d, feet_2d, com_pos[:2])
    print("SSM:", ssm)
    print("ALL STEP OPERATIONS SUCCEEDED!")
except Exception:
    traceback.print_exc()
