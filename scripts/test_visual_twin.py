"""
Test script to compile and render the authentic visual mesh twin in MuJoCo.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mujoco
import numpy as np

def test_mesh_compilation():
    xml_path = "simulation/model/sesame.xml"
    print(f"Loading {xml_path}...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    print(f"Successfully compiled MuJoCo model! Meshes: {model.nmesh}, Geoms: {model.ngeom}")
    
    from robot.parameters import STAND_POSE_RAD
    data.qpos[2] = 0.06
    data.qpos[7:15] = STAND_POSE_RAD
    mujoco.mj_forward(model, data)
    
    renderer = mujoco.Renderer(model, width=640, height=480)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.35
    cam.azimuth = 135.0
    cam.elevation = -25.0
    cam.lookat = np.array([0.0, 0.0, 0.05])
    
    opt = mujoco.MjvOption()
    renderer.update_scene(data, camera=cam, scene_option=opt)
    rgb = renderer.render()
    print(f"Offscreen rendering succeeded! Frame shape: {rgb.shape}")
    
    import matplotlib.pyplot as plt
    out_img = "results/visual_twin_preview.png"
    os.makedirs("results", exist_ok=True)
    plt.imsave(out_img, rgb)
    print(f"Saved visual twin preview to {out_img}")

if __name__ == "__main__":
    test_mesh_compilation()
