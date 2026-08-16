"""
Model & Configuration Verification Tests for Sesame Digital Twin.
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from robot.parameters import (
    JOINT_NAMES,
    JOINT_LIMITS_RAD,
    STAND_POSE_RAD,
    REST_POSE_RAD,
    BASE_LENGTH,
    BASE_WIDTH,
    BASE_HEIGHT,
)

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulation/model/sesame.xml"))


def test_xml_syntax_and_structure():
    """Verify that sesame.xml is valid XML and contains all essential tags."""
    assert os.path.exists(MODEL_PATH), f"File does not exist: {MODEL_PATH}"
    tree = ET.parse(MODEL_PATH)
    root = tree.getroot()
    assert root.tag == "mujoco", "Root element must be <mujoco>"
    
    # Check worldbody
    worldbody = root.find("worldbody")
    assert worldbody is not None, "Missing <worldbody>"
    
    # Check base_link body
    base_link = worldbody.find("./body[@name='base_link']")
    assert base_link is not None, "Missing <body name='base_link'>"
    
    # Check freejoint
    freejoint = base_link.find("./freejoint")
    assert freejoint is not None, "Missing <freejoint> on base_link"
    
    # Check 4 leg bodies
    for leg in ["fl_femur", "fr_femur", "rl_femur", "rr_femur"]:
        leg_body = base_link.find(f".//body[@name='{leg}']")
        assert leg_body is not None, f"Missing leg body: {leg}"


def test_joints_and_actuators_match():
    """Verify all 8 actuated joints have corresponding position actuators."""
    tree = ET.parse(MODEL_PATH)
    root = tree.getroot()
    
    # Collect all hinge joints
    joints = [j.get("name") for j in root.findall(".//joint[@type='hinge']")]
    assert len(joints) == 8, f"Expected 8 hinge joints, found {len(joints)}: {joints}"
    
    for name in JOINT_NAMES:
        assert name in joints, f"Expected joint {name} in XML model"
        
    # Collect actuators
    actuator_joints = [a.get("joint") for a in root.findall("./actuator/position")]
    assert len(actuator_joints) == 8, f"Expected 8 position actuators, found {len(actuator_joints)}"
    
    for name in JOINT_NAMES:
        assert name in actuator_joints, f"Missing actuator for joint {name}"


def test_joint_limits_match_parameters():
    """Verify that XML joint ranges match JOINT_LIMITS_RAD in parameters.py."""
    tree = ET.parse(MODEL_PATH)
    root = tree.getroot()
    
    for j_elem in root.findall(".//joint[@type='hinge']"):
        j_name = j_elem.get("name")
        range_str = j_elem.get("range")
        assert range_str is not None, f"Missing range on joint {j_name}"
        
        low, high = map(float, range_str.split())
        expected_low, expected_high = JOINT_LIMITS_RAD[j_name]
        
        assert np.isclose(low, expected_low, atol=1e-3), f"{j_name} min limit mismatch: {low} != {expected_low}"
        assert np.isclose(high, expected_high, atol=1e-3), f"{j_name} max limit mismatch: {high} != {expected_high}"


def test_mujoco_compilation():
    """If mujoco is installed, verify that the model compiles with zero warnings/errors."""
    try:
        import mujoco
    except ImportError:
        print("Skipping direct MuJoCo compilation test (mujoco module not yet loaded in Python).")
        return
        
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    assert model.nq == 15, f"Expected 15 generalized coords (7 freejoint + 8 hinges), got {model.nq}"
    assert model.nv == 14, f"Expected 14 degrees of freedom (6 freejoint + 8 hinges), got {model.nv}"
    assert model.nu == 8, f"Expected 8 actuators, got {model.nu}"
    print("MuJoCo direct compilation verified successfully!")


if __name__ == "__main__":
    test_xml_syntax_and_structure()
    test_joints_and_actuators_match()
    test_joint_limits_match_parameters()
    test_mujoco_compilation()
    print("ALL MODEL STRUCTURE & INTEGRITY TESTS PASSED!")
