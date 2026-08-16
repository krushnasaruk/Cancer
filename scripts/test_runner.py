import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

with open("test_output.txt", "w") as out:
    def log(msg):
        print(msg)
        out.write(msg + "\n")
        out.flush()

    try:
        log("Importing test_gui...")
        import tests.test_gui as tg
        log("Running test_robot_interface...")
        tg.test_robot_interface()
        log("Running test_controller_manager...")
        tg.test_controller_manager()
        log("Running test_environment_presets...")
        tg.test_environment_presets()
        log("Running test_simulation_manager_lifecycle...")
        tg.test_simulation_manager_lifecycle()
        log("Running test_experiment_worker...")
        tg.test_experiment_worker()
        log("ALL TESTS COMPLETED SUCCESSFULLY!")
    except Exception:
        err = traceback.format_exc()
        log("ERROR OCCURRED:")
        log(err)
        sys.exit(1)
