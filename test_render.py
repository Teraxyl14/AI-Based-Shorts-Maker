import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from wsl2_gpu.test_render import (
    test_trajectory_smoother,
    test_poi_hierarchical_weights,
    test_script_grounded_semantic_framing,
    test_duration_gate_limits,
    test_micro_chunking,
    test_bicameral_uppercasing,
    test_subtitle_compositor,
    test_broll_pillarbox,
    test_fullscreen_916_crop,
    test_two_mode_universal_framing
)

if __name__ == "__main__":
    print("==================================================")
    print("   Project Aether V2 Comprehensive Unit Tests     ")
    print("   2-Mode Universal Production Standards          ")
    print("==================================================")
    test_trajectory_smoother()
    test_poi_hierarchical_weights()
    test_script_grounded_semantic_framing()
    test_duration_gate_limits()
    test_micro_chunking()
    test_bicameral_uppercasing()
    test_subtitle_compositor()
    test_broll_pillarbox()
    test_fullscreen_916_crop()
    test_two_mode_universal_framing()
    print("\n[SUCCESS] ALL PROJECT AETHER V2 TESTS PASSED SUCCESSFULLY!\n")
