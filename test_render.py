import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from wsl2_gpu.test_render import (
    test_trajectory_smoother,
    test_subtitle_compositor,
    test_broll_pillarbox,
    test_multi_layout_framing_engine
)

if __name__ == "__main__":
    print("══════════════════════════════════════════════════")
    print("   Project Aether V2 Comprehensive Unit Tests    ")
    print("   Multi-Layout Semantic Framing Engine           ")
    print("══════════════════════════════════════════════════")
    test_trajectory_smoother()
    test_subtitle_compositor()
    test_broll_pillarbox()
    test_multi_layout_framing_engine()
    print("\n🎉 ALL PROJECT AETHER V2 TESTS PASSED SUCCESSFULLY! 🎉\n")
