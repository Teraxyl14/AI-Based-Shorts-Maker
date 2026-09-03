import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from wsl2_gpu.test_render import (
    test_trajectory_smoother,
    test_subtitle_compositor,
    test_vram_orchestrator,
    test_broll_pillarbox
)

if __name__ == "__main__":
    print("══════════════════════════════════════════════════")
    print("   Project Aether V2 Comprehensive Unit Tests    ")
    print("══════════════════════════════════════════════════")
    test_trajectory_smoother()
    test_subtitle_compositor()
    test_vram_orchestrator()
    test_broll_pillarbox()
    print("\n🎉 ALL PROJECT AETHER V2 TESTS PASSED SUCCESSFULLY! 🎉\n")
