# 🎬 Project Aether — Autonomous AI Shorts Maker

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B_CUDA_12.8-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Faster-Whisper](https://img.shields.io/badge/ASR-Faster--Whisper_INT8-green.svg)](https://github.com/SYSTRAN/faster-whisper)
[![YOLO11-Pose](https://img.shields.io/badge/Vision-YOLO11--Pose_TRT-ffd43b.svg)](https://docs.ultralytics.com/)
[![vLLM Brain](https://img.shields.io/badge/VLM-Qwen2.5--VL--7B_FP8-purple.svg)](https://vllm.ai/)
[![OSQP Solver](https://img.shields.io/badge/Camera-OSQP_Convex_Optimizer-orange.svg)](https://osqp.org/)
[![Tests](https://img.shields.io/badge/Tests-531%20Passed%20(100%25)-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An enterprise-grade, fully offline, edge-capable AI engine that transforms long-form horizontal video into viral, production-ready 9:16 vertical shorts with zero human intervention.**

</div>

---

## ⚡ Highlights & Key Innovations

- **Universal Multi-Layout Semantic Framing Engine**:
  - **Mode 1 — Smart Portrait Pan with Adaptive Face Headroom**: Dynamically senses face bounding box scale. Wide/medium shots receive a 100% full-bleed 9:16 vertical crop. Close-up shots widen the crop by 25% with slim ~10% Gaussian-blurred top/bottom bands, preserving natural shoulders and comfortable breathing room.
  - **Mode 2 — Full-Frame Topic & Screencast Close-Up**: Software windows, desktop recordings, monitors, maps, gameplay, and demonstrated items render in complete **9:16 vertical ratio** centered on active UI elements and cursor movements without letterbox padding.
  - **Mode 3 — Dual-Pane Split-Screen with Neon Glowing Divider**: Automatically pairs verified speakers with active demonstration topics using a styled 4px `#CCFF00` neon glowing divider. Gated by strict spatial separation ($\text{IoU} < 0.05$, Euclidean distance $\ge 0.28$) to eliminate duplicate identical presenter panes.
- **Anti-Hunting Kinematic Dead-Zone Stabilization**:
  - Virtual camera pan paths are optimized via **OSQP Quadratic Programming**:
    $$\min_{x} \sum_{t} (x_t - p_t)^2 + \lambda_v (x_t - x_{t-1})^2 + \lambda_a (x_{t+1} - 2x_t + x_{t-1})^2$$
  - Anchored $\pm 35\text{px}$ kinematic dead-zone filter suppresses natural presenter head bobbing and detection noise, producing silky, human-like camera motion without camera hunting or jitter.
- **Deterministic In-Video Sponsor & Fluff Exclusion Scanner**:
  - Comprehensive regex and multi-word token scanner (`detect_sponsor_segments`) detects paid promotions, brand integrations, and channel pitch intervals, applying temporal safety padding to guarantee candidate shorts are 100% sponsor-free.
- **Autonomous Viral Pacing & Self-Correction Loop**:
  - Evaluates clips across 11 engagement heuristics, enforcing a continuous single-crop duration cap $\le 7.0$s and dynamically alternating cuts between presenter portraits and topic close-ups.
- **GPU Glyph Atlas Kinetic Typography Compositor**:
  - Real-time word-level pop highlighting in neon yellow (`#FFFF00`), dual-pass stroke rendering, vertical safe-zone lock ($Y \in [0.60, 0.70]$), and multi-script font fallback chain (Latin, CJK, Devanagari, Arabic).
- **Serial 3-Phase Subprocess VRAM Isolation**:
  - Strictly guarantees peak GPU memory consumption remains under **16 GB VRAM** (tested on single RTX 4080 / 5080 Mobile GPUs) via serial subprocess spawning and post-phase CUDA cache sanitization.

---

## 🏗️ System Architecture

```
                                  PROJECT AETHER ARCHITECTURE
                                  
   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │                                   NATS JetStream Bus                                   │
   └───────────────▲───────────────────────────▲────────────────────────────▲───────────────┘
                   │ video.ingest              │ pipeline.status            │ video.render
   ┌───────────────┴───────────────────────────┴────────────────────────────┴───────────────┐
   │                            Aether Unified Orchestrator Container                       │
   │                                                                                        │
   │   ┌────────────────────────────────────────────────────────────────────────────────┐   │
   │   │  Phase 1: Subprocess ASR Transcription (~3.9 GB Peak VRAM)                     │   │
   │   │  • Faster-Whisper Base/Large-v3 (INT8) + VAD Filter + Language Autodetect      │   │
   │   │  • PyAnnote 3.1 Acoustic Clustering (Speaker Diarization)                      │   │
   │   └───────────────────────────────────────┬────────────────────────────────────────┘   │
   │                                           ▼ VRAM Sanitization (torch.cuda.empty_cache) │
   │   ┌────────────────────────────────────────────────────────────────────────────────┐   │
   │   │  Phase 2: Subprocess Vision & VLM Editorial (~8.5 GB Peak VRAM)                │   │
   │   │  • Kornia GPU Color-Histogram Scene-Change Detection (150+ shot cuts)          │   │
   │   │  • YOLO11s-Pose TRT 3-Tier Spatial Detection (Face, Human, Objects, Screens)   │   │
   │   │  • In-Video Sponsor Scanner (Excludes paid promotions & channel fluff)         │   │
   │   │  • Multimodal VLM Editorial (Qwen2.5-VL via vLLM sidecar): Top-4 Story Arcs   │   │
   │   └───────────────────────────────────────┬────────────────────────────────────────┘   │
   │                                           ▼ VRAM Sanitization (torch.cuda.empty_cache) │
   │   ┌────────────────────────────────────────────────────────────────────────────────┐   │
   │   │  Phase 3: Subprocess GPU Compositing & Render (~4.2 GB Peak VRAM)              │   │
   │   │  • OSQP 2D Convex Trajectory Solver + Anchored ±35px Kinematic Dead-Zone       │   │
   │   │  • EngagementEvaluator Autonomous Pacing Self-Correction Loop                  │   │
   │   │  • Universal 3-Mode GPU Compositor (Portrait Pan / Split-Screen / Topic)       │   │
   │   │  • Kinetic Subtitle Glyph Atlas (Word-by-word active pop highlight)           │   │
   │   │  • 3-Thread Lock-Free Ring Buffer (NVDEC Decode ➔ GPU Crop ➔ libx264 Encode)    │   │
   │   └────────────────────────────────────────────────────────────────────────────────┘   │
   └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 VRAM Lifecycle & Performance Profile

All heavy GPU tasks run in dedicated subprocesses with isolated CUDA contexts. Memory is completely reclaimed and verified via NVML after each phase:

| Phase | Core Component | Peak VRAM | Typical Duration (15m Video) | Memory Management Action |
|:---:|:---|:---:|:---:|:---|
| **Phase 1** | Faster-Whisper INT8 + Acoustic Diarization | **3.9 GB** | ~30 seconds | Subprocess termination + `cuda.empty_cache()` |
| **Phase 2** | Kornia SceneDet + YOLO11s-Pose + VLM Editorial | **8.5 GB** | ~130 seconds | Subprocess termination + `gc.collect()` |
| **Phase 3** | OSQP Solver + GPU Compositor + NVDEC/x264 | **4.2 GB** | ~150 seconds (4 shorts) | Lock-free 8-frame ring buffer |
| **Total** | **End-to-End Pipeline Execution** | **< 16 GB** | **~310–330 seconds** | **Zero Memory Leaks / Zero OOM** |

---

## 📐 Universal Framing Modes

| Mode | Target Content | Framing Mechanism |
|:---:|:---|:---|
| **Smart Portrait Pan** | Dialogue, Presenters, Solo Speakers, Podcasts | Full 9:16 vertical crop. When face scale is tight ($h \ge 0.26$), expands crop width by 25% with slim ~10% Gaussian-blurred top/bottom bands for comfortable headroom and shoulder space. |
| **Dual-Pane Split-Screen** | Demonstrations, Interviews, Product Reviews | Stacked 50/50 split (Speaker top, Topic bottom) separated by a styled 4px `#CCFF00` neon glowing divider. Gated by strict Euclidean distance $\ge 0.28$ to prevent duplicate panes. |
| **Full-Frame Topic Closeup** | Screencasts, Software UI, Monitors, Gameplay, B-Roll | Complete **9:16 vertical ratio full-frame crop** centered directly on active UI elements, mouse clicks, and focal objects with zero letterbox blur. |

---

## 🛠️ Prerequisites & Hardware Requirements

- **GPU**: NVIDIA GPU with $\ge 16$ GB VRAM (RTX 3080/3090, RTX 4080/4090, RTX 5080/5090 Mobile/Desktop).
- **Driver**: NVIDIA Driver $\ge 535$ with CUDA 12.x support.
- **Environment**: Ubuntu 22.04 LTS or Windows 11 with WSL2.
- **Container Tools**:
  - [Docker Engine](https://docs.docker.com/engine/install/) $\ge 24.0$
  - [Docker Compose](https://docs.docker.com/compose/) v2+
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Teraxyl14/AI-Based-Shorts-Maker.git
cd AI-Based-Shorts-Maker
```

### 2. Configure Environment
```bash
cp .env.example .env
```
*(Default settings are pre-tuned for local edge execution with zero external API dependencies.)*

### 3. Launch Services
```bash
docker compose up -d
```
Verify all containers are healthy:
```bash
docker compose ps
```

### 4. Process a Video
Drop any source video into the input folder:
```bash
cp /path/to/my_video.mp4 data/input_videos/
```

Trigger ingest via the container CLI:
```bash
docker exec aether-orchestrator python3 wsl2_gpu/trigger_ingest.py /data/input_videos/my_video.mp4
```

### 5. Retrieve Rendered Shorts
Once rendering finishes (typically ~5 minutes for a 15-minute 1080p video), access the production shorts:
```bash
ls -l data/output_shorts/<video_hash>/
# Contains: short_1.mp4, short_2.mp4, short_3.mp4, short_4.mp4, thumbnails, and metadata.json
```

---

## ⚙️ Configuration Reference (`.env`)

| Variable | Default Value | Description |
|:---|:---:|:---|
| `NATS_URL` | `nats://nats:4222` | Internal NATS message broker endpoint |
| `VLLM_BRAIN_URL` | `http://vllm-brain:8000` | vLLM OpenAI-compatible multimodal endpoint |
| `SPATIAL_TRACKER_MODEL` | `yolo26s-pose` | YOLO spatial pose tracking model variant |
| `TARGET_LANGUAGE` | `auto` | ASR transcription language (`auto` for native autodetection) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True,max_split_size_mb:512` | PyTorch memory fragmentation prevention |
| `VLLM_MOE_KERNEL_BACKEND` | `triton` | Accelerated kernel backend for vision-language sidecar |

---

## 🧪 Certified Test Suite

The Project Aether test suite includes **531 passing tests** with 100% pass rate:

```bash
docker exec aether-orchestrator pytest tests/ -q
# 531 passed, 1 skipped, 1 warning in 108.02s (100% pass rate)
```

### Test Suite Structure
- **Tier 1: Functional Feature Coverage** (`tests/test_tier1_features.py`): 51 tests covering ASR timestamps, acoustic diarization, Kornia scene cuts, YOLO pose tracking, OSQP trajectory bounds, canvas compositor, and kinetic subtitles.
- **Tier 2: Boundary & Corner Stress** (`tests/test_tier2_boundaries.py`): 51 tests testing 0.01s micro-words, high-motion sports cuts, extreme coordinate clamps ($x \in [303.75, 1616.25]\text{px}$), CJK/Arabic typography, and subprocess timeouts.
- **Tier 3: Pairwise Module Integration** (`tests/test_tier3_pairwise.py`): 12 tests verifying cross-spoke interactions between ASR, scene cuts, OSQP teleports, and VRAM phase handovers.
- **Tier 4: Physical Workloads** (`tests/test_tier4_workloads.py`): Real-world end-to-end ingest workloads on multi-minute 1080p videos validating NVML VRAM limits.
- **Milestone 6 Multi-Cut & Framing Certification** (`tests/test_milestone6_*.py`, `test_auditor_m6_2_*.py`): 140 tests validating visual engagement self-correction, #CCFF00 neon glowing divider styling, and topic spatial separation.
- **In-Video Sponsor Scanner Tests** (`tests/test_sponsor_detection.py`): Validates deterministic detection and temporal removal of sponsor intervals.

---

## 📁 Repository Structure

```
.
├── docker-compose.yml              # Production stack: NATS + vLLM Brain + Unified Orchestrator
├── nats-server.conf                # NATS JetStream stream & consumer configuration
├── .dockerignore                   # Build context sanitization
├── .env.example                    # Environment variable template
├── .gitignore                      # Secure exclusion of media, weights, and caches
├── LICENSE                         # MIT Open-Source License
├── README.md                       # Master Documentation
│
├── shared/                         # Cross-Phase Data Contracts & Schemas
│   ├── schemas.py                  # Pydantic models (WordTimestamp, RenderJob, etc.)
│   └── nats_subjects.py            # NATS topic constants
│
├── wsl2_gpu/                       # Core GPU Pipeline (PyTorch, CUDA, TRT)
│   ├── orchestrator.py             # Main event listener & pipeline coordinator
│   ├── vram_orchestrator.py        # Serial 3-phase subprocess VRAM lifecycle manager
│   ├── transcription_spoke.py      # Faster-Whisper ASR + acoustic diarization
│   ├── vision_spoke.py             # Kornia scene detection + YOLO11s-Pose tracking
│   ├── editorial_engine.py         # Sponsor scanner + multimodal VLM short selection
│   ├── engagement_evaluator.py     # 11-rule viral pacing self-correction loop
│   ├── trajectory_solver.py        # OSQP 2D trajectory solver + ±35px dead-zone filter
│   ├── trajectory_smoother.py      # RTS Kalman & Savitzky-Golay smoothing fallbacks
│   ├── gpu_compositor.py           # 3-mode GPU tensor compositor (portrait, split, topic)
│   ├── render_spoke.py             # 3-thread lock-free video renderer & subtitle burner
│   ├── subtitle_compositor.py      # GPU glyph atlas kinetic typography engine
│   ├── video_io.py                 # High-throughput frame streaming & decoding
│   ├── fusion_consumer.py          # Audio-visual timeline fusion
│   ├── debate_graph.py             # Multi-agent editorial reasoning graph
│   ├── trigger_ingest.py           # Manual CLI trigger utility
│   ├── test_render.py              # Render unit test runner
│   ├── Dockerfile.unified          # Production multi-stage CUDA container
│   └── requirements-unified.txt    # Frozen Python dependencies
│
├── tests/                          # 531-Item Pytest Verification Suite
│   ├── conftest.py                 # Test fixtures & mocks
│   ├── test_tier1_features.py      # F1–F10 functional contract tests
│   ├── test_tier2_boundaries.py    # Numerical boundary & stress tests
│   ├── test_tier3_pairwise.py      # Cross-phase integration tests
│   ├── test_tier4_workloads.py     # Physical video workload tests
│   ├── test_sponsor_detection.py   # In-video sponsor scanner tests
│   └── test_milestone*.py          # Multi-cut framing & pacing tests
│
└── data/                           # Local Edge Storage (Gitignored)
    ├── input_videos/               # Input long-form video drop folder
    └── output_shorts/              # Generated 9:16 MP4 shorts & metadata manifests
```

---

## 🔒 Security & Privacy Invariants

- **100% Local & Edge-Native**: Operates fully offline within your local environment. Zero external API calls, zero telemetry, zero cloud data leakage.
- **Zero Hardcoded Secrets**: All model weights and authorization credentials use environment variables (`HF_TOKEN`) and secret isolation.
- **Sanitized Repository**: Strict `.gitignore` policy excludes all input video media, output shorts, local temporary audio files, and model weights from version control.

---

## 📜 License

Project Aether is released under the [MIT License](LICENSE).
