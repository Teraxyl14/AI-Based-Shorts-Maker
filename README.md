# 🎬 Project Aether — AI-Based Short-Form Video Maker

An AI-powered, fully offline pipeline that automatically transforms long-form videos into engaging, platform-ready short-form vertical clips (9:16). Built for edge deployment on a single NVIDIA GPU machine.

## ✨ What It Does

Drop a long-form video into the input folder → Aether automatically:

1. **Transcribes** speech with word-level timestamps (Whisper + forced alignment)
2. **Analyzes** visual content — faces, objects, screens, gestures, scene cuts (YOLOE + Kornia)
3. **Fuses** audio and visual timelines into a unified semantic map
4. **Debates** editorially via a multi-agent LLM graph (Editor → Director → Publisher) to select the best 4+ story arcs
5. **Renders** each clip as a polished 1080×1920 vertical short with:
   - Script-grounded semantic framing (tracks speakers, screens, held objects)
   - Smooth 1D horizontal camera panning (full-height, zero digital zoom)
   - Kinetic outlined subtitles with active-word highlighting

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    docker compose up                        │
├──────────┬──────────────────────────┬───────────────────────┤
│   NATS   │      vLLM Brain          │    Orchestrator       │
│ JetStream│  (Qwen3-VL-8B FP8)      │  (Unified Container)  │
│          │                          │                       │
│  Message │  Editorial reasoning:    │  Phase 1: ASR         │
│  broker  │  • Editor proposals      │  Phase 2: Vision +    │
│  for     │  • Director critiques    │           Debate      │
│  events  │  • Arousal scoring       │  Phase 3: Render      │
│          │                          │                       │
└──────────┴──────────────────────────┴───────────────────────┘
```

### Pipeline Phases

| Phase | Component | GPU Usage | Description |
|-------|-----------|-----------|-------------|
| **1** | Transcription Spoke | ~3.9 GB | Whisper INT8 + forced word alignment |
| **2** | Vision Spoke | ~8.5 GB | YOLOE scene detection + 3-tier spatial tracking |
| **2** | Debate Graph | LLM (vLLM) | Multi-agent editorial: Editor → Director → Publisher |
| **3** | Render Spoke | ~5.5 GB | NVDEC → trajectory crop → subtitles → H.264/AV1 encode |

VRAM phases are **serialized** — only one phase loads models at a time, staying within 16 GB VRAM.

## 📋 Prerequisites

- **NVIDIA GPU** with ≥16 GB VRAM (tested on RTX 5080 Mobile)
- **Docker** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **Docker Compose** v2+
- ~20 GB disk space for model downloads on first run

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Teraxyl14/AI-Based-Shorts-Maker.git
cd AI-Based-Shorts-Maker

# 2. Copy environment template
cp .env.example .env

# 3. Start all services
docker compose up -d

# 4. Drop a video into the input folder
cp your_video.mp4 data/input_videos/

# 5. Trigger the pipeline (auto-detected by file watchdog, or manually):
docker exec aether-orchestrator python wsl2_gpu/trigger_ingest.py /data/input_videos/your_video.mp4

# 6. Find your shorts in:
ls data/output_shorts/
```

## 📁 Project Structure

```
.
├── docker-compose.yml          # Service definitions (NATS, vLLM, Orchestrator)
├── nats-server.conf            # NATS JetStream configuration
├── trigger_ingest.py           # Manual pipeline trigger utility
├── test_render.py              # Test runner
├── .env.example                # Environment variable template
│
├── shared/                     # Shared schemas & constants
│   ├── schemas.py              # Pydantic models (WordTimestamp, RenderJob, etc.)
│   └── nats_subjects.py        # NATS subject constants
│
├── wsl2_gpu/                   # Core GPU pipeline
│   ├── orchestrator.py         # Main pipeline orchestrator
│   ├── transcription_spoke.py  # Whisper ASR + forced alignment
│   ├── vision_spoke.py         # YOLOE + Kornia scene detection
│   ├── fusion_consumer.py      # Audio-visual timeline fusion + LLM arousal
│   ├── debate_graph.py         # LangGraph multi-agent editorial debate
│   ├── render_spoke.py         # GPU video renderer (crop, subtitle, encode)
│   ├── trajectory_smoother.py  # 1D Savitzky-Golay / RTS Kalman smoothing
│   ├── subtitle_compositor.py  # Kinetic outlined subtitle engine
│   ├── vram_orchestrator.py    # VRAM lifecycle phase manager
│   ├── ingest_watchdog.py      # File watchdog for auto-ingest
│   ├── test_render.py          # Comprehensive unit tests
│   ├── Dockerfile.unified      # Production container (used by compose)
│   ├── Dockerfile.*            # Alternative single-spoke Dockerfiles
│   └── requirements-*.txt      # Python dependencies
│
├── windows_npu/                # (Experimental) Windows NPU transcription agent
│   ├── transcription_agent.py
│   ├── ipc_bridge.py
│   └── requirements_*.txt
│
└── data/                       # Runtime data (gitignored)
    ├── input_videos/           # Drop source videos here
    ├── output_shorts/          # Generated shorts appear here
    ├── models/                 # Cached model weights
    └── nats_store/             # NATS JetStream persistence
```

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `NATS_URL` | `nats://nats:4222` | NATS message broker URL |
| `VLLM_BRAIN_URL` | `http://vllm-brain:8000` | vLLM inference endpoint |
| `SPATIAL_TRACKER_MODEL` | `yolo26s-pose` | YOLO model variant for spatial tracking |
| `TARGET_LANGUAGE` | `auto` | Transcription language (`auto` = detect) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True,max_split_size_mb:512` | CUDA memory allocator config |

## 🧪 Running Tests

```bash
# Inside the orchestrator container
docker exec aether-orchestrator python wsl2_gpu/test_render.py

# Or from the host (if dependencies are installed locally)
python test_render.py
```

## 📜 Output Specifications

Each generated short:
- **Resolution**: 1080×1920 (9:16 portrait)
- **Duration**: 30–48 seconds
- **Framing**: Full-height natural crop, horizontal pan only (zero digital zoom)
- **Subtitles**: Kinetic outlined typography with active-word highlighting
- **Codec**: H.264 (AV1 where supported)
- **Audio**: AAC 128kbps mono

## 📄 License

[MIT](LICENSE)
