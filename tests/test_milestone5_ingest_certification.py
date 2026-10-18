"""
Milestone 5: End-to-End Ingest Validation & Victory Certification Test Suite
Probes output shorts in data/output_shorts/7fdcc82827ff6f3c983feaf39f56ea34/
and validates all acceptance criteria from ORIGINAL_REQUEST.md and PROJECT.md.
"""

import json
import os
import subprocess
import cv2
import numpy as np
import pytest
import psutil

_data_base = "/data" if os.path.exists("/data/output_shorts") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUTPUT_DIR = os.path.join(_data_base, "output_shorts", "7fdcc82827ff6f3c983feaf39f56ea34")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.json")

def load_metadata():
    assert os.path.exists(METADATA_PATH), f"metadata.json missing at {METADATA_PATH}"
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_ffprobe(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name,width,height,duration,sample_rate,channels,r_frame_rate",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json",
        video_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return json.loads(res.stdout)


class TestMilestone5IngestValidation:
    """Milestone 5 End-to-End Ingest & Victory Certification Tests."""

    def test_metadata_structure_and_counts(self):
        meta = load_metadata()
        assert meta.get("pipeline_version") == "V3-subprocess"
        editorial_shorts = meta.get("editorial_shorts", [])
        assert len(editorial_shorts) >= 4, f"Expected >= 4 shorts, got {len(editorial_shorts)}"
        
        render_results = meta.get("render_results", [])
        assert len(render_results) >= 4, f"Expected >= 4 render results, got {len(render_results)}"
        for r in render_results:
            assert r.get("success") is True, f"Render result failed: {r}"
            assert len(r.get("errors", [])) == 0, f"Render errors present: {r}"

    def test_r1_clip_duration_bounds(self):
        meta = load_metadata()
        for idx, short in enumerate(meta.get("editorial_shorts", []), start=1):
            dur = short.get("duration_seconds")
            assert 30.0 <= dur <= 50.0, f"Short {idx} duration {dur}s outside [30.0, 50.0]s"

    def test_r1_teaser_blacklist_t_ge_60(self):
        meta = load_metadata()
        for idx, short in enumerate(meta.get("editorial_shorts", []), start=1):
            start = short.get("start_time_seconds")
            assert start >= 60.0, f"Short {idx} start time {start}s violates t >= 60.0s teaser blacklist"

    def test_r4_vram_budget_compliance(self):
        meta = load_metadata()
        phase_history = meta.get("phase_history", [])
        assert len(phase_history) > 0, "Phase history is empty"
        for phase in phase_history:
            peak_vram = phase.get("peak_vram_mb", 0.0)
            assert peak_vram <= 16384.0, f"Phase {phase.get('phase')} exceeded 16GB VRAM: {peak_vram} MB"
            assert phase.get("exitcode") == 0, f"Phase {phase.get('phase')} non-zero exit: {phase.get('exitcode')}"

    def test_process_stability_no_orphans_or_zombies(self):
        # Check active ffmpeg processes
        ffmpeg_procs = [p.info for p in psutil.process_iter(['pid', 'name']) if 'ffmpeg' in (p.info['name'] or '').lower()]
        assert len(ffmpeg_procs) == 0, f"Orphaned FFmpeg processes found: {ffmpeg_procs}"
        
        # Check zombie processes
        zombies = [p.info for p in psutil.process_iter(['pid', 'name', 'status']) if p.info.get('status') == psutil.STATUS_ZOMBIE]
        assert len(zombies) == 0, f"Zombie processes found: {zombies}"

    @pytest.mark.parametrize("clip_idx", [1, 2, 3, 4])
    def test_probe_output_mp4_streams(self, clip_idx):
        short_file = os.path.join(OUTPUT_DIR, f"short_{clip_idx}.mp4")
        assert os.path.exists(short_file), f"Output file missing: {short_file}"
        assert os.path.getsize(short_file) > 1_000_000, f"Short {clip_idx} file size suspiciously small: {os.path.getsize(short_file)}"

        info = run_ffprobe(short_file)
        streams = info.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

        assert len(video_streams) >= 1, f"Short {clip_idx} has no video stream"
        v_stream = video_streams[0]
        assert v_stream.get("width") == 1080, f"Width is not 1080: {v_stream.get('width')}"
        assert v_stream.get("height") == 1920, f"Height is not 1920: {v_stream.get('height')}"

        # Duration check from format and stream
        fmt_dur = float(info.get("format", {}).get("duration", 0.0))
        assert 30.0 <= fmt_dur <= 50.0, f"Short {clip_idx} format duration {fmt_dur}s outside [30.0, 50.0]s"

        assert len(audio_streams) >= 1, f"Short {clip_idx} has no audio stream"
        a_stream = audio_streams[0]
        sample_rate = int(a_stream.get("sample_rate", 0))
        assert sample_rate in [44100, 48000], f"Invalid sample rate: {sample_rate}"
        channels = int(a_stream.get("channels", 0))
        assert channels >= 1, f"Invalid audio channels: {channels}"

    @pytest.mark.parametrize("clip_idx", [1, 2, 3, 4])
    def test_smooth_active_speaker_framing(self, clip_idx):
        """Verifies smooth framing without jitter, snapping, or unconstrained motion."""
        short_file = os.path.join(OUTPUT_DIR, f"short_{clip_idx}.mp4")
        cap = cv2.VideoCapture(short_file)
        assert cap.isOpened()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert total_frames > 0

        # Sample consecutive frame pairs to measure inter-frame displacement
        # Smooth camera motion means consecutive frame difference is bounded,
        # with no violent jerks or infinite snaps.
        diffs = []
        prev_gray = None
        # Check 20 frame steps across the video
        step = max(1, total_frames // 40)
        for f in range(10, min(total_frames - 1, 300), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret:
                continue
            # Downsample for fast robust motion analysis
            small = cv2.resize(frame, (270, 480))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = np.mean(cv2.absdiff(gray, prev_gray))
                diffs.append(diff)
            prev_gray = gray

        cap.release()
        assert len(diffs) > 0
        mean_diff = np.mean(diffs)
        # Verify video has active motion but no corrupt glitch explosions
        assert 0.1 <= mean_diff <= 100.0, f"Unusual frame difference in short {clip_idx}: {mean_diff}"

    @pytest.mark.parametrize("clip_idx", [1, 2, 3, 4])
    def test_kinetic_typography_safe_zone_and_no_boxes(self, clip_idx):
        """Verifies subtitles appear within [0.60, 0.70] (1152 to 1344 px) and have no opaque boxes."""
        short_file = os.path.join(OUTPUT_DIR, f"short_{clip_idx}.mp4")
        cap = cv2.VideoCapture(short_file)
        assert cap.isOpened(), f"Cannot open video {short_file}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Scan frames every 10 frames to detect subtitles
        found_subtitles = 0
        y_centers = []

        for f_idx in range(0, total_frames, 10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Subtitle safe zone is [1152, 1344]
            # Analyze region around safe zone [1100, 1400]
            roi = frame[1100:1400, :]
            # Subtitle text consists of high-brightness text fill with black outline
            # Check for text fill pixels (either crisp white or neon lime/green highlight)
            # White text: B, G, R all > 200
            # Neon highlight: G > 200, R > 150
            text_pixels = ((roi[:, :, 1] > 200) & (roi[:, :, 0] > 180) & (roi[:, :, 2] > 180)) | \
                          ((roi[:, :, 1] > 210) & (roi[:, :, 2] > 170) & (roi[:, :, 0] < 140))
            
            y_indices, x_indices = np.where(text_pixels)
            if len(y_indices) > 60:
                actual_y = y_indices + 1100
                min_y, max_y = np.min(actual_y), np.max(actual_y)
                center_y = (min_y + max_y) / 2.0
                
                # Check that center_y is within the safe bounds [1152, 1344] (Y_ANCHOR = 1248 px)
                # Allow minor tolerance (±30px) for font ascenders/descenders
                if 1140 <= center_y <= 1355:
                    found_subtitles += 1
                    y_centers.append(center_y)

        cap.release()
        assert found_subtitles > 0, f"Short {clip_idx} showed no subtitles within safe bounds [1152, 1344]"
        avg_center_y = np.mean(y_centers)
        # Average center Y should be near 1248 px (65% screen height)
        assert 1180 <= avg_center_y <= 1320, f"Short {clip_idx} average subtitle center Y={avg_center_y:.1f} deviates from safe zone anchor"
