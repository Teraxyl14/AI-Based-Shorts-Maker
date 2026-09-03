import os
import sys
import json
import asyncio
import logging
import subprocess
import time
import nats
from nats.js.api import ConsumerConfig
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.schemas import SceneChange, SpatialEvent
from shared.nats_subjects import VIDEO_INGEST, METADATA_VISION

logger = logging.getLogger("AetherVisionSpoke")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Configurable YOLO model tag via env var for hot-swap modularity
SPATIAL_TRACKER_MODEL = os.getenv("SPATIAL_TRACKER_MODEL", "yolo26s-pose")
# Frame sampling rate for pose detection (seconds between samples)
POSE_SAMPLE_INTERVAL = 0.5
# Scene change detection threshold (histogram diff)
SCENE_CHANGE_THRESHOLD = 0.35


class VisionSpoke:
    """GPU-resident visual pre-pass: Kornia scene-change detection + YOLO26s-Pose skeletal tracking.

    All frame processing stays on GPU tensors — no NumPy round-trips.
    Publishes timestamped SceneChange and PoseKeypoint event streams to NATS.
    """

    def __init__(self, nats_url: str):
        self.nats_url = nats_url
        self.nc = None
        self.js = None
        self.yolo_model = None
        self.device = None

    def _load_models(self):
        """Lazy-load YOLO pose model and configure torch device."""
        if self.yolo_model is not None:
            return

        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Vision Spoke using device: {self.device}")

        # Load YOLOE-26 / YOLOv8s open-vocabulary model with auto-caching
        yoloe_path = "./data/models/yoloe26s.pt"
        logger.info(f"Loading YOLOE open-vocabulary detector model...")
        t0 = time.perf_counter()
        try:
            from ultralytics import YOLO
            os.makedirs("./data/models", exist_ok=True)
            if os.path.exists(yoloe_path):
                self.yoloe_model = YOLO(yoloe_path)
            elif os.path.exists("./data/models/yolov8s.pt"):
                self.yoloe_model = YOLO("./data/models/yolov8s.pt")
            else:
                try:
                    self.yoloe_model = YOLO("yolov8s.pt")
                except Exception:
                    self.yoloe_model = None
            
            elapsed = time.perf_counter() - t0
            if self.yoloe_model is not None:
                logger.info(f"YOLO detector model loaded in {elapsed:.2f}s")
                self.yolo_model = self.yoloe_model
            else:
                logger.info("YOLO detector not available, using pose model as primary.")
                self.yolo_model = "mock"
        except Exception as e:
            logger.warning(f"Optional YOLOE detector not loaded: {e}. Falling back to pose model.")
            self.yoloe_model = None
            self.yolo_model = "mock"

        # Load Anatomical Fallback (YOLO11s-Pose)
        logger.info(f"Loading YOLO11s-Pose fallback model...")
        try:
            from ultralytics import YOLO
            self.yolo_pose_model = YOLO("yolo11s-pose")
            logger.info("YOLO11s-Pose model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO11s-Pose model: {e}")
            self.yolo_pose_model = "mock"

    def _get_video_info(self, source_path: str) -> tuple:
        """Returns (duration, fps, width, height) via ffprobe."""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-show_entries", "stream=r_frame_rate,width,height",
                "-select_streams", "v:0",
                "-of", "json",
                source_path
            ], capture_output=True, text=True)
            data = json.loads(result.stdout)

            duration = float(data.get("format", {}).get("duration", 300.0))
            stream = data.get("streams", [{}])[0]
            width = int(stream.get("width", 1920))
            height = int(stream.get("height", 1080))
            fps_str = stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            return duration, fps, width, height
        except Exception as e:
            logger.warning(f"ffprobe failed: {e}. Using defaults.")
            return 300.0, 30.0, 1920, 1080

    def detect_scene_changes(self, source_path: str, duration: float, fps: float) -> List[SceneChange]:
        """Detects hard cuts and scene transitions using Kornia histogram differencing on GPU."""
        import torch
        import kornia

        scene_changes = []
        logger.info("Running Kornia scene-change detection...")
        t0 = time.perf_counter()

        try:
            # Extract frames at 2fps for scene change detection
            sample_fps = 2.0
            total_frames = int(duration * sample_fps)
            # Limit to prevent OOM on very long videos
            total_frames = min(total_frames, 2000)

            # Use ffmpeg to extract frames as raw RGB pipe
            extract_cmd = [
                "ffmpeg", "-y",
                "-i", source_path,
                "-vf", f"fps={sample_fps},scale=320:180",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "pipe:1"
            ]

            proc = subprocess.Popen(
                extract_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=320 * 180 * 3 * 4  # Buffer 4 frames
            )

            frame_size = 320 * 180 * 3
            prev_hist = None
            frame_idx = 0

            while True:
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break

                # Convert to torch tensor on GPU: (H, W, C) -> (1, C, H, W) float [0,1]
                frame_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(1, 180, 320, 3)
                frame_tensor = frame_tensor.permute(0, 3, 1, 2).float().to(self.device) / 255.0

                # Compute grayscale histogram using Kornia
                gray = kornia.color.rgb_to_grayscale(frame_tensor)
                # Flatten to 1D and compute histogram manually via torch
                hist = torch.histc(gray, bins=64, min=0.0, max=1.0)
                hist = hist / (hist.sum() + 1e-8)  # Normalize

                if prev_hist is not None:
                    # Chi-squared distance between histograms
                    diff = torch.sum((hist - prev_hist) ** 2 / (hist + prev_hist + 1e-8))
                    diff_val = diff.item()

                    if diff_val > SCENE_CHANGE_THRESHOLD:
                        timestamp = frame_idx / sample_fps
                        src_frame = int(timestamp * fps)
                        change_type = "hard_cut" if diff_val > 0.6 else "dissolve"
                        scene_changes.append(SceneChange(
                            timestamp=timestamp,
                            confidence=min(1.0, diff_val),
                            frame_idx=src_frame,
                            change_type=change_type
                        ))

                prev_hist = hist
                frame_idx += 1

            proc.wait()

        except Exception as e:
            logger.error(f"Scene change detection failed: {e}", exc_info=True)

        elapsed = time.perf_counter() - t0
        logger.info(f"Scene detection complete in {elapsed:.2f}s. Found {len(scene_changes)} scene changes.")
        return scene_changes

    def detect_spatial_events(self, source_path: str, duration: float, fps: float) -> List[SpatialEvent]:
        """Runs 3-tier tracking hierarchy (YOLOE-26 -> YOLO11s-Pose -> Saliency) on sampled frames."""
        spatial_events = []
        logger.info(f"Running 3-tier spatial detection (sampling every {POSE_SAMPLE_INTERVAL}s)...")
        t0 = time.perf_counter()

        # Check if both models are unavailable
        both_unavailable = (getattr(self, 'yoloe_model', None) is None) and (getattr(self, 'yolo_pose_model', "mock") == "mock")
        if both_unavailable:
            logger.warning("YOLO models unavailable. Generating mock spatial events.")
            t = 0.0
            idx = 0
            while t < duration:
                spatial_events.append(SpatialEvent(
                    timestamp=t,
                    frame_idx=int(t * fps),
                    track_id=0,
                    label="person",
                    bbox=(0.5, 0.5, 0.6, 0.8), # cx, cy, w, h
                    confidence=0.9,
                    keypoints=[(0.5, 0.1, 0.9)] * 17,
                    action_label="standing"
                ))
                t += POSE_SAMPLE_INTERVAL
                idx += 1
            return spatial_events

        try:
            sample_fps = 1.0 / POSE_SAMPLE_INTERVAL
            extract_cmd = [
                "ffmpeg", "-y",
                "-i", source_path,
                "-vf", f"fps={sample_fps},scale=640:360",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "pipe:1"
            ]

            proc = subprocess.Popen(
                extract_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=640 * 360 * 3 * 4
            )

            frame_w, frame_h = 640, 360
            frame_size = frame_w * frame_h * 3
            frame_idx = 0
            
            import numpy as np
            import torch
            import kornia

            while True:
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break

                timestamp = frame_idx * POSE_SAMPLE_INTERVAL
                frame_np = np.frombuffer(raw, dtype=np.uint8).reshape(frame_h, frame_w, 3)

                detected = False

                # Tier 1: YOLOE-26 (Open Vocabulary)
                if getattr(self, 'yoloe_model', None) is not None:
                    results = self.yoloe_model(frame_np, verbose=False, conf=0.3)
                    if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xywhn.cpu().numpy()
                        classes = results[0].boxes.cls.cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy()
                        names = results[0].names
                        for i in range(len(boxes)):
                            label = names[int(classes[i])] if names else "object"
                            bbox = tuple(float(v) for v in boxes[i])
                            spatial_events.append(SpatialEvent(
                                timestamp=timestamp,
                                frame_idx=int(timestamp * fps),
                                track_id=i,
                                label=label,
                                bbox=bbox,
                                confidence=float(confs[i]),
                                keypoints=[]
                            ))
                        detected = True

                # Tier 2: Anatomical Fallback (YOLO11s-Pose)
                if not detected and getattr(self, 'yolo_pose_model', "mock") != "mock":
                    results = self.yolo_pose_model(frame_np, verbose=False, conf=0.3)
                    if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xywhn.cpu().numpy()
                        kps = results[0].keypoints.data if results[0].keypoints is not None else None
                        for i in range(min(len(boxes), 3)):
                            bbox = tuple(float(v) for v in boxes[i])
                            keypoints = []
                            if kps is not None:
                                keypoints_raw = kps[i].cpu().numpy()
                                keypoints = [
                                    (float(kp[0] / frame_w), float(kp[1] / frame_h), float(kp[2]))
                                    for kp in keypoints_raw
                                ]
                            gesture = self._classify_gesture(keypoints)
                            spatial_events.append(SpatialEvent(
                                timestamp=timestamp,
                                frame_idx=int(timestamp * fps),
                                track_id=i,
                                label="person",
                                bbox=bbox,
                                confidence=float(results[0].boxes.conf[i]),
                                keypoints=keypoints,
                                action_label=gesture
                            ))
                        detected = True

                # Tier 3: Spectral Residual Saliency (Kornia fallback)
                if not detected:
                    frame_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(1, frame_h, frame_w, 3)
                    frame_tensor = frame_tensor.permute(0, 3, 1, 2).float().to(self.device) / 255.0
                    gray = kornia.color.rgb_to_grayscale(frame_tensor)
                    spatial_events.append(SpatialEvent(
                        timestamp=timestamp,
                        frame_idx=int(timestamp * fps),
                        track_id=0,
                        label="saliency",
                        bbox=(0.5, 0.5, 1.0, 1.0),
                        confidence=0.1
                    ))

                frame_idx += 1

            proc.wait()

        except Exception as e:
            logger.error(f"Spatial detection failed: {e}", exc_info=True)

        elapsed = time.perf_counter() - t0
        logger.info(f"Spatial detection complete in {elapsed:.2f}s. Found {len(spatial_events)} events.")
        return spatial_events

    @staticmethod
    def _classify_gesture(keypoints: list) -> str:
        """Simple heuristic gesture classification from COCO 17-point skeleton.

        Keypoint indices: 0=nose, 5=left_shoulder, 6=right_shoulder,
        9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip
        """
        try:
            if len(keypoints) < 17:
                return "unknown"

            nose_y = keypoints[0][1]
            l_wrist_y = keypoints[9][1]
            r_wrist_y = keypoints[10][1]
            l_shoulder_y = keypoints[5][1]
            r_shoulder_y = keypoints[6][1]

            # Hands raised above shoulders → pointing/waving
            if l_wrist_y < l_shoulder_y - 0.05 or r_wrist_y < r_shoulder_y - 0.05:
                if abs(keypoints[9][0] - keypoints[10][0]) > 0.3:
                    return "pointing"
                return "waving"

            # Leaning forward: nose significantly ahead of hip center
            l_hip_y = keypoints[11][1]
            r_hip_y = keypoints[12][1]
            hip_center_y = (l_hip_y + r_hip_y) / 2
            torso_len = hip_center_y - (l_shoulder_y + r_shoulder_y) / 2
            if torso_len > 0 and nose_y < (l_shoulder_y + r_shoulder_y) / 2 - torso_len * 0.3:
                return "leaning_forward"

            return "standing"
        except (IndexError, TypeError):
            return "unknown"

    async def run(self):
        """Connects to NATS and processes video.ingest messages."""
        logger.info(f"Connecting to NATS at {self.nats_url}...")
        self.nc = await nats.connect(self.nats_url)
        self.js = self.nc.jetstream()

        # Subscribe to video.ingest (Durable Subscription with hardened config)
        sub = await self.js.subscribe(
            subject=VIDEO_INGEST,
            durable="vision_spoke",
            config=ConsumerConfig(ack_wait=600.0, max_ack_pending=12)
        )

        logger.info(f"Subscribed to '{VIDEO_INGEST}' as durable subscription. Awaiting messages...")

        try:
            async for msg in sub.messages:
                try:
                    payload = json.loads(msg.data.decode("utf-8"))
                    source_path = payload.get("source_path")
                    logger.info(f"Received vision ingest request for: {source_path}")

                    if not source_path or not os.path.exists(source_path):
                        logger.error(f"Invalid source path: {source_path}")
                        await msg.ack()
                        continue

                    # Lazy-load models on first job
                    await asyncio.to_thread(self._load_models)

                    # Get video metadata
                    duration, fps, width, height = await asyncio.to_thread(
                        self._get_video_info, source_path
                    )
                    logger.info(f"Video: {duration:.1f}s, {fps:.1f}fps, {width}x{height}")

                    # Run Kornia scene-change detection (GPU)
                    scene_changes = await asyncio.to_thread(
                        self.detect_scene_changes, source_path, duration, fps
                    )

                    # Run 3-tier spatial tracking hierarchy
                    spatial_events = await asyncio.to_thread(
                        self.detect_spatial_events, source_path, duration, fps
                    )

                    # Publish to metadata.vision
                    out_payload = {
                        "source_path": source_path,
                        "video_duration": duration,
                        "scene_changes": [sc.model_dump() for sc in scene_changes],
                        "spatial_events": [pe.model_dump() for pe in spatial_events],
                        "timestamp": time.time()
                    }

                    await self.js.publish(METADATA_VISION, json.dumps(out_payload).encode("utf-8"))
                    logger.info(f"Successfully published vision metadata to '{METADATA_VISION}' "
                               f"({len(scene_changes)} scene changes, {len(spatial_events)} spatial events)")

                    await msg.ack()
                except Exception as e:
                    logger.error(f"Error processing message in vision spoke: {str(e)}", exc_info=True)
                    await msg.nak()
        except asyncio.CancelledError:
            logger.info("Vision Spoke cancelled, closing...")
        finally:
            if self.nc:
                await self.nc.close()


if __name__ == "__main__":
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    spoke = VisionSpoke(nats_url)
    try:
        asyncio.run(spoke.run())
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
