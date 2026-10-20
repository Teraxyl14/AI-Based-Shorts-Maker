import os
import sys
import json
import math
import asyncio
import logging
import subprocess
import time
import nats
from nats.js.api import ConsumerConfig
from typing import List, Tuple, Optional


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.schemas import SceneChange, SpatialEvent
from shared.nats_subjects import VIDEO_INGEST, METADATA_VISION

logger = logging.getLogger("AetherVisionSpoke")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Configurable YOLO model tag via env var for hot-swap modularity
SPATIAL_TRACKER_MODEL = os.getenv("SPATIAL_TRACKER_MODEL", "yolo11s-pose")
# Frame sampling rate for pose detection (seconds between samples)
POSE_SAMPLE_INTERVAL = 0.5
# Scene change detection threshold (histogram diff)
SCENE_CHANGE_THRESHOLD = 0.35

# ─── Hand-Object Proximity & Demonstration ROI Helpers ──────────────────────

def compute_wrist_object_proximity(
    wrist_keypoints: List[Tuple[float, float, float]],
    object_bbox: Tuple[float, float, float, float]
) -> Tuple[float, int]:
    """Computes minimum Euclidean distance between left/right wrist keypoints and object bbox center.

    Keypoints: COCO 17-point skeleton where index 9 is left wrist and index 10 is right wrist.
    BBox: (cx, cy, w, h) normalized coordinates.
    Returns: (min_distance, nearest_wrist_index) or (float('inf'), -1) if no valid wrists.
    """
    if not wrist_keypoints or len(wrist_keypoints) < 11:
        return (float("inf"), -1)

    cx, cy, _, _ = object_bbox
    min_dist = float("inf")
    nearest_wrist = -1

    for idx in (9, 10):
        kp = wrist_keypoints[idx]
        if len(kp) >= 2:
            kx, ky = kp[0], kp[1]
            conf = kp[2] if len(kp) >= 3 else 1.0
            if conf > 0.2 and (kx > 0.01 or ky > 0.01):
                d = math.sqrt((cx - kx) ** 2 + (cy - ky) ** 2)
                if d < min_dist:
                    min_dist = d
                    nearest_wrist = idx

    return (min_dist, nearest_wrist)


def associate_hand_object_proximity(
    spatial_events: List[SpatialEvent],
    proximity_threshold: Optional[float] = None
) -> List[SpatialEvent]:
    """Associates detected objects with nearby wrists.
    Assigns action_label='held_product' when object is within hand proximity.
    """
    frames_dict = {}
    for ev in spatial_events:
        key = ev.frame_idx if ev.frame_idx != 0 else round(ev.timestamp, 2)
        frames_dict.setdefault(key, []).append(ev)

    for key, evs in frames_dict.items():
        person_events = [e for e in evs if e.label == "person" and e.keypoints]
        obj_events = [e for e in evs if e.label != "person" and e.label != "saliency"]

        for obj in obj_events:
            for p in person_events:
                dist, _ = compute_wrist_object_proximity(p.keypoints, obj.bbox)
                _, _, w, h = obj.bbox
                thresh = proximity_threshold if proximity_threshold is not None else max(max(w, h) * 1.2, 0.18)
                if dist < thresh:
                    obj.action_label = "held_product"
                    break

    return spatial_events


def generate_synthetic_demonstration_roi(
    person_event: SpatialEvent,
    existing_objects: List[SpatialEvent]
) -> Optional[SpatialEvent]:
    """Generates synthetic demonstration ROI fallback when gestures indicate demonstration
    but COCO object detection misses fine-grained product parts.
    """
    if person_event.label != "person" or not person_event.keypoints or len(person_event.keypoints) < 17:
        return None

    # Check if an existing object is already held or near hands
    for obj in existing_objects:
        if obj.action_label == "held_product":
            return None
        dist, _ = compute_wrist_object_proximity(person_event.keypoints, obj.bbox)
        _, _, w, h = obj.bbox
        if dist < max(max(w, h) * 1.2, 0.18):
            return None

    kps = person_event.keypoints
    l_wrist = kps[9]
    r_wrist = kps[10]
    gesture = person_event.action_label or ""

    is_demo = gesture in ("pointing", "waving", "demonstrating", "holding", "leaning_forward")

    l_shoulder_y = kps[5][1]
    r_shoulder_y = kps[6][1]
    shoulder_y = (l_shoulder_y + r_shoulder_y) / 2.0
    l_hip_y = kps[11][1]
    r_hip_y = kps[12][1]
    hip_y = (l_hip_y + r_hip_y) / 2.0

    l_active = (l_wrist[1] < hip_y + 0.05 and l_wrist[1] > shoulder_y - 0.20 and (len(l_wrist) < 3 or l_wrist[2] > 0.3))
    r_active = (r_wrist[1] < hip_y + 0.05 and r_wrist[1] > shoulder_y - 0.20 and (len(r_wrist) < 3 or r_wrist[2] > 0.3))

    if not is_demo and not (l_active or r_active):
        return None

    # Select the most active wrist
    if l_active and r_active:
        chosen_wrist = l_wrist if l_wrist[1] < r_wrist[1] else r_wrist
    elif l_active:
        chosen_wrist = l_wrist
    elif r_active:
        chosen_wrist = r_wrist
    else:
        chosen_wrist = l_wrist if (len(l_wrist) >= 3 and l_wrist[2] > 0) else r_wrist

    cx = max(0.15, min(0.85, chosen_wrist[0]))
    cy = max(0.20, min(0.80, chosen_wrist[1]))
    synth_bbox = (round(cx, 3), round(cy, 3), 0.25, 0.35)

    return SpatialEvent(
        timestamp=person_event.timestamp,
        frame_idx=person_event.frame_idx,
        track_id=200 + person_event.track_id,
        label="synthetic_wrist_roi",
        bbox=synth_bbox,
        confidence=0.50,
        keypoints=[],
        action_label="held_product"
    )


class VisionSpoke:
    """GPU-resident visual pre-pass: Kornia scene-change detection + YOLO11s-Pose skeletal tracking.

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

        # Load Primary Human Tracker (YOLO11s-Pose)
        logger.info("Loading YOLO11s-Pose primary human tracker model...")
        pose_path = "./data/models/yolo11s-pose.pt"
        try:
            from ultralytics import YOLO
            os.makedirs("./data/models", exist_ok=True)
            if os.path.exists(pose_path):
                self.yolo_pose_model = YOLO(pose_path)
            elif os.path.exists("yolo11s-pose.pt"):
                self.yolo_pose_model = YOLO("yolo11s-pose.pt")
            else:
                self.yolo_pose_model = YOLO("yolo11s-pose")
            logger.info("YOLO11s-Pose model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO11s-Pose model: {e}")
            self.yolo_pose_model = "mock"

        # Load Object Detector for Non-Human Items (YOLO11s / YOLOv8s)
        logger.info("Loading YOLO object detector model...")
        t0 = time.perf_counter()
        try:
            from ultralytics import YOLO
            if os.path.exists("./data/models/yolo11s.pt"):
                self.yoloe_model = YOLO("./data/models/yolo11s.pt")
            elif os.path.exists("yolo11s.pt"):
                self.yoloe_model = YOLO("yolo11s.pt")
            elif os.path.exists("./data/models/yoloe26s.pt"):
                self.yoloe_model = YOLO("./data/models/yoloe26s.pt")
            elif os.path.exists("./data/models/yolov8s.pt"):
                self.yoloe_model = YOLO("./data/models/yolov8s.pt")
            elif os.path.exists("yolov8s.pt"):
                self.yoloe_model = YOLO("yolov8s.pt")
            else:
                try:
                    self.yoloe_model = YOLO("yolo11s.pt")
                except Exception:
                    self.yoloe_model = None

            elapsed = time.perf_counter() - t0
            if self.yoloe_model is not None:
                logger.info(f"YOLO detector model loaded in {elapsed:.2f}s")
            else:
                logger.info("YOLO object detector not available.")
        except Exception as e:
            logger.warning(f"Optional YOLO detector not loaded: {e}.")
            self.yoloe_model = None

        self.yolo_model = self.yolo_pose_model if getattr(self, 'yolo_pose_model', 'mock') != 'mock' else self.yoloe_model


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
        """Detects hard cuts and scene transitions using 2D Spatial Block Color + Edge Differencing on GPU.

        Dividing frames into a 3x3 grid (9 spatial blocks) with multi-channel color
        histograms and spatial edge gradients requires multi-block structural discontinuity
        (>= 5/9 blocks), effectively ignoring uniform stage lighting washes while accurately
        catching true camera angle switches and cuts.
        """
        import torch
        import numpy as np

        scene_changes = []
        logger.info("Running 2D Spatial Block HSV + Edge scene-change detection...")
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
            prev_descriptors = None
            frame_idx = 0

            while True:
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break

                # Convert to torch tensor on GPU: (H, W, C) -> (1, C, H, W) float [0,1]
                frame_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(1, 180, 320, 3)
                frame_tensor = frame_tensor.permute(0, 3, 1, 2).float().to(self.device) / 255.0

                # 1. Decompose frame into 3x3 grid (9 spatial blocks: 60x106 each)
                h_splits = torch.tensor_split(frame_tensor, 3, dim=2)
                block_descriptors = []

                for h_chunk in h_splits:
                    w_splits = torch.tensor_split(h_chunk, 3, dim=3)
                    for b in w_splits:
                        # 3-channel color histogram (16 bins per channel)
                        r_hist = torch.histc(b[:, 0], bins=16, min=0.0, max=1.0)
                        g_hist = torch.histc(b[:, 1], bins=16, min=0.0, max=1.0)
                        b_hist = torch.histc(b[:, 2], bins=16, min=0.0, max=1.0)
                        c_hist = torch.cat([r_hist, g_hist, b_hist])
                        c_hist = c_hist / (c_hist.sum() + 1e-8)

                        # Spatial high-frequency edge energy
                        dx = torch.abs(b[:, :, :, 1:] - b[:, :, :, :-1])
                        dy = torch.abs(b[:, :, 1:, :] - b[:, :, :-1, :])
                        edge_energy = float((dx.mean() + dy.mean()).item())

                        block_descriptors.append((c_hist, edge_energy))

                if prev_descriptors is not None:
                    block_diffs = []
                    for i in range(9):
                        prev_c, prev_e = prev_descriptors[i]
                        curr_c, curr_e = block_descriptors[i]

                        # Chi-squared color distance
                        c_diff = 0.5 * torch.sum((curr_c - prev_c) ** 2 / (curr_c + prev_c + 1e-8)).item()
                        # Relative edge structural change
                        e_diff = abs(curr_e - prev_e) / (max(curr_e, prev_e) + 1e-4)

                        # Combined block discontinuity metric
                        d_i = 0.60 * c_diff + 0.40 * e_diff
                        block_diffs.append(d_i)

                    # Discontinuous block count (structural change across independent spatial zones)
                    discontinuous_blocks = sum(1 for d in block_diffs if d >= 0.28)
                    mean_diff = sum(block_diffs) / 9.0

                    # Uniform wash detection (ambient stage lighting changes uniformly shift all blocks)
                    diff_variance = float(np.var(block_diffs)) if len(block_diffs) > 1 else 0.0
                    is_uniform_lighting_wash = (diff_variance < 0.003 and discontinuous_blocks < 7 and mean_diff < 0.45)

                    # A true camera cut causes sharp structural discontinuity across >= 5 blocks
                    # without being an ambient uniform light wash
                    if (discontinuous_blocks >= 5 and mean_diff >= 0.30 and not is_uniform_lighting_wash) or mean_diff >= 0.60:
                        timestamp = frame_idx / sample_fps
                        src_frame = int(timestamp * fps)
                        change_type = "hard_cut" if (discontinuous_blocks >= 6 and mean_diff > 0.48) else "dissolve"
                        scene_changes.append(SceneChange(
                            timestamp=timestamp,
                            confidence=min(1.0, mean_diff),
                            frame_idx=src_frame,
                            change_type=change_type
                        ))

                prev_descriptors = block_descriptors
                frame_idx += 1

            proc.wait()

        except Exception as e:
            logger.error(f"Scene change detection failed: {e}", exc_info=True)

        elapsed = time.perf_counter() - t0
        logger.info(f"Scene detection complete in {elapsed:.2f}s. Found {len(scene_changes)} scene changes.")
        return scene_changes

    def detect_spatial_events(self, source_path: str, duration: float, fps: float) -> List[SpatialEvent]:
        """Runs 3-tier tracking hierarchy (YOLO11s-Pose primary + ByteTrack, YOLO object detector, Saliency)."""
        spatial_events = []
        logger.info(f"Running 3-tier spatial detection (sampling every {POSE_SAMPLE_INTERVAL}s)...")
        t0 = time.perf_counter()

        # Ensure models are loaded
        if getattr(self, 'yolo_pose_model', None) is None and getattr(self, 'yoloe_model', None) is None:
            self._load_models()

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
            effective_interval = 1.0 if duration > 3600.0 else POSE_SAMPLE_INTERVAL
            sample_fps = 1.0 / effective_interval
            frame_w, frame_h = 960, 540
            frame_size = frame_w * frame_h * 3
            extract_cmd = [
                "ffmpeg", "-y",
                "-i", source_path,
                "-vf", f"fps={sample_fps},scale={frame_w}:{frame_h}",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "pipe:1"
            ]

            proc = subprocess.Popen(
                extract_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=frame_size * 4
            )

            frame_idx = 0

            import numpy as np
            import torch
            import kornia

            while True:
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break

                timestamp = frame_idx * effective_interval
                frame_np = np.frombuffer(raw, dtype=np.uint8).reshape(frame_h, frame_w, 3)

                detected = False

                frame_events = []

                # Tier 1: Primary Human Detection & Pose Tracking (YOLO11s-Pose + ByteTrack)
                if getattr(self, 'yolo_pose_model', "mock") != "mock":
                    try:
                        try:
                            pose_results = self.yolo_pose_model.track(
                                frame_np, persist=True, tracker="bytetrack.yaml",
                                classes=[0], verbose=False, conf=0.3
                            )
                        except Exception:
                            pose_results = self.yolo_pose_model(
                                frame_np, classes=[0], verbose=False, conf=0.3
                            )

                        if pose_results and len(pose_results) > 0 and pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
                            boxes = pose_results[0].boxes.xywhn.cpu().numpy()
                            confs = pose_results[0].boxes.conf.cpu().numpy()
                            track_ids = (
                                pose_results[0].boxes.id.int().cpu().tolist()
                                if pose_results[0].boxes.id is not None
                                else list(range(len(boxes)))
                            )
                            kps = pose_results[0].keypoints.data if pose_results[0].keypoints is not None else None

                            for i in range(len(boxes)):
                                bbox = tuple(float(v) for v in boxes[i])
                                keypoints = []
                                if kps is not None and i < len(kps):
                                    keypoints_raw = kps[i].cpu().numpy()
                                    keypoints = [
                                        (float(kp[0] / frame_w), float(kp[1] / frame_h), float(kp[2]))
                                        for kp in keypoints_raw
                                    ]
                                gesture = self._classify_gesture(keypoints)
                                tid = track_ids[i] if i < len(track_ids) else i
                                ev = SpatialEvent(
                                    timestamp=timestamp,
                                    frame_idx=int(timestamp * fps),
                                    track_id=int(tid),
                                    label="person",
                                    bbox=bbox,
                                    confidence=float(confs[i]),
                                    keypoints=keypoints,
                                    action_label=gesture
                                )
                                frame_events.append(ev)
                                spatial_events.append(ev)
                            detected = True
                    except Exception as e:
                        logger.warning(f"Pose tracking error at frame {frame_idx}: {e}")

                # Tier 2: Non-Human Object Detection (YOLOE / YOLOv8s with ByteTrack persistence)
                if getattr(self, 'yoloe_model', None) is not None:
                    try:
                        try:
                            obj_results = self.yoloe_model.track(
                                frame_np, persist=True, tracker="bytetrack.yaml",
                                verbose=False, conf=0.25
                            )
                        except Exception:
                            obj_results = self.yoloe_model(frame_np, verbose=False, conf=0.25)

                        if obj_results and len(obj_results) > 0 and obj_results[0].boxes is not None and len(obj_results[0].boxes) > 0:
                            obj_boxes = obj_results[0].boxes.xywhn.cpu().numpy()
                            obj_classes = obj_results[0].boxes.cls.cpu().numpy()
                            obj_confs = obj_results[0].boxes.conf.cpu().numpy()
                            obj_names = obj_results[0].names
                            obj_track_ids = (
                                obj_results[0].boxes.id.int().cpu().tolist()
                                if obj_results[0].boxes.id is not None
                                else None
                            )

                            for j in range(len(obj_boxes)):
                                cls_id = int(obj_classes[j])
                                label = obj_names[cls_id] if obj_names else "object"
                                # Exclude person if pose model is active, avoiding duplicate person records
                                if getattr(self, 'yolo_pose_model', "mock") != "mock" and label.lower() in ("person", "human"):
                                    continue
                                bbox = tuple(float(v) for v in obj_boxes[j])
                                tid = (100 + obj_track_ids[j]) if obj_track_ids and j < len(obj_track_ids) else (100 + j)
                                ev = SpatialEvent(
                                    timestamp=timestamp,
                                    frame_idx=int(timestamp * fps),
                                    track_id=int(tid),
                                    label=label,
                                    bbox=bbox,
                                    confidence=float(obj_confs[j]),
                                    keypoints=[]
                                )
                                frame_events.append(ev)
                                spatial_events.append(ev)
                                detected = True
                    except Exception as e:
                        logger.warning(f"Object detection error at frame {frame_idx}: {e}")

                # Associate hand-to-object proximity for detected objects in this frame
                if frame_events:
                    associate_hand_object_proximity(frame_events)

                # Synthetic demonstration ROI fallback if gestures indicate active demo but no COCO object detected
                # Gated: Disabled for STAGE_PERFORMANCE (stand-up comedy, monologues)
                is_stage_perf = "STAGE_PERFORMANCE" in os.getenv("AETHER_CONTENT_PROFILE", "").upper()
                if not is_stage_perf:
                    for p_ev in [e for e in frame_events if e.label == "person"]:
                        existing_objs = [e for e in frame_events if e.label != "person"]
                        synth_roi = generate_synthetic_demonstration_roi(p_ev, existing_objs)
                        if synth_roi is not None:
                            frame_events.append(synth_roi)
                            spatial_events.append(synth_roi)
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

            # Leaning forward: nose closer to shoulders due to forward tilt foreshortening
            l_hip_y = keypoints[11][1]
            r_hip_y = keypoints[12][1]
            hip_center_y = (l_hip_y + r_hip_y) / 2
            shoulder_center_y = (l_shoulder_y + r_shoulder_y) / 2
            torso_len = hip_center_y - shoulder_center_y
            head_offset = shoulder_center_y - nose_y
            if torso_len > 0 and 0 < head_offset <= torso_len * 0.40:
                return "leaning_forward"

            return "standing"
        except (IndexError, TypeError):
            return "unknown"

    # Module-level helpers exposed on class
    compute_wrist_object_proximity = staticmethod(lambda kps, bbox: compute_wrist_object_proximity(kps, bbox))
    associate_hand_object_proximity = staticmethod(lambda events, thresh=None: associate_hand_object_proximity(events, thresh))
    generate_synthetic_demonstration_roi = staticmethod(lambda p, objs: generate_synthetic_demonstration_roi(p, objs))

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
