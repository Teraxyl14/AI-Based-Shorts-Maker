"""
Project Aether V2 — Global Trajectory Optimizer

Replaces the reactive 4-state hysteresis tracker with offline global camera path
optimization. Computes the entire crop trajectory before rendering begins.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger("AetherTrajectorySmoother")

# Headroom bias: shift crop center 4% upward
HEADROOM_BIAS_Y = 0.04

# Velocity clamp: max pan per tick as fraction of frame width (strictly <= 15.0 px/frame)
MAX_PAN_PER_TICK = 15.0 / 1920.0


# Savitzky-Golay parameters for short shots
SAVGOL_WINDOW = 31
SAVGOL_POLYORDER = 3


def _classify_detection_tier(label: str) -> Tuple[float, float]:
    """Classifies a detection label into the 3-tier POI weight hierarchy.

    Returns:
        (weight, min_confidence) tuple for this detection tier.
        - Primary Face: w=0.65, min_conf=0.40
        - Upper-Torso / Person: w=0.25, min_conf=0.30
        - Salient Objects: w=0.10, min_conf=0.30
    """
    label_lower = label.lower().strip() if label else ""

    # Tier 1: Primary Face Detection
    if label_lower in ("face", "head", "face_detection"):
        return 0.65, 0.40

    # Tier 2: Upper-Torso / Person (pose keypoints, shoulder, person)
    if label_lower in ("person", "torso", "upper_body", "shoulders", "pose",
                        "human", "body", "upper_torso"):
        return 0.25, 0.30

    # Tier 3: Salient Moving/Held Objects (everything else with a valid bbox)
    return 0.10, 0.30


# Default fallback POI when no detections exist
FALLBACK_POI = np.array([0.50, 0.45], dtype=np.float64)

# Label sets for focal target routing
FOCAL_DISPLAY_LABELS = frozenset({"laptop", "monitor", "tv", "screen", "display", "cell phone", "tablet"})
HELD_OBJECT_LABELS = frozenset({"cell phone", "bottle", "cup", "book", "remote", "scissors",
                                 "knife", "spoon", "fork", "mouse", "keyboard", "toothbrush"})
FACE_LABELS = frozenset({"face", "head", "face_detection"})
PERSON_LABELS = frozenset({"person", "torso", "upper_body", "shoulders", "pose",
                            "human", "body", "upper_torso"})


def _extract_poi_from_spatial_events(spatial_events: List[Any],
                                     fps: float,
                                     total_frames: int) -> np.ndarray:
    """Extracts a per-frame POI array enforcing Strict Face-Centering Priority and Zero-Lag Initial State.

    Hierarchy:
        1. Primary Face (conf >= 0.45): cx_t = face_cx (monitors, held objects, saliency strictly ignored).
        2. Upper-Torso / Person (conf >= 0.35): cx_t = person_cx (upper torso midpoint fallback).
        3. Secondary Salient Detections (conf >= 0.30): weighted centroid fallback.

    Zero-Lag Initial State:
        Finds the first valid detection in the shot/clip and back-fills frames 0..first_frame,
        guaranteeing State_0 = cx_frame_0 without dragging from 0.50.

    Returns:
        np.ndarray of shape [total_frames, 2] — normalized (cx, cy) per frame.
    """
    poi = np.full((total_frames, 2), FALLBACK_POI, dtype=np.float64)

    frame_faces: Dict[int, List[Tuple[float, float, float]]] = {}
    frame_persons: Dict[int, List[Tuple[float, float, float]]] = {}
    frame_others: Dict[int, List[Tuple[float, float, float]]] = {}

    for pe in spatial_events:
        if isinstance(pe, dict):
            ts = pe.get("timestamp", 0.0)
            bbox = pe.get("bbox", (0.5, 0.5, 1.0, 1.0))
            confidence = float(pe.get("confidence", 0.0))
            label = str(pe.get("label", "")).lower().strip()
        else:
            ts = getattr(pe, "timestamp", 0.0)
            bbox = getattr(pe, "bbox", (0.5, 0.5, 1.0, 1.0))
            confidence = float(getattr(pe, "confidence", 0.0))
            label = str(getattr(pe, "label", "")).lower().strip()

        frame_idx = min(max(0, int(ts * fps)), total_frames - 1)
        cx, cy, w, h = bbox

        if label in FACE_LABELS and confidence >= 0.45:
            frame_faces.setdefault(frame_idx, []).append((cx, cy, confidence))
        elif label in PERSON_LABELS and confidence >= 0.35:
            head_cy = max(0.08, cy - 0.32 * h)
            frame_persons.setdefault(frame_idx, []).append((cx, head_cy, confidence))
        elif label not in FACE_LABELS and label not in PERSON_LABELS and confidence >= 0.30:
            tier_w, _ = _classify_detection_tier(label)
            frame_others.setdefault(frame_idx, []).append((cx, cy, tier_w * confidence))

    frame_detected_poi: Dict[int, np.ndarray] = {}

    for frame_idx in range(total_frames):
        if frame_idx in frame_faces and len(frame_faces[frame_idx]) > 0:
            # Strict Face-Centering Priority: ignore objects, monitors, and secondary saliency
            best_face = max(frame_faces[frame_idx], key=lambda x: x[2])
            frame_detected_poi[frame_idx] = np.array([best_face[0], best_face[1] - HEADROOM_BIAS_Y], dtype=np.float64)
        elif frame_idx in frame_persons and len(frame_persons[frame_idx]) > 0:
            # Fallback 1: Upper-body torso midpoint
            best_person = max(frame_persons[frame_idx], key=lambda x: x[2])
            frame_detected_poi[frame_idx] = np.array([best_person[0], best_person[1] - HEADROOM_BIAS_Y], dtype=np.float64)
        elif frame_idx in frame_others and len(frame_others[frame_idx]) > 0:
            # Fallback 2: Secondary detections weighted centroid
            boxes = frame_others[frame_idx]
            sum_w = sum(b[2] for b in boxes)
            if sum_w > 1e-8:
                sum_cx = sum(b[0] * b[2] for b in boxes)
                sum_cy = sum(b[1] * b[2] for b in boxes)
                frame_detected_poi[frame_idx] = np.array([sum_cx / sum_w, (sum_cy / sum_w) - HEADROOM_BIAS_Y], dtype=np.float64)

    # Zero-Lag Initial State:
    # Find the first frame with valid detections and back-fill frames 0..first_frame
    first_detected = None
    for f in range(total_frames):
        if f in frame_detected_poi:
            first_detected = frame_detected_poi[f].copy()
            break

    if first_detected is None:
        first_detected = FALLBACK_POI.copy()

    current_val = first_detected.copy()
    for f in range(total_frames):
        if f in frame_detected_poi:
            current_val = frame_detected_poi[f].copy()
        poi[f] = current_val.copy()

    return poi


def _extract_poi_for_focal_target(spatial_events: List[Any],
                                  fps: float,
                                  total_frames: int,
                                  focal_target: str) -> np.ndarray:
    """Extracts per-frame POI filtered by focal_target semantics with Strict Face Priority & Zero-Lag.

    Focal target routing:
        - SPEAKER_PRIMARY / SPEAKER_REACTION: Face (conf >= 0.45) strictly prioritized, then upper-torso.
        - FOCAL_DISPLAY: Monitor/laptop/screen detections, fallback to face/person.
        - HELD_OBJECT: Small object detections + hands, fallback to face/person.
        - ACTION_SCENE: All detections equally weighted.

    Returns:
        np.ndarray of shape [total_frames, 2] — normalized (cx, cy) per frame.
    """
    poi = np.full((total_frames, 2), FALLBACK_POI, dtype=np.float64)

    frame_target_boxes: Dict[int, List[Tuple[float, float, float]]] = {}
    frame_faces: Dict[int, List[Tuple[float, float, float]]] = {}
    frame_persons: Dict[int, List[Tuple[float, float, float]]] = {}

    for pe in spatial_events:
        if isinstance(pe, dict):
            ts = pe.get("timestamp", 0.0)
            bbox = pe.get("bbox", (0.5, 0.5, 1.0, 1.0))
            confidence = float(pe.get("confidence", 0.0))
            label = str(pe.get("label", "")).lower().strip()
        else:
            ts = getattr(pe, "timestamp", 0.0)
            bbox = getattr(pe, "bbox", (0.5, 0.5, 1.0, 1.0))
            confidence = float(getattr(pe, "confidence", 0.0))
            label = str(getattr(pe, "label", "")).lower().strip()

        if confidence < 0.30:
            continue

        frame_idx = min(max(0, int(ts * fps)), total_frames - 1)
        cx, cy, w, h = bbox

        if label in FACE_LABELS and confidence >= 0.45:
            frame_faces.setdefault(frame_idx, []).append((cx, cy, confidence))
        elif label in PERSON_LABELS and confidence >= 0.35:
            head_cy = max(0.08, cy - 0.32 * h)
            frame_persons.setdefault(frame_idx, []).append((cx, head_cy, confidence))

        target_cy = cy
        include = False
        weight = confidence

        if focal_target in ("SPEAKER_PRIMARY", "SPEAKER_REACTION"):
            # Handled by face/person priority check below
            pass
        elif focal_target == "FOCAL_DISPLAY":
            if label in FOCAL_DISPLAY_LABELS:
                include = True
                weight = confidence
        elif focal_target == "HELD_OBJECT":
            if label in HELD_OBJECT_LABELS:
                include = True
                weight = 0.90 * confidence
            elif label in PERSON_LABELS:
                include = True
                weight = 0.10 * confidence
        elif focal_target == "ACTION_SCENE":
            include = True
            weight = confidence
        else:
            tier_w, min_c = _classify_detection_tier(label)
            if confidence >= min_c:
                include = True
                weight = tier_w * confidence

        if include:
            frame_target_boxes.setdefault(frame_idx, []).append((cx, target_cy, weight))

    frame_detected_poi: Dict[int, np.ndarray] = {}

    for frame_idx in range(total_frames):
        if focal_target in ("SPEAKER_PRIMARY", "SPEAKER_REACTION"):
            # Strict Face-Centering Priority
            if frame_idx in frame_faces and len(frame_faces[frame_idx]) > 0:
                best_face = max(frame_faces[frame_idx], key=lambda x: x[2])
                frame_detected_poi[frame_idx] = np.array([best_face[0], best_face[1] - HEADROOM_BIAS_Y], dtype=np.float64)
            elif frame_idx in frame_persons and len(frame_persons[frame_idx]) > 0:
                best_person = max(frame_persons[frame_idx], key=lambda x: x[2])
                frame_detected_poi[frame_idx] = np.array([best_person[0], best_person[1] - HEADROOM_BIAS_Y], dtype=np.float64)
        else:
            # Semantic focal target priority (FOCAL_DISPLAY, HELD_OBJECT, ACTION_SCENE, etc.)
            if frame_idx in frame_target_boxes and len(frame_target_boxes[frame_idx]) > 0:
                boxes = frame_target_boxes[frame_idx]
                sum_w = sum(b[2] for b in boxes)
                if sum_w > 1e-8:
                    sum_cx = sum(b[0] * b[2] for b in boxes)
                    sum_cy = sum(b[1] * b[2] for b in boxes)
                    frame_detected_poi[frame_idx] = np.array([sum_cx / sum_w, (sum_cy / sum_w) - HEADROOM_BIAS_Y], dtype=np.float64)

    # Zero-Lag Initial State: Back-fill frames 0..first_detected
    first_detected = None
    for f in range(total_frames):
        if f in frame_detected_poi:
            first_detected = frame_detected_poi[f].copy()
            break

    if first_detected is None:
        first_detected = FALLBACK_POI.copy()

    current_val = first_detected.copy()
    for f in range(total_frames):
        if f in frame_detected_poi:
            current_val = frame_detected_poi[f].copy()
        poi[f] = current_val.copy()

    return poi


def _identify_shot_boundaries(scene_cuts: List[int], total_frames: int) -> List[Tuple[int, int]]:
    """Splits the frame range into shots based on scene cut frame indices.

    Returns:
        List of (start_frame, end_frame) tuples (end exclusive).
    """
    boundaries = sorted(set([0] + [c for c in scene_cuts if 0 < c < total_frames] + [total_frames]))
    shots = []
    for i in range(len(boundaries) - 1):
        shots.append((boundaries[i], boundaries[i + 1]))
    return shots


def _savgol_smooth(data: np.ndarray, window: int = SAVGOL_WINDOW,
                   polyorder: int = SAVGOL_POLYORDER) -> np.ndarray:
    """Applies Savitzky-Golay filter to a 1D array."""
    from scipy.signal import savgol_filter

    n = len(data)
    if n < window:
        w = max(polyorder + 2, n)
        if w % 2 == 0:
            w -= 1
        if w < polyorder + 2:
            return data.copy()
        return savgol_filter(data, window_length=w, polyorder=polyorder)
    return savgol_filter(data, window_length=window, polyorder=polyorder)


def _rts_kalman_smooth(data: np.ndarray) -> np.ndarray:
    """Two-pass Rauch-Tung-Striebel Kalman smoother with constant-velocity motion model.

    State: [position, velocity]
    """
    n = len(data)
    if n < 3:
        return data.copy()

    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[0.001, 0.0], [0.0, 0.0005]])
    R = np.array([[0.005]])

    # Forward pass
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))
    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))

    # Zero-Lag Initial State: Initialize directly at first frame target with tight covariance
    x_filt[0] = [data[0], 0.0]
    P_filt[0] = np.eye(2) * 0.01

    for k in range(1, n):
        x_pred[k] = F @ x_filt[k - 1]
        P_pred[k] = F @ P_filt[k - 1] @ F.T + Q

        y = data[k] - H @ x_pred[k]
        S = H @ P_pred[k] @ H.T + R
        K = P_pred[k] @ H.T @ np.linalg.inv(S)
        x_filt[k] = x_pred[k] + K.flatten() * y.item()
        P_filt[k] = (np.eye(2) - K @ H) @ P_pred[k]

    # Backward (RTS) pass
    x_smooth = np.zeros((n, 2))
    x_smooth[-1] = x_filt[-1]

    for k in range(n - 2, -1, -1):
        P_pred_inv = np.linalg.inv(P_pred[k + 1] + np.eye(2) * 1e-10)
        G = P_filt[k] @ F.T @ P_pred_inv
        x_smooth[k] = x_filt[k] + G @ (x_smooth[k + 1] - x_pred[k + 1])

    return x_smooth[:, 0]


def smooth_trajectory(raw_poi: np.ndarray,
                      scene_cuts: List[int],
                      crop_ratio: float = 9.0 / 16.0,
                      total_frames: Optional[int] = None,
                      lock_cy: bool = True) -> np.ndarray:
    """Smoothes raw POI coordinates per shot using Savitzky-Golay or RTS Kalman.

    Enforces:
        - Instantaneous teleport across scene cut boundaries (no smoothing across shots).
        - Fixed 9:16 crop width window clamping: cx in [cw/2, 1 - cw/2].
        - 1D horizontal velocity clamping: <= 0.008 * W_src per tick within shots.
        - Full-height lock: cy = 0.5 when lock_cy=True, or smoothed cy when lock_cy=False.

    Args:
        raw_poi: np.ndarray of shape [T, 2] with raw (cx, cy) per frame.
        scene_cuts: Frame indices where scene transitions occur.
        total_frames: Total frame count (defaults to len(raw_poi)).
        lock_cy: When True, locks cy = 0.5 (full frame height). When False, smooths cy for 2D framing.

    Returns:
        np.ndarray [T, 2] — globally smoothed crop center path.
    """
    if total_frames is None:
        total_frames = len(raw_poi)

    smoothed = np.zeros((total_frames, 2), dtype=np.float64)
    if lock_cy:
        # Strict Full-Height Lock: cy is always locked to 0.5 (full frame height)
        smoothed[:, 1] = 0.5
    else:
        smoothed[:, 1] = raw_poi[:, 1].copy()

    shots = _identify_shot_boundaries(scene_cuts, total_frames)
    scene_cut_set = set(scene_cuts)
    for shot_start, _ in shots:
        scene_cut_set.add(shot_start)

    for shot_start, shot_end in shots:
        shot_len = shot_end - shot_start
        if shot_len < 2:
            smoothed[shot_start:shot_end, 0] = raw_poi[shot_start:shot_end, 0]
            if not lock_cy:
                smoothed[shot_start:shot_end, 1] = raw_poi[shot_start:shot_end, 1]
            continue

        shot_cx = raw_poi[shot_start:shot_end, 0]

        if shot_len < 300:
            smoothed[shot_start:shot_end, 0] = _savgol_smooth(shot_cx)
            if not lock_cy:
                smoothed[shot_start:shot_end, 1] = _savgol_smooth(raw_poi[shot_start:shot_end, 1])
        else:
            smoothed[shot_start:shot_end, 0] = _rts_kalman_smooth(shot_cx)
            if not lock_cy:
                smoothed[shot_start:shot_end, 1] = _rts_kalman_smooth(raw_poi[shot_start:shot_end, 1])

        # Force teleport at the first frame of the shot
        smoothed[shot_start, 0] = raw_poi[shot_start, 0]
        if not lock_cy:
            smoothed[shot_start, 1] = raw_poi[shot_start, 1]

    # Fixed 9:16 Width Window (cw = 0.31640625 for 16:9 source)
    cw = 0.31640625
    margin_x = cw / 2.0

    smoothed[:, 0] = np.clip(smoothed[:, 0], margin_x, 1.0 - margin_x)
    if not lock_cy:
        smoothed[:, 1] = np.clip(smoothed[:, 1], 0.20, 0.80)

    # 1D Horizontal Pan Velocity Clamping (<= 0.008 * W_src per tick)
    for i in range(1, total_frames):
        if i in scene_cut_set:
            continue  # Instantaneous cut teleport: do NOT velocity clamp across shot boundaries

        dx = smoothed[i, 0] - smoothed[i - 1, 0]
        if abs(dx) > MAX_PAN_PER_TICK:
            smoothed[i, 0] = smoothed[i - 1, 0] + np.sign(dx) * MAX_PAN_PER_TICK

    return smoothed


def compute_crop_path(spatial_events: List[Any],
                      scene_changes: List[Any],
                      fps: float = 30.0,
                      total_frames: int = 0,
                      crop_ratio: float = 9.0 / 16.0,
                      focal_targets: Optional[List[Tuple[int, int, str]]] = None,
                      lock_cy: bool = True) -> np.ndarray:
    """Top-level API: computes the full pre-rendered crop path for a clip.

    Args:
        spatial_events: List of SpatialEvent dicts/objects from the vision spoke.
        scene_changes: List of SceneChange dicts/objects.
        fps: Target rendering FPS.
        total_frames: Total number of frames to generate crop path for.
        crop_ratio: Width/height crop ratio (default 9:16).
        focal_targets: Optional list of (start_frame, end_frame, focal_target) tuples.
            When provided, POI extraction uses target-specific detection filtering
            per frame range. When None, uses the default hierarchical fusion.

    Returns:
        np.ndarray [total_frames, 2] — normalized (cx, cy) crop center per frame.
    """
    scene_cut_frames = []
    for sc in scene_changes:
        if isinstance(sc, dict):
            fi = sc.get("frame_idx", int(sc.get("timestamp", 0.0) * fps))
        else:
            fi = getattr(sc, "frame_idx", int(getattr(sc, "timestamp", 0.0) * fps))
        if 0 < fi < total_frames:
            scene_cut_frames.append(fi)

    if focal_targets:
        # Build per-frame POI using focal-target-specific extraction
        raw_poi = np.full((total_frames, 2), FALLBACK_POI, dtype=np.float64)

        for ft_start, ft_end, focal_target in focal_targets:
            ft_start = max(0, ft_start)
            ft_end = min(total_frames, ft_end)
            segment_len = ft_end - ft_start
            if segment_len <= 0:
                continue

            segment_poi = _extract_poi_for_focal_target(
                spatial_events, fps, total_frames, focal_target
            )
            # Copy only the relevant frame range
            raw_poi[ft_start:ft_end] = segment_poi[ft_start:ft_end]

        logger.info(f"Focal-target-aware POI extraction: {len(focal_targets)} segments")
    else:
        raw_poi = _extract_poi_from_spatial_events(spatial_events, fps, total_frames)

    smoothed = smooth_trajectory(raw_poi, scene_cut_frames, total_frames=total_frames, lock_cy=lock_cy)

    logger.info(f"Computed global crop path: {total_frames} frames, "
                f"{len(scene_cut_frames)} scene cuts, "
                f"cx range [{smoothed[:, 0].min():.3f}, {smoothed[:, 0].max():.3f}], "
                f"cy range [{smoothed[:, 1].min():.3f}, {smoothed[:, 1].max():.3f}]")

    return smoothed
