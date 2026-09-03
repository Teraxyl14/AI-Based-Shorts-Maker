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

# Velocity clamp: max pan per tick as fraction of frame width
MAX_PAN_PER_TICK = 0.008

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
    """Extracts a per-frame POI array using hierarchical weighted centroid fusion.

    Weight hierarchy:
        - Primary Face Detection: w=0.65 (conf ≥ 0.40)
        - Upper-Torso / Shoulders: w=0.25 (conf ≥ 0.30)
        - Salient Moving Objects:  w=0.10 (conf ≥ 0.30)

    Returns:
        np.ndarray of shape [total_frames, 2] — normalized (cx, cy) per frame.
    """
    poi = np.full((total_frames, 2), FALLBACK_POI, dtype=np.float64)

    # Group spatial events by nearest frame index
    # Each entry: (cx, cy, combined_weight) where combined_weight = tier_weight * confidence
    frame_boxes: Dict[int, List[Tuple[float, float, float]]] = {}

    for pe in spatial_events:
        if isinstance(pe, dict):
            ts = pe.get("timestamp", 0.0)
            bbox = pe.get("bbox", (0.5, 0.5, 1.0, 1.0))
            confidence = float(pe.get("confidence", 0.0))
            label = str(pe.get("label", ""))
        else:
            ts = getattr(pe, "timestamp", 0.0)
            bbox = getattr(pe, "bbox", (0.5, 0.5, 1.0, 1.0))
            confidence = float(getattr(pe, "confidence", 0.0))
            label = str(getattr(pe, "label", ""))

        tier_weight, min_conf = _classify_detection_tier(label)

        if confidence >= min_conf:
            frame_idx = min(int(ts * fps), total_frames - 1)
            if frame_idx < 0:
                frame_idx = 0

            cx, cy, w, h = bbox
            combined_weight = tier_weight * confidence

            if frame_idx not in frame_boxes:
                frame_boxes[frame_idx] = []
            frame_boxes[frame_idx].append((cx, cy, combined_weight))

    # Compute weighted centroid for frames that have detections
    for frame_idx, boxes in frame_boxes.items():
        sum_cx = sum(b[0] * b[2] for b in boxes)
        sum_cy = sum(b[1] * b[2] for b in boxes)
        sum_w = sum(b[2] for b in boxes)

        if sum_w > 1e-8:
            poi[frame_idx, 0] = sum_cx / sum_w
            # Apply 4% upward headroom bias
            poi[frame_idx, 1] = (sum_cy / sum_w) - HEADROOM_BIAS_Y

    # Forward-fill gaps (frames without detections inherit from last known)
    last_valid = FALLBACK_POI.copy()
    for i in range(total_frames):
        if i in frame_boxes and sum(b[2] for b in frame_boxes[i]) > 1e-8:
            last_valid = poi[i].copy()
        else:
            poi[i] = last_valid

    return poi


def _extract_poi_for_focal_target(spatial_events: List[Any],
                                  fps: float,
                                  total_frames: int,
                                  focal_target: str) -> np.ndarray:
    """Extracts per-frame POI filtered by focal_target semantics.

    Focal target routing:
        - SPEAKER_PRIMARY / SPEAKER_REACTION: Face/person detections (hierarchical weights)
        - MONITOR_SCREEN: Monitor/laptop/TV/screen/display/cell phone detections only
        - HELD_OBJECT: Small object + hands detections
        - ACTION_SCENE: All detections equally weighted

    Falls back to upper-center (0.50, 0.45) when no matching detections exist.

    Returns:
        np.ndarray of shape [total_frames, 2] — normalized (cx, cy) per frame.
    """
    poi = np.full((total_frames, 2), FALLBACK_POI, dtype=np.float64)
    frame_boxes: Dict[int, List[Tuple[float, float, float]]] = {}

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

        # Filter detections based on focal target
        include = False
        weight = 1.0

        target_cy = cy
        if focal_target in ("SPEAKER_PRIMARY", "SPEAKER_REACTION"):
            # Only face and person detections
            if label in FACE_LABELS:
                include = True
                weight = 0.85 * confidence
            elif label in PERSON_LABELS:
                include = True
                weight = 0.35 * confidence
                # Head/face is located in the upper portion of the person bounding box
                target_cy = max(0.08, cy - 0.32 * h)
        elif focal_target == "FOCAL_DISPLAY":
            if label in FOCAL_DISPLAY_LABELS:
                include = True
                weight = confidence
        elif focal_target == "HELD_OBJECT":
            if label in HELD_OBJECT_LABELS or label in FACE_LABELS or label in PERSON_LABELS:
                # Include hands/person for context, but weight objects higher
                if label in HELD_OBJECT_LABELS:
                    include = True
                    weight = 0.80 * confidence
                elif label in PERSON_LABELS:
                    include = True
                    weight = 0.20 * confidence
                # Exclude face for held object — camera should be on hands/item
        elif focal_target == "ACTION_SCENE":
            # All detections with equal weight
            include = True
            weight = confidence
        else:
            # Unknown focal target — fall back to hierarchical
            tier_weight, min_conf = _classify_detection_tier(label)
            if confidence >= min_conf:
                include = True
                weight = tier_weight * confidence

        if include:
            if frame_idx not in frame_boxes:
                frame_boxes[frame_idx] = []
            frame_boxes[frame_idx].append((cx, target_cy, weight))

    # Weighted centroid per frame
    for frame_idx, boxes in frame_boxes.items():
        sum_cx = sum(b[0] * b[2] for b in boxes)
        sum_cy = sum(b[1] * b[2] for b in boxes)
        sum_w = sum(b[2] for b in boxes)

        if sum_w > 1e-8:
            poi[frame_idx, 0] = sum_cx / sum_w
            poi[frame_idx, 1] = (sum_cy / sum_w) - HEADROOM_BIAS_Y

    # Forward-fill gaps
    last_valid = FALLBACK_POI.copy()
    for i in range(total_frames):
        if i in frame_boxes and sum(b[2] for b in frame_boxes[i]) > 1e-8:
            last_valid = poi[i].copy()
        else:
            poi[i] = last_valid

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

    x_filt[0] = [data[0], 0.0]
    P_filt[0] = np.eye(2) * 0.1

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
        - Teleport (instant jump) across scene cut boundaries.
        - Fixed 9:16 crop width window clamping: cx in [cw/2, 1 - cw/2].
        - 1D horizontal velocity clamping: <= 0.008 * W_src per tick.
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
    scene_cut_set = set(scene_cuts)
    for i in range(1, total_frames):
        if i in scene_cut_set:
            continue

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
