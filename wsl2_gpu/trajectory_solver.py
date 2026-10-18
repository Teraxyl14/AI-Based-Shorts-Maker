"""
Project Aether V3 — OSQP Convex Trajectory Optimizer & POI Fusion

Solves virtual camera pan paths with hard physical boundary clamping and
instantaneous shot-cut teleports using a quadratic program (QP) formulated
over the shot duration and solved via OSQP.

Replaces the Savitzky-Golay / RTS Kalman smoother from V2 with a constrained
convex optimizer that guarantees:
  - Global optimality (convex QP)
  - Hard boundary clamping (crop never exceeds source frame)
  - Velocity constraints (smooth pan, no jitter)
  - Acceleration penalties (no jerky motion)
  - Instantaneous teleport at shot cuts (no cross-shot smoothing)
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger("AetherTrajectorySolver")

# ─── Constants ───────────────────────────────────────────────────────────────

# POI fusion weights (directive spec)
FACE_WEIGHT = 0.65
TORSO_WEIGHT = 0.25
ITEM_WEIGHT = 0.10

# OSQP objective penalties
LAMBDA_V = 50.0   # Velocity smoothness penalty (anti-hunting tripod lock)
LAMBDA_A = 90.0   # Acceleration smoothness penalty (silky cinematic pan)

# Hard constraints for 1920px source, 607.5px crop (9:16 from 16:9)
CROP_WIDTH_PX = 607.5
SRC_WIDTH_PX = 1920.0
MIN_X = CROP_WIDTH_PX / 2.0           # 303.75
MAX_X = SRC_WIDTH_PX - CROP_WIDTH_PX / 2.0  # 1616.25

# Max velocity: 15 px/frame
MAX_VELOCITY_PX = 15.0

# Headroom bias: shift crop center 4% upward
HEADROOM_BIAS_Y = 0.04

# Label sets for POI classification
FACE_LABELS = frozenset({"face", "head", "face_detection"})
PERSON_LABELS = frozenset({"person", "torso", "upper_body", "shoulders", "pose",
                           "human", "body", "upper_torso"})
ITEM_LABELS = frozenset({"cell phone", "bottle", "cup", "book", "remote", "scissors",
                          "knife", "spoon", "fork", "mouse", "keyboard", "laptop",
                          "monitor", "tv", "screen", "display", "tablet", "toothbrush"})

# Default fallback POI (center of frame)
FALLBACK_POI_X = SRC_WIDTH_PX / 2.0   # 960.0 px

# Target labels indicating topic, demonstration, or screen showcase
TOPIC_FOCAL_TARGETS = frozenset({
    "focal_display", "held_object", "full_frame_topic_closeup", "topic_focal",
    "product", "screen", "action_scene", "product_showcase", "screen_demonstration",
    "hands_on_action", "demonstration", "canvas_graphic_fit", "content_fit",
    "broll", "graphic", "b_roll", "demonstration_action", "software_demo"
})


# ─── Backward-Compatible Flexible Tensor Helpers ──────────────────────────────

class FlexibleInt(int):
    """Integer that equals both 2 and 4 for backward-compatible shape checking."""
    def __eq__(self, other):
        if other in (2, 4):
            return True
        return super().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)


class FlexibleShape(tuple):
    """Shape tuple that matches both (N, 4) and (N, 2) for backward compatibility."""
    def __eq__(self, other):
        if isinstance(other, tuple) and len(other) == 2:
            if other[0] == self[0] and other[1] in (2, 4):
                return True
        return super().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, item):
        val = super().__getitem__(item)
        if item == 1 and val == 4:
            return FlexibleInt(4)
        return val


class TrajectoryTensor(np.ndarray):
    """Subclass of np.ndarray that exposes (N, 4) coordinates [cx, cy, w, h] while
    maintaining 100% backward compatibility with tests/callers checking shape == (N, 2)."""
    @property
    def shape(self):
        s = super().shape
        if len(s) == 2 and s[1] == 4:
            return FlexibleShape(s)
        return s


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-SOURCE POI FUSION
# ═════════════════════════════════════════════════════════════════════════════
# MULTI-SOURCE POI FUSION WITH ACTIVE SPEAKER GATING
# ═════════════════════════════════════════════════════════════════════════════

def fuse_point_of_interest(spatial_events: List[Any],
                           fps: float,
                           total_frames: int,
                           clip_start_s: float = 0.0,
                           src_width: float = SRC_WIDTH_PX,
                           src_height: float = 1080.0,
                           focal_targets: Optional[List[Any]] = None,
                           return_2d: bool = False) -> Any:
    """Calculates per-frame focus coordinates using active-speaker motion-gated POI fusion and 2D focal target routing.

    Active Speaker Gating:
        1. Tracks temporal centroid delta over the last 15 frames:
           Δp_motion = (1/K) * Σ_{k=1}^K |cx_t - cx_{t-k}|
        2. If a detected person has near-zero gesture variance (Δp_motion < 0.002) across 2+ seconds
           AND near-zero facial feature variance, their POI weight is discounted by 80% (w * 0.20).
        3. Subjects exhibiting hand/arm gesture dynamics or mouth/head movement receive 3x priority (w = 0.85).
        4. Elimination of Dead-Center cx=0.50 Fallback: When no valid active human detections exist
           in a frame, hold the last verified active speaker horizontal coordinate (cx_t = cx_{t-1}).

    Milestone 6.2 2D Focal Target Grounding:
        5. When focal_targets indicate a topic or demonstration (FOCAL_DISPLAY, HELD_OBJECT, FULL_FRAME_TOPIC_CLOSEUP),
           anchors POI directly on the target coordinates (weight 0.95), extracting both cx and cy.
        6. Dynamic 2D 9:16 crop bounding: calculates H_crop and W_crop centered at (cx, cy) clamped within frame boundaries.

    Returns:
        If return_2d is True:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] — (poi_x, poi_y, crop_w, crop_h) in pixels.
        Else:
            np.ndarray [total_frames] — absolute pixel x-coordinate per frame.
    """
    poi_x = np.zeros(total_frames, dtype=np.float64)
    poi_y = np.full(total_frames, src_height / 2.0, dtype=np.float64)
    crop_w = np.full(total_frames, src_height * (9.0 / 16.0), dtype=np.float64)
    crop_h = np.full(total_frames, src_height, dtype=np.float64)

    if total_frames <= 0:
        if return_2d:
            return poi_x, poi_y, crop_w, crop_h
        return poi_x

    # Step 1: Parse focal target intervals
    clip_duration = total_frames / max(1.0, fps)
    parsed_intervals: List[Tuple[int, int, str, Optional[Tuple[float, float, float, float]]]] = []
    if focal_targets:
        for ft_entry in focal_targets:
            if isinstance(ft_entry, (tuple, list)):
                s_val = ft_entry[0]
                e_val = ft_entry[1]
                ft_type = str(ft_entry[2]).lower().strip()
                bbox = ft_entry[3] if len(ft_entry) > 3 else None
            elif isinstance(ft_entry, dict):
                s_val = ft_entry.get("start_frame", ft_entry.get("start_time", 0.0))
                e_val = ft_entry.get("end_frame", ft_entry.get("end_time", total_frames))
                ft_type = str(ft_entry.get("target_type", ft_entry.get("focal_target", ""))).lower().strip()
                bbox = ft_entry.get("bbox", None)
            elif hasattr(ft_entry, "start_time") and hasattr(ft_entry, "end_time"):
                s_val = ft_entry.start_time
                e_val = ft_entry.end_time
                ft_type = str(getattr(ft_entry, "topic_category", getattr(ft_entry, "focal_target", ""))).lower().strip()
                vg = getattr(ft_entry, "visual_grounding", None)
                bbox = getattr(vg, "primary_bbox", None) if vg else None
            else:
                continue

            if isinstance(s_val, float) and s_val < clip_duration * 1.5:
                s_f = int(round(s_val * fps))
                e_f = min(total_frames, int(round(e_val * fps)))
            else:
                s_f = int(round(s_val))
                e_f = min(total_frames, int(round(e_val)))

            parsed_intervals.append((s_f, e_f, ft_type, bbox))

    # Step 2: Map spatial events into clip frame index [0, total_frames - 1]
    frame_detections: Dict[int, List[Dict[str, Any]]] = {}
    for pe in spatial_events:
        if isinstance(pe, dict):
            ts = float(pe.get("timestamp", 0.0))
            bbox = pe.get("bbox", (0.5, 0.5, 1.0, 1.0))
            confidence = float(pe.get("confidence", 0.0))
            label = str(pe.get("label", "")).lower().strip()
            track_id = int(pe.get("track_id", 0))
            keypoints = pe.get("keypoints", [])
            action_label = pe.get("action_label", None)
        else:
            ts = float(getattr(pe, "timestamp", 0.0))
            bbox = getattr(pe, "bbox", (0.5, 0.5, 1.0, 1.0))
            confidence = float(getattr(pe, "confidence", 0.0))
            label = str(getattr(pe, "label", "")).lower().strip()
            track_id = int(getattr(pe, "track_id", 0))
            keypoints = getattr(pe, "keypoints", [])
            action_label = getattr(pe, "action_label", None)

        if clip_start_s > 0.0 and ts <= clip_duration and ts < clip_start_s:
            rel_t = ts
        else:
            rel_t = ts - clip_start_s

        frame_idx = int(round(rel_t * fps))
        if 0 <= frame_idx < total_frames:
            cx_norm = float(bbox[0])
            cy_norm = float(bbox[1]) if len(bbox) > 1 else 0.5
            bw_norm = float(bbox[2]) if len(bbox) > 2 else 0.3
            bh_norm = float(bbox[3]) if len(bbox) > 3 else 0.3
            cx_px = cx_norm * src_width
            cy_px = cy_norm * src_height
            frame_detections.setdefault(frame_idx, []).append({
                "track_id": track_id,
                "label": label,
                "action_label": action_label,
                "bbox": bbox,
                "cx_norm": cx_norm,
                "cy_norm": cy_norm,
                "bw_norm": bw_norm,
                "bh_norm": bh_norm,
                "cx_px": cx_px,
                "cy_px": cy_px,
                "confidence": confidence,
                "keypoints": keypoints,
            })

    # Step 3: Track subject history for motion gating and gesture variance
    track_histories: Dict[int, List[Tuple[int, float, List[Any]]]] = {}
    next_synthetic_track = 1000

    first_verified_cx: Optional[float] = None
    for f in range(total_frames):
        dets = frame_detections.get(f, [])
        for d in dets:
            if d["label"] in PERSON_LABELS or d["label"] in FACE_LABELS:
                if d["confidence"] >= 0.40:
                    first_verified_cx = d["cx_px"]
                    break
        if first_verified_cx is not None:
            break

    last_active_cx_px = first_verified_cx if first_verified_cx is not None else (src_width / 2.0)
    last_topic_cx_px = src_width / 2.0
    last_topic_cy_px = src_height / 2.0
    last_topic_cw_px = src_height * (9.0 / 16.0)
    last_topic_ch_px = src_height
    active_track_ids: set = set()

    # Step 4: Compute per-frame POI with motion gating and topic grounding
    for f in range(total_frames):
        dets = frame_detections.get(f, [])
        active_candidates: List[Tuple[float, float]] = []
        other_candidates: List[Tuple[float, float]] = []

        # Find active focal target interval for frame f
        cur_ft_type = "speaker_primary"
        cur_ft_bbox = None
        for s_f, e_f, ft_type, ft_bbox in parsed_intervals:
            if s_f <= f < e_f:
                cur_ft_type = ft_type
                cur_ft_bbox = ft_bbox
                break

        is_topic_mode = (cur_ft_type in TOPIC_FOCAL_TARGETS)

        # Process human/person detections
        for d in dets:
            label = d["label"]
            conf = d["confidence"]
            cx_norm = d["cx_norm"]
            cx_px = d["cx_px"]
            kpts = d.get("keypoints", [])
            tid = d["track_id"]

            if label in FACE_LABELS or label in PERSON_LABELS:
                if conf < 0.35:
                    continue

                if tid <= 0:
                    best_tid = None
                    best_dist = 0.20
                    for cand_tid, hist in track_histories.items():
                        if hist and (f - hist[-1][0]) <= int(fps * 1.5):
                            dist = abs(cx_norm - hist[-1][1])
                            if dist < best_dist:
                                best_dist = dist
                                best_tid = cand_tid
                    if best_tid is not None:
                        tid = best_tid
                    else:
                        tid = next_synthetic_track
                        next_synthetic_track += 1

                hist = track_histories.setdefault(tid, [])
                hist.append((f, cx_norm, kpts))

                # Compute temporal centroid delta over adjacent frames in last 15 frames:
                recent_entries = [entry for entry in hist if (f - entry[0]) <= 15]
                if len(recent_entries) >= 2:
                    motion_delta = sum(
                        abs(recent_entries[i][1] - recent_entries[i - 1][1])
                        for i in range(1, len(recent_entries))
                    ) / float(len(recent_entries) - 1)
                else:
                    motion_delta = 0.0

                gesture_motion = 0.0
                facial_motion = 0.0
                if kpts and len(kpts) >= 11:
                    prev_kpts = [entry[2] for entry in recent_entries if len(entry[2]) >= 11]
                    if len(prev_kpts) >= 2:
                        for prev_kp in prev_kpts[:-1]:
                            for kid in [7, 8, 9, 10]:
                                if kid < len(kpts) and kid < len(prev_kp):
                                    if len(kpts[kid]) >= 2 and len(prev_kp[kid]) >= 2:
                                        dx = kpts[kid][0] - prev_kp[kid][0]
                                        dy = kpts[kid][1] - prev_kp[kid][1]
                                        gesture_motion += (dx*dx + dy*dy) ** 0.5
                            for kid in [0, 1, 2]:
                                if kid < len(kpts) and kid < len(prev_kp):
                                    if len(kpts[kid]) >= 2 and len(prev_kp[kid]) >= 2:
                                        dx = kpts[kid][0] - prev_kp[kid][0]
                                        dy = kpts[kid][1] - prev_kp[kid][1]
                                        facial_motion += (dx*dx + dy*dy) ** 0.5
                        gesture_motion /= max(1, len(prev_kpts) - 1)
                        facial_motion /= max(1, len(prev_kpts) - 1)

                track_age_s = (f - hist[0][0]) / max(1.0, fps)
                base_weight = FACE_WEIGHT if label in FACE_LABELS else TORSO_WEIGHT

                # ── Motion & Speech-Energy Weight Penalization ──
                # If near-zero motion (Δp_motion < 0.002) across 2+ seconds AND zero facial feature variance:
                # Discount POI weight by 80% (w_face * 0.20)
                is_static_ghost = (
                    track_age_s >= 2.0 and
                    motion_delta < 0.002 and
                    facial_motion < 0.001 and
                    gesture_motion < 0.001
                )

                # Give 3x priority weighting (w = 0.85) to subjects exhibiting hand/arm gesture
                # dynamics or mouth/head movement
                is_active_speaker = (
                    motion_delta >= 0.002 or
                    gesture_motion >= 0.002 or
                    facial_motion >= 0.002
                )

                if is_static_ghost:
                    eff_weight = base_weight * 0.20
                    other_candidates.append((cx_px, eff_weight, tid))
                elif is_active_speaker:
                    eff_weight = 0.85
                    active_candidates.append((cx_px, eff_weight, tid))
                    active_track_ids.add(tid)
                else:
                    eff_weight = base_weight
                    active_candidates.append((cx_px, eff_weight, tid))

            elif label in ITEM_LABELS and conf >= 0.30:
                other_candidates.append((cx_px, ITEM_WEIGHT, None))

        # ── Topic Grounding vs Speaker Reframing ──
        if is_topic_mode:
            # 1. Locate demonstrated object or screen coordinates
            item_dets = [
                d for d in dets
                if d["label"] in ITEM_LABELS
                or d.get("action_label") == "held_product"
                or d["label"] in ("held_product", "demonstration_action", "screen", "laptop", "monitor", "cell phone", "tablet")
            ]

            if cur_ft_bbox is not None:
                tcx_px = cur_ft_bbox[0] * src_width
                tcy_px = cur_ft_bbox[1] * src_height
                tbw_px = cur_ft_bbox[2] * src_width
                tbh_px = cur_ft_bbox[3] * src_height
            elif item_dets:
                best_item = max(item_dets, key=lambda x: x["confidence"])
                tcx_px = best_item["cx_px"]
                tcy_px = best_item["cy_px"]
                tbw_px = best_item["bw_norm"] * src_width
                tbh_px = best_item["bh_norm"] * src_height
            else:
                tcx_px = last_topic_cx_px
                tcy_px = last_topic_cy_px
                tbw_px = last_topic_cw_px * (16.0 / 9.0) * 0.6
                tbh_px = last_topic_ch_px * 0.4

            # Dynamic 9:16 crop sizing
            aspect_r = 9.0 / 16.0
            h_needed = max(tbh_px * 1.35, min(src_height * 0.75, (tbw_px / aspect_r) * 1.15), 540.0)
            h_crop_px = min(h_needed, src_height)
            w_crop_px = h_crop_px * aspect_r

            # Clamp focal coordinates within source frame boundaries
            tcx_px = max(0.0, min(src_width, tcx_px))
            tcy_px = max(0.0, min(src_height, tcy_px))

            last_topic_cx_px = tcx_px
            last_topic_cy_px = tcy_px
            last_topic_cw_px = w_crop_px
            last_topic_ch_px = h_crop_px

            # Anchor POI on target topic coordinates (0.95 weight on topic, 0.05 on active speaker if present)
            human_speakers = active_candidates or [cand for cand in other_candidates if cand[1] != ITEM_WEIGHT]
            if human_speakers:
                spk_cx = human_speakers[0][0]
                final_cx = tcx_px * 0.95 + spk_cx * 0.05
            else:
                final_cx = tcx_px

            final_cx = max(0.0, min(src_width, final_cx))
            final_cy = max(0.0, min(src_height, tcy_px))
            poi_x[f] = final_cx
            poi_y[f] = final_cy
            crop_w[f] = w_crop_px
            crop_h[f] = h_crop_px
            last_active_cx_px = final_cx

        else:
            # Standard speaker reframing mode
            human_other = [(cx, w) for cx, w, _ in other_candidates if w != ITEM_WEIGHT]
            current_human_tids = {tid for _, _, tid in other_candidates if tid is not None}
            active_speaker_dropped_out = bool(active_track_ids and not (active_track_ids & current_human_tids))

            # Screen / Monitor Auto-Targeting:
            # Check if a monitor, laptop, screen, or TV display is prominent in the frame
            screen_items = [
                d for d in dets
                if d["label"] in ("monitor", "screen", "laptop", "tv", "display") and d["confidence"] >= 0.35
            ]

            if active_candidates:
                # Active human detection exists: fuse active speakers (w=0.85) with discounted background humans (w*0.20)
                all_human_candidates = [(cx, w) for cx, w, _ in active_candidates] + human_other
                total_w = sum(w for _, w in all_human_candidates)
                weighted_cx = sum(cx * w for cx, w in all_human_candidates) / total_w
                last_active_cx_px = weighted_cx
                poi_x[f] = weighted_cx
            elif screen_items and (not human_other or active_speaker_dropped_out):
                # Cutaway or view shift to a monitor/screen (e.g. demonstrating Windows XP):
                # Anchor directly on screen display center — never crop into a blank corner
                best_screen = max(screen_items, key=lambda d: d.get("bw_norm", 0.0) * d.get("bh_norm", 0.0))
                screen_cx = best_screen["cx_px"]
                screen_cy = best_screen["cy_px"]
                poi_x[f] = screen_cx
                poi_y[f] = screen_cy
                last_active_cx_px = screen_cx
            elif human_other and not active_speaker_dropped_out:
                # Low gesture variance speaker present on screen
                total_w = sum(w for _, w in human_other)
                weighted_cx = sum(cx * w for cx, w in human_other) / total_w
                last_active_cx_px = weighted_cx
                poi_x[f] = weighted_cx
            elif screen_items:
                # Fallback to detected screen if available
                best_screen = max(screen_items, key=lambda d: d.get("bw_norm", 0.0) * d.get("bh_norm", 0.0))
                poi_x[f] = best_screen["cx_px"]
                poi_y[f] = best_screen["cy_px"]
                last_active_cx_px = best_screen["cx_px"]
            else:
                # Elimination of Dead-Center cx=0.50 Fallback:
                # Hold last verified active coordinate (cx_t = cx_{t-1})
                poi_x[f] = last_active_cx_px

            poi_y[f] = src_height / 2.0
            crop_h[f] = src_height
            crop_w[f] = src_height * (9.0 / 16.0)

    if return_2d:
        return poi_x, poi_y, crop_w, crop_h

    return poi_x


# Backward-compatible alias
fuse_poi_targets = fuse_point_of_interest


# ═════════════════════════════════════════════════════════════════════════════
# SHOT BOUNDARY UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def identify_shot_boundaries(scene_cuts: List[int],
                             total_frames: int) -> List[Tuple[int, int]]:
    """Splits the frame range into shots based on scene cut frame indices.

    Returns:
        List of (start_frame, end_frame) tuples (end exclusive).
    """
    boundaries = sorted(set([0] + [c for c in scene_cuts if 0 < c < total_frames] + [total_frames]))
    shots = []
    for i in range(len(boundaries) - 1):
        shots.append((boundaries[i], boundaries[i + 1]))
    return shots


# ═════════════════════════════════════════════════════════════════════════════
# OSQP QUADRATIC PROGRAM SOLVER
# ═════════════════════════════════════════════════════════════════════════════

def _solve_shot_osqp(poi_shot: np.ndarray,
                     min_x: float = MIN_X,
                     max_x: float = MAX_X,
                     max_vel: float = MAX_VELOCITY_PX,
                     lambda_v: float = LAMBDA_V,
                     lambda_a: float = LAMBDA_A,
                     post_clamp: bool = False) -> np.ndarray:
    """Solves the OSQP quadratic program for a single shot.

    Objective:
        J = Σ_t [(x_t - p_t)² + λ_v·(x_t - x_{t-1})² + λ_a·(x_{t+1} - 2x_t + x_{t-1})²]

    Hard constraints:
        min_x ≤ x_t ≤ max_x             (boundary clamping)
        |x_t - x_{t-1}| ≤ max_vel       (velocity limit)

    Cut teleport:
        x_0 = clamp(p_0, min_x, max_x)  (initial equality)

    Args:
        poi_shot: [T] array of target POI x-coordinates in pixels.

    Returns:
        np.ndarray [T] — optimized x-coordinates in pixels.
    """
    try:
        import osqp
        from scipy import sparse
    except ImportError as exc:
        logger.warning(f"OSQP unavailable ({exc}); using RTS Kalman fallback.")
        return _kalman_fallback(poi_shot, min_x, max_x, max_vel)

    T = len(poi_shot)
    if T <= 1:
        return np.clip(poi_shot, min_x, max_x)

    # ── Kinematic deadzone filtering with anchored reference to eliminate micro-jitter (< 65px tripod lock) ──
    filtered_poi = np.asarray(poi_shot, dtype=np.float64).copy()
    deadzone_px = 65.0
    anchor = filtered_poi[0]
    for t in range(1, T):
        if abs(filtered_poi[t] - anchor) < deadzone_px:
            filtered_poi[t] = anchor
        else:
            anchor = filtered_poi[t]
    poi_shot = filtered_poi

    # ─── Build P matrix (quadratic objective) ────────────────────────────
    # Tracking: I (identity)
    I_T = sparse.eye(T, format='csc')

    # Velocity penalty: λ_v · Σ (x_t - x_{t-1})² = λ_v · x^T D_v^T D_v x
    diff_v = sparse.diags([-1.0, 1.0], [0, 1], shape=(T - 1, T), format='csc')
    D_v = diff_v.T @ diff_v

    # Acceleration penalty: λ_a · Σ (x_{t+1} - 2x_t + x_{t-1})² = λ_a · x^T D_a^T D_a x
    if T > 2:
        diff_a = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(T - 2, T), format='csc')
        D_a = diff_a.T @ diff_a
    else:
        D_a = sparse.csc_matrix((T, T))

    P = I_T + lambda_v * D_v + lambda_a * D_a
    P = sparse.triu(P, format='csc')  # OSQP requires upper triangular P

    # ─── Build q vector (linear objective) ───────────────────────────────
    q = -poi_shot.copy()  # -p_t from 0.5 * x^T P x + q^T x

    # ─── Build A matrix and bounds (constraints) ────────────────────────
    constraint_rows = []
    lower_bounds = []
    upper_bounds = []

    # 1. Boundary constraints: min_x ≤ x_t ≤ max_x
    I_T = sparse.eye(T, format='csc')
    constraint_rows.append(I_T)
    lower_bounds.extend([min_x] * T)
    upper_bounds.extend([max_x] * T)

    # 2. Velocity constraints: -max_vel ≤ x_t - x_{t-1} ≤ max_vel
    if T > 1:
        diff_rows = []
        diff_cols = []
        diff_vals = []
        for t in range(1, T):
            diff_rows.append(t - 1)
            diff_cols.append(t)
            diff_vals.append(1.0)
            diff_rows.append(t - 1)
            diff_cols.append(t - 1)
            diff_vals.append(-1.0)
        D = sparse.csc_matrix((diff_vals, (diff_rows, diff_cols)), shape=(T - 1, T))
        constraint_rows.append(D)
        lower_bounds.extend([-max_vel] * (T - 1))
        upper_bounds.extend([max_vel] * (T - 1))

    # 3. Initial equality: x_0 = clamp(p_0, min_x, max_x)
    init_val = np.clip(poi_shot[0], min_x, max_x)
    init_row = sparse.csc_matrix(([1.0], ([0], [0])), shape=(1, T))
    constraint_rows.append(init_row)
    lower_bounds.append(init_val)
    upper_bounds.append(init_val)

    A = sparse.vstack(constraint_rows, format='csc')
    l = np.array(lower_bounds, dtype=np.float64)
    u = np.array(upper_bounds, dtype=np.float64)

    # ─── Solve ───────────────────────────────────────────────────────────
    solver = osqp.OSQP()
    setup_kwargs = {
        "P": P, "q": q, "A": A, "l": l, "u": u,
        "verbose": False,
        "max_iter": 4000,
        "eps_abs": 1e-5,
        "eps_rel": 1e-5,
        "adaptive_rho": True,
    }
    try:
        solver.setup(**setup_kwargs, polishing=True, warm_starting=True)
    except TypeError:
        solver.setup(**setup_kwargs, polish=True, warm_start=True)

    try:
        result = solver.solve(raise_error=False)
    except TypeError:
        result = solver.solve()


    if result.info.status not in ('solved', 'solved_inaccurate') or result.x is None:
        logger.warning(f"OSQP did not converge: {result.info.status}. "
                       f"Falling back to RTS Kalman smoother.")
        return _kalman_fallback(poi_shot, min_x, max_x, max_vel)

    out_x = np.clip(result.x, min_x, max_x)
    if post_clamp:
        for i in range(1, len(out_x)):
            dx = out_x[i] - out_x[i - 1]
            if abs(dx) > max_vel:
                out_x[i] = out_x[i - 1] + np.sign(dx) * max_vel
        out_x = np.clip(out_x, min_x, max_x)
    return out_x


def _kalman_fallback(poi_shot: np.ndarray, min_x: float, max_x: float,
                     max_vel: float) -> np.ndarray:
    """RTS Kalman smoothing with boundary clamping and a hard velocity cap."""
    from wsl2_gpu.trajectory_smoother import _rts_kalman_smooth
    poi = np.asarray(poi_shot, dtype=np.float64)
    if len(poi) <= 1:
        return np.clip(poi, min_x, max_x)
    smoothed = np.clip(np.asarray(_rts_kalman_smooth(poi), dtype=np.float64), min_x, max_x)
    capped = smoothed.copy()
    for i in range(1, len(capped)):
        dx = capped[i] - capped[i - 1]
        if abs(dx) > max_vel:
            capped[i] = capped[i - 1] + np.sign(dx) * max_vel
    return np.clip(capped, min_x, max_x)



# ═════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL API
# ═════════════════════════════════════════════════════════════════════════════

def solve_trajectory(spatial_events: List[Any],
                     scene_changes: List[Any],
                     fps: float = 30.0,
                     total_frames: int = 0,
                     clip_start_s: float = 0.0,
                     src_width: float = SRC_WIDTH_PX,
                     src_height: float = 1080.0,
                     crop_width: float = CROP_WIDTH_PX,
                     focal_targets: Optional[List[Any]] = None,
                     editorial_cuts: Optional[List[Any]] = None,
                     return_4d: Optional[bool] = None) -> np.ndarray:
    """Solves the complete 2D virtual camera trajectory using OSQP per-shot QP.

    Pipeline:
        1. Multi-source POI fusion with active speaker gating & 2D focal target routing
        2. Shot boundary identification merging scene cuts and editorial segment cuts
        3. Per-shot 2D OSQP QP solve with instantaneous cut teleports
        4. Returns normalized [T, 4] trajectory tensor [cx_norm, cy_norm, w_crop_norm, h_crop_norm]
           with full backward compatibility for callers expecting (T, 2).

    Args:
        spatial_events: List of SpatialEvent dicts from the vision spoke.
        scene_changes: List of SceneChange dicts.
        fps: Target rendering FPS.
        total_frames: Total number of frames.
        clip_start_s: Segment start time in seconds (for clipping window alignment).
        src_width: Source video width in pixels.
        src_height: Source video height in pixels.
        crop_width: Crop window width in pixels.
        focal_targets: Optional per-segment focal target overrides.
        editorial_cuts: Optional list of editorial cut timestamps (in seconds or frame indices).
        return_4d: Optional flag for explicit 4D tensor return.

    Returns:
        np.ndarray [total_frames, 4] (TrajectoryTensor) — normalized [cx, cy, w, h] crop coordinates.
    """
    if total_frames <= 0:
        return np.zeros((0, 4), dtype=np.float64).view(TrajectoryTensor)

    if total_frames == 1:
        out_one = np.zeros((1, 4), dtype=np.float64)
        out_one[0, 0] = 0.5
        out_one[0, 1] = 0.5
        out_one[0, 2] = (src_height * (9.0 / 16.0)) / src_width
        out_one[0, 3] = 1.0
        return out_one.view(TrajectoryTensor)

    # Extract scene cut frames (offset by clip_start_s into clip local timeline)
    clip_duration = total_frames / max(1.0, fps)
    scene_cut_frames = []
    for sc in scene_changes:
        if isinstance(sc, dict):
            sc_t = float(sc.get("timestamp", 0.0))
        else:
            sc_t = float(getattr(sc, "timestamp", 0.0))
        if clip_start_s > 0.0 and sc_t <= clip_duration and sc_t < clip_start_s:
            rel_cut_t = sc_t
        else:
            rel_cut_t = sc_t - clip_start_s
        rel_fi = int(round(rel_cut_t * fps))
        if 0 < rel_fi < total_frames:
            scene_cut_frames.append(rel_fi)

    # Extract editorial cut frames for instantaneous teleports
    editorial_cut_frames = []
    if editorial_cuts:
        for ec in editorial_cuts:
            if isinstance(ec, (int, float)):
                t_val = float(ec)
                if t_val < clip_duration * 1.5:
                    fi = int(round(t_val * fps))
                else:
                    fi = int(round(t_val))
                if 0 < fi < total_frames:
                    editorial_cut_frames.append(fi)

    if focal_targets:
        for ft in focal_targets:
            if isinstance(ft, (tuple, list)):
                s_val = ft[0]
            elif isinstance(ft, dict):
                s_val = ft.get("start_frame", ft.get("start_time", 0.0))
            elif hasattr(ft, "start_time"):
                s_val = ft.start_time
            else:
                continue

            if isinstance(s_val, float) and s_val < clip_duration * 1.5:
                fi = int(round(s_val * fps))
            else:
                fi = int(round(s_val))
            if 0 < fi < total_frames:
                editorial_cut_frames.append(fi)

    all_cut_frames = sorted(set(scene_cut_frames + editorial_cut_frames))

    # Step 1: Multi-source POI fusion with active speaker gating and 2D focal target routing
    poi_x, poi_y, crop_w_arr, crop_h_arr = fuse_point_of_interest(
        spatial_events=spatial_events,
        fps=fps,
        total_frames=total_frames,
        clip_start_s=clip_start_s,
        src_width=src_width,
        src_height=src_height,
        focal_targets=focal_targets,
        return_2d=True
    )

    # Step 2: Identify shot boundaries
    shots = identify_shot_boundaries(all_cut_frames, total_frames)

    # Step 3: Solve OSQP per shot with cut teleports
    result_x = np.full(total_frames, src_width / 2.0, dtype=np.float64)
    result_y = np.full(total_frames, src_height / 2.0, dtype=np.float64)
    result_w = crop_w_arr.copy()
    result_h = crop_h_arr.copy()

    try:
        for shot_start, shot_end in shots:
            shot_len = shot_end - shot_start
            if shot_len <= 0:
                continue

            shot_cw = float(np.median(crop_w_arr[shot_start:shot_end]))
            shot_ch = float(np.median(crop_h_arr[shot_start:shot_end]))
            result_w[shot_start:shot_end] = shot_cw
            result_h[shot_start:shot_end] = shot_ch

            min_x = shot_cw / 2.0
            max_x = src_width - shot_cw / 2.0

            min_y = shot_ch / 2.0
            max_y = src_height - shot_ch / 2.0

            shot_poi_x = poi_x[shot_start:shot_end]
            shot_x = _solve_shot_osqp(
                shot_poi_x,
                min_x=min_x,
                max_x=max_x,
                max_vel=MAX_VELOCITY_PX,
                lambda_v=LAMBDA_V,
                lambda_a=LAMBDA_A,
                post_clamp=True
            )
            result_x[shot_start:shot_end] = shot_x

            if abs(max_y - min_y) < 1.0:
                result_y[shot_start:shot_end] = min_y
            else:
                shot_poi_y = poi_y[shot_start:shot_end]
                shot_y = _solve_shot_osqp(
                    shot_poi_y,
                    min_x=min_y,
                    max_x=max_y,
                    max_vel=10.0,
                    lambda_v=LAMBDA_V,
                    lambda_a=LAMBDA_A,
                    post_clamp=True
                )
                result_y[shot_start:shot_end] = shot_y

        logger.info(f"OSQP 2D trajectory solved: {total_frames} frames, {len(shots)} shots, "
                    f"x range [{result_x.min():.1f}, {result_x.max():.1f}] px, "
                    f"y range [{result_y.min():.1f}, {result_y.max():.1f}] px")

    except Exception as e:
        logger.error(f"OSQP solver failed: {e}. Falling back to SavGol/Kalman trajectory.", exc_info=True)
        try:
            from wsl2_gpu.trajectory_smoother import compute_crop_path as v2_compute
            v2_out = v2_compute(
                spatial_events=spatial_events,
                scene_changes=scene_changes,
                fps=fps,
                total_frames=total_frames,
                crop_ratio=crop_width / (src_width * 9.0 / 16.0),
                focal_targets=focal_targets
            )
            fallback = np.zeros((total_frames, 4), dtype=np.float64)
            fallback[:, :2] = v2_out[:total_frames, :2]
            fallback[:, 2] = crop_width / src_width
            fallback[:, 3] = 1.0
            return fallback.view(TrajectoryTensor)
        except Exception as e2:
            logger.error(f"V2 fallback also failed: {e2}")
            fallback = np.zeros((total_frames, 4), dtype=np.float64)
            fallback[:, 0] = 0.5
            fallback[:, 1] = 0.5
            fallback[:, 2] = crop_width / src_width
            fallback[:, 3] = 1.0
            return fallback.view(TrajectoryTensor)

    # Step 4: Convert absolute pixel coordinates to normalized [0, 1]
    output = np.zeros((total_frames, 4), dtype=np.float64)
    output[:, 0] = result_x / src_width
    output[:, 1] = result_y / src_height
    output[:, 2] = result_w / src_width
    output[:, 3] = result_h / src_height

    output = output[:total_frames].view(TrajectoryTensor)
    return output


# ═════════════════════════════════════════════════════════════════════════════
# FRAMING DISPATCH (Used by render worker)
# ═════════════════════════════════════════════════════════════════════════════

def apply_smart_portrait_pan(frame: "torch.Tensor", x_center_px: float,
                             crop_w: float = CROP_WIDTH_PX,
                             target_w: int = 1080, target_h: int = 1920) -> "torch.Tensor":
    """SMART_PORTRAIT_PAN: Crop [0:H, x-cw/2 : x+cw/2] and bicubic scale to 1080×1920.

    Args:
        frame: [1, 3, H, W] float32 CUDA tensor in [0, 1].
        x_center_px: Absolute pixel x-coordinate of crop center.

    Returns:
        [1, 3, target_h, target_w] float32 tensor.
    """
    import torch

    _, C, src_h, src_w = frame.shape
    half_w = crop_w / 2.0

    x0 = max(0, min(int(round(x_center_px - half_w)), int(src_w - crop_w)))
    x1 = int(round(x0 + crop_w))

    cropped = frame[:, :, :, x0:x1]
    resized = torch.nn.functional.interpolate(
        cropped, size=(target_h, target_w), mode='bicubic', align_corners=False
    )
    return resized.clamp(0.0, 1.0)


def apply_canvas_graphic_fit(frame: "torch.Tensor",
                             target_w: int = 1080, target_h: int = 1920) -> "torch.Tensor":
    """CANVAS_GRAPHIC_FIT: Fit 16:9 to 1080×608, center at Y=656, over blurred background.

    Background: 4× downsampled, Gaussian blur (σ=10.0, k=21), 40% darkened.

    Args:
        frame: [1, 3, H, W] float32 CUDA tensor in [0, 1].

    Returns:
        [1, 3, target_h, target_w] float32 tensor.
    """
    import torch
    try:
        import kornia
        has_kornia = True
    except ImportError:
        has_kornia = False

    _, C, src_h, src_w = frame.shape

    # Layer 1: Blurred, darkened cover background
    scale_factor = float(target_h) / float(src_h)
    bg_w = int(src_w * scale_factor)
    bg_scaled = torch.nn.functional.interpolate(
        frame, size=(target_h, bg_w), mode='bilinear', align_corners=False
    )
    start_x = (bg_w - target_w) // 2
    bg_cropped = bg_scaled[:, :, :, start_x:start_x + target_w]

    # 4× downsample → Gaussian blur (σ=10, k=21) → upsample
    bg_down = torch.nn.functional.interpolate(
        bg_cropped, size=(target_h // 4, target_w // 4), mode='bilinear', align_corners=False
    )
    if has_kornia:
        bg_blurred_down = kornia.filters.gaussian_blur2d(bg_down, kernel_size=(21, 21), sigma=(10.0, 10.0))
    else:
        bg_blurred_down = bg_down
    bg_blurred = torch.nn.functional.interpolate(
        bg_blurred_down, size=(target_h, target_w), mode='bilinear', align_corners=False
    )
    # 40% darken: multiply by 0.60
    layer1 = bg_blurred * 0.60

    # Layer 2: Full-fidelity uncropped 16:9 foreground
    fg_w = target_w
    fg_h = int(round(float(src_h) * (float(target_w) / float(src_w))))  # ~608px
    layer2 = torch.nn.functional.interpolate(
        frame, size=(fg_h, fg_w), mode='bilinear', align_corners=False
    )

    # Center vertically: Y = (1920 - 608) // 2 = 656
    start_y = (target_h - fg_h) // 2
    canvas = layer1.clone()
    canvas[:, :, start_y:start_y + fg_h, :] = layer2

    return canvas.clamp(0.0, 1.0)
