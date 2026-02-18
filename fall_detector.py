"""
Fall Detector - Temporal Analysis Engine v3.2

v3.2 Improvements over v3.1:
- Savitzky-Golay filter for keypoint smoothing (replaces simple EMA)
- Polygon area Z-axis proxy (nose-shoulder-hip triangle area)
- Bone length consistency check (sudden 2D shortening = depth rotation)
- Low-keypoint tolerance mode (dark clothing / low confidence)
- Long-lie logic: motion spike + sustained stillness + low R_compression = ALARM
- Adaptive score tolerance when visible_keypoints < 10
"""

import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
import json
import os
import config

# =============================================================================
# Savitzky-Golay 1D filter (no scipy dependency)
# =============================================================================

def _sg_coeffs(window: int, poly: int) -> np.ndarray:
    """Compute Savitzky-Golay smoothing coefficients."""
    half = window // 2
    A = np.array([[i**p for p in range(poly + 1)] for i in range(-half, half + 1)],
                 dtype=float)
    ATA_inv = np.linalg.pinv(A.T @ A)
    return (ATA_inv @ A.T)[0]  # Row 0 = smoothing coefficients


# Pre-compute SG coefficients (window=7, poly=2)
_SG_WINDOW = 7
_SG_POLY = 2
_SG_COEFFS = _sg_coeffs(_SG_WINDOW, _SG_POLY)


def sg_smooth(values: List[float]) -> float:
    """Apply SG filter to a list, return smoothed value at the last position."""
    n = len(values)
    if n < _SG_WINDOW:
        # Fallback: simple mean
        return sum(values) / n
    window = values[-_SG_WINDOW:]
    return float(np.dot(_SG_COEFFS, window))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PoseSnapshot:
    """Single frame of pose data."""
    timestamp: float
    center_of_mass: Optional[Tuple[float, float]]
    torso_angle: Optional[float]
    bbox_ratio: float
    bbox_center_y: float
    bbox_height: float
    visible_keypoints: int
    is_horizontal_pose: bool
    hip_shoulder_proximity: Optional[float]
    torso_compression: Optional[float]      # R_compression = |y_torso| / |x_shoulders|
    nose_hip_spread: Optional[float]
    nose_y: Optional[float]
    knee_ankle_proximity: Optional[float]
    vertical_keypoint_spread: Optional[float]
    lower_keypoints_missing: bool
    upper_lower_collision: Optional[float]
    polygon_area: Optional[float]           # Area of nose-shoulder-hip polygon (Z-proxy)
    shoulder_width: Optional[float]         # W_shoulders (scale reference)
    is_kneeling: bool = False
    in_bed_zone: bool = False


@dataclass
class FallState:
    """Current state of fall detection for one person."""
    score: float = 0.0
    state: str = "MONITORING"
    state_start_time: Optional[float] = None
    alarm_active: bool = False
    standing_com_y: Optional[float] = None
    standing_bbox_height: Optional[float] = None
    standing_shoulder_width: Optional[float] = None   # For bone length consistency
    standing_polygon_area: Optional[float] = None     # Calibrated polygon area
    calibration_samples: List[float] = field(default_factory=list)
    last_descent_detected: Optional[float] = None
    last_motion_spike_time: Optional[float] = None    # For long-lie logic

    # Disappearance tracking
    person_visible: bool = False
    last_person_seen: Optional[float] = None
    person_gone_since: Optional[float] = None


# =============================================================================
# FallDetector v3.2
# =============================================================================

class FallDetector:
    """
    Temporal fall detection engine v3.2.

    Signals:
    1. CoM Velocity       - rapid downward movement
    2. Pose Score         - horizontal bbox / torso angle
    3. Descent Score      - CoM drop from standing height
    4. Stillness Score    - post-fall inactivity (with long-lie logic)
    5. SD Posture Score   - foreshortening / depth-axis falls
       - Torso Compression (R_compression) with 20-frame voting
       - Nose Vertical Velocity (SG-smoothed)
       - Polygon Area Z-proxy (nose-shoulder-hip triangle)
       - Bone Length Consistency
       - Keypoint Collision
       - Prone/Supine patterns
    """

    SD_BUFFER_SIZE = 20
    SD_COMPRESSION_THRESHOLD = 0.55
    SD_MIN_FRAMES = 10

    def __init__(self):
        self.history: deque[PoseSnapshot] = deque(maxlen=150)  # 5s at 30fps
        self.fall_state = FallState()
        self._fps_estimate = 30.0
        self._last_timestamp = None
        self._last_kp_data = None

        # SD voting buffer
        self._sd_compression_buffer: deque[float] = deque(maxlen=self.SD_BUFFER_SIZE)

        # SG-smoothed nose Y buffer (timestamp, raw_y)
        self._nose_raw_buffer: deque[Tuple[float, float]] = deque(maxlen=30)
        # Smoothed nose positions
        self._nose_smooth_buffer: deque[Tuple[float, float]] = deque(maxlen=30)

        # Polygon area buffer (for Z-proxy trend)
        self._polygon_area_buffer: deque[float] = deque(maxlen=20)

        # Bone length history: shoulder_width over time
        self._shoulder_width_buffer: deque[float] = deque(maxlen=20)

        self._zones = {'bed': [], 'door': []}
        self.load_zones()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def load_zones(self):
        try:
            with open('zones.json', 'r') as f:
                loaded_zones = json.load(f)
            self._zones = {'bed': [], 'door': []}
            for key in ['bed', 'door']:
                if key in loaded_zones:
                    for z in loaded_zones[key]:
                        if len(z) == 4 and isinstance(z[0], (int, float)):
                            x1, y1, x2, y2 = z
                            self._zones[key].append([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
                        else:
                            self._zones[key].append(z)
            self.save_zones(self._zones)
        except (FileNotFoundError, json.JSONDecodeError):
            self._zones = {'bed': [], 'door': []}

    def save_zones(self, zones):
        with open('zones.json', 'w') as f:
            json.dump(zones, f)
        self._zones = zones

    def get_zones(self):
        return self._zones

    def update(self, bbox, keypoints, timestamp: Optional[float] = None) -> dict:
        if timestamp is None:
            timestamp = time.time()

        if self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp
            if 0.001 < dt < 1.0:
                self._fps_estimate = 0.9 * self._fps_estimate + 0.1 * (1.0 / dt)
        self._last_timestamp = timestamp

        snapshot = self._create_snapshot(bbox, keypoints, timestamp)
        self.history.append(snapshot)

        # Update SD compression buffer
        self._sd_compression_buffer.append(
            snapshot.torso_compression if snapshot.torso_compression is not None else 1.0
        )

        # Update nose buffers with SG smoothing
        if snapshot.nose_y is not None:
            self._nose_raw_buffer.append((timestamp, snapshot.nose_y))
            raw_ys = [y for _, y in self._nose_raw_buffer]
            smoothed_y = sg_smooth(raw_ys)
            self._nose_smooth_buffer.append((timestamp, smoothed_y))

        # Update polygon area buffer
        if snapshot.polygon_area is not None:
            self._polygon_area_buffer.append(snapshot.polygon_area)

        # Update shoulder width buffer
        if snapshot.shoulder_width is not None:
            self._shoulder_width_buffer.append(snapshot.shoulder_width)

        self.fall_state.person_visible = True
        self.fall_state.last_person_seen = timestamp
        self.fall_state.person_gone_since = None

        if len(self.history) < 5:
            return self._make_result()

        self._update_calibration(snapshot)

        # Low-keypoint mode: relax tolerances when data is noisy (dark clothing)
        low_kp_mode = snapshot.visible_keypoints < 10

        # --- Compute all signal scores ---
        velocity_score   = self._compute_velocity_score(low_kp_mode)
        pose_score       = self._compute_pose_score(snapshot)
        descent_score    = self._compute_descent_score(snapshot)
        stillness_score  = self._compute_stillness_score(timestamp)
        sd_posture_score = self._compute_sd_posture_score(snapshot, timestamp, low_kp_mode)

        # Bed Zone suppression
        if snapshot.in_bed_zone:
            self.fall_state.state = "RESTING"
            self.fall_state.score = 0.0
            self.fall_state.alarm_active = False
            return self._make_result(in_bed=1.0)

        if self.fall_state.state == "RESTING":
            self.fall_state.state = "MONITORING"

        # Kneeling suppression
        if snapshot.is_kneeling:
            pose_score       *= 0.3
            descent_score    *= 0.4
            sd_posture_score *= 0.3

        # SD posture replaces pose_score when foreshortening detected
        effective_pose = max(pose_score, sd_posture_score)

        raw_score = (
            config.WEIGHT_VELOCITY  * velocity_score  +
            config.WEIGHT_POSE      * effective_pose   +
            config.WEIGHT_DESCENT   * descent_score    +
            config.WEIGHT_STILLNESS * stillness_score
        )

        # Long-lie boost: motion spike + sustained stillness + compressed torso
        long_lie_boost = self._compute_long_lie_boost(timestamp, raw_score)
        raw_score = min(1.0, raw_score + long_lie_boost)

        # SG smoothing of score (replaces simple EMA)
        score_history = [s.bbox_ratio for s in list(self.history)[-_SG_WINDOW:]]  # placeholder
        # Use proper score smoothing: keep last N raw scores
        if not hasattr(self, '_raw_score_buffer'):
            self._raw_score_buffer = deque(maxlen=_SG_WINDOW)
        self._raw_score_buffer.append(raw_score)
        smoothed_score = sg_smooth(list(self._raw_score_buffer))

        self.fall_state.score = smoothed_score
        self._update_state(timestamp, raw_score)

        return self._make_result(
            velocity_score=velocity_score,
            pose_score=pose_score,
            descent_score=descent_score,
            stillness_score=stillness_score,
            sd_posture_score=sd_posture_score,
            effective_pose=effective_pose,
            long_lie_boost=long_lie_boost,
            low_kp_mode=1.0 if low_kp_mode else 0.0,
            kneeling=1.0 if snapshot.is_kneeling else 0.0,
            compression=snapshot.torso_compression or 0.0,
            nose_hip_spread=snapshot.nose_hip_spread or 0.0,
        )

    def update_no_person(self, timestamp: Optional[float] = None) -> dict:
        if timestamp is None:
            timestamp = time.time()
        state = self.fall_state

        if state.person_visible and state.last_person_seen is not None:
            state.person_visible = False
            if state.person_gone_since is None:
                state.person_gone_since = timestamp

        if state.person_gone_since is not None:
            last_pos = None
            if self.history:
                last_snap = self.history[-1]
                last_pos = last_snap.center_of_mass or (0.5, last_snap.bbox_center_y)

            was_in_door = bool(last_pos and
                self._is_point_in_zones(last_pos, self._zones.get('door', [])))

            if was_in_door:
                state.state = "MONITORING"
                state.person_gone_since = None
                state.state_start_time = None
                return self._make_result(in_door=1.0)

            gone_duration = timestamp - state.person_gone_since
            if gone_duration >= config.PERSON_GONE_ALERT_TIME:
                if state.state in ("MONITORING", "VERIFICATION"):
                    self._transition_to("PRE_ALARM", timestamp)
                    state.score = max(state.score, config.FALL_SCORE_THRESHOLD)
                if state.state == "PRE_ALARM":
                    time_in_state = timestamp - (state.state_start_time or timestamp)
                    if time_in_state >= config.PRE_ALARM_TIME:
                        self._transition_to("ALARM", timestamp)

        return self._make_result()

    def reset(self):
        self.fall_state = FallState()
        self.history.clear()
        self._sd_compression_buffer.clear()
        self._nose_raw_buffer.clear()
        self._nose_smooth_buffer.clear()
        self._polygon_area_buffer.clear()
        self._shoulder_width_buffer.clear()
        if hasattr(self, '_raw_score_buffer'):
            self._raw_score_buffer.clear()

    # -------------------------------------------------------------------------
    # Snapshot Creation
    # -------------------------------------------------------------------------

    def _create_snapshot(self, bbox, keypoints, timestamp: float) -> PoseSnapshot:
        bbox_w = bbox.width()
        bbox_h = bbox.height()
        bbox_ratio = bbox_w / max(bbox_h, 0.001)
        bbox_center_y = bbox.ymin() + bbox_h / 2.0

        kp_data = self._extract_keypoints(bbox, keypoints)
        visible_count = sum(1 for k in kp_data.values() if k is not None)

        com = self._compute_com(kp_data)
        torso_angle = self._compute_torso_angle(kp_data)
        is_horizontal = bbox_ratio > config.FALL_ASPECT_RATIO
        hip_shoulder_prox = self._compute_hip_shoulder_proximity(kp_data)
        is_kneeling_pose = self._is_kneeling(kp_data, torso_angle)
        self._last_kp_data = kp_data

        ls, rs = kp_data.get('left_shoulder'), kp_data.get('right_shoulder')
        lh, rh = kp_data.get('left_hip'), kp_data.get('right_hip')
        nose = kp_data.get('nose')

        # Torso Compression Ratio
        torso_compression = None
        shoulder_width = None
        if ls and rs and lh and rh:
            shoulder_y = (ls[1] + rs[1]) / 2
            hip_y = (lh[1] + rh[1]) / 2
            vertical_torso = abs(shoulder_y - hip_y)
            shoulder_width = abs(ls[0] - rs[0])
            if shoulder_width > 0.03:
                torso_compression = vertical_torso / shoulder_width

        # Nose-Hip Spread
        nose_hip_spread = None
        nose_y = nose[1] if nose else None
        if nose and lh and rh:
            hip_y_avg = (lh[1] + rh[1]) / 2
            nose_hip_spread = abs(hip_y_avg - nose[1])

        # Upper-Lower Keypoint Collision
        upper_lower_collision = None
        upper_ys = [p[1] for p in [nose, ls, rs] if p]
        lower_ys = [p[1] for p in [lh, rh] if p]
        if upper_ys and lower_ys:
            upper_lower_collision = abs(
                sum(upper_ys)/len(upper_ys) - sum(lower_ys)/len(lower_ys)
            )

        # Knee-Ankle Proximity
        lk = kp_data.get('left_knee')
        rk = kp_data.get('right_knee')
        la = kp_data.get('left_ankle')
        ra = kp_data.get('right_ankle')
        knees = [k for k in [lk, rk] if k]
        ankles = [a for a in [la, ra] if a]
        knee_ankle_proximity = None
        if knees and ankles:
            avg_knee_y = sum(k[1] for k in knees) / len(knees)
            avg_ankle_y = sum(a[1] for a in ankles) / len(ankles)
            knee_ankle_proximity = abs(avg_ankle_y - avg_knee_y)

        # Vertical Keypoint Spread
        all_visible = [v for v in kp_data.values() if v is not None]
        vertical_keypoint_spread = None
        if len(all_visible) >= 4:
            ys = [p[1] for p in all_visible]
            vertical_keypoint_spread = max(ys) - min(ys)

        # Lower Keypoints Missing
        lower_kp_names = ['left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        lower_missing = sum(1 for n in lower_kp_names if kp_data.get(n) is None)
        lower_keypoints_missing = lower_missing >= 3

        # --- Polygon Area Z-proxy ---
        # Area of polygon: nose, left_shoulder, right_shoulder, right_hip, left_hip
        # In standing: large area. In depth-axis fall: area collapses toward 0.
        polygon_area = None
        poly_pts = [p for p in [nose, ls, rs, rh, lh] if p is not None]
        if len(poly_pts) >= 3:
            # Shoelace formula for polygon area
            pts = np.array(poly_pts, dtype=float)
            n = len(pts)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += pts[i][0] * pts[j][1]
                area -= pts[j][0] * pts[i][1]
            polygon_area = abs(area) / 2.0

        in_bed_zone = False
        if com:
            in_bed_zone = self._is_point_in_zones(com, self._zones.get('bed', []))

        return PoseSnapshot(
            timestamp=timestamp,
            center_of_mass=com,
            torso_angle=torso_angle,
            bbox_ratio=bbox_ratio,
            bbox_center_y=bbox_center_y,
            bbox_height=bbox_h,
            visible_keypoints=visible_count,
            is_horizontal_pose=is_horizontal,
            hip_shoulder_proximity=hip_shoulder_prox,
            is_kneeling=is_kneeling_pose,
            in_bed_zone=in_bed_zone,
            torso_compression=torso_compression,
            nose_hip_spread=nose_hip_spread,
            nose_y=nose_y,
            knee_ankle_proximity=knee_ankle_proximity,
            vertical_keypoint_spread=vertical_keypoint_spread,
            lower_keypoints_missing=lower_keypoints_missing,
            upper_lower_collision=upper_lower_collision,
            polygon_area=polygon_area,
            shoulder_width=shoulder_width,
        )

    def _extract_keypoints(self, bbox, keypoints) -> dict:
        names = {
            0: 'nose', 5: 'left_shoulder', 6: 'right_shoulder',
            11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee',
            15: 'left_ankle', 16: 'right_ankle',
        }
        result = {name: None for name in names.values()}
        for idx, name in names.items():
            try:
                kp = keypoints[idx]
                if kp.confidence() > config.SCORE_THRESHOLD:
                    x = kp.x() * bbox.width() + bbox.xmin()
                    y = kp.y() * bbox.height() + bbox.ymin()
                    result[name] = (x, y)
            except (IndexError, AttributeError):
                pass
        return result

    def _compute_com(self, kp_data: dict) -> Optional[Tuple[float, float]]:
        for names in [['left_hip', 'right_hip'], ['left_shoulder', 'right_shoulder']]:
            pts = [kp_data[n] for n in names if kp_data.get(n)]
            if pts:
                return (sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts))
        return None

    def _compute_torso_angle(self, kp_data: dict) -> Optional[float]:
        ls, rs = kp_data.get('left_shoulder'), kp_data.get('right_shoulder')
        lh, rh = kp_data.get('left_hip'), kp_data.get('right_hip')
        if not (ls or rs) or not (lh or rh):
            return None
        sm = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2) if ls and rs else (ls or rs)
        hm = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2) if lh and rh else (lh or rh)
        dx, dy = hm[0]-sm[0], hm[1]-sm[1]
        if abs(dy) < 0.001:
            return 90.0
        return math.degrees(math.atan2(abs(dx), abs(dy)))

    def _compute_hip_shoulder_proximity(self, kp_data: dict) -> Optional[float]:
        ls, rs = kp_data.get('left_shoulder'), kp_data.get('right_shoulder')
        lh, rh = kp_data.get('left_hip'), kp_data.get('right_hip')
        sy = [p[1] for p in [ls, rs] if p]
        hy = [p[1] for p in [lh, rh] if p]
        if sy and hy:
            return abs(sum(hy)/len(hy) - sum(sy)/len(sy))
        return None

    def _is_point_in_zones(self, point, zones) -> bool:
        x, y = point
        pt = (int(x*1000), int(y*1000))
        for zone in zones:
            if len(zone) < 3:
                continue
            vp = np.array([(int(v[0]*1000), int(v[1]*1000)) for v in zone],
                          dtype=np.int32).reshape((-1,1,2))
            if cv2.pointPolygonTest(vp, pt, False) >= 0:
                return True
        return False

    def _is_kneeling(self, kp_data: dict, torso_angle: Optional[float]) -> bool:
        lk, rk = kp_data.get('left_knee'), kp_data.get('right_knee')
        lh, rh = kp_data.get('left_hip'), kp_data.get('right_hip')
        la, ra = kp_data.get('left_ankle'), kp_data.get('right_ankle')
        knees = [k for k in [lk, rk] if k]
        hips = [h for h in [lh, rh] if h]
        if not knees or not hips:
            return False
        avg_knee_y = sum(k[1] for k in knees) / len(knees)
        avg_hip_y = sum(h[1] for h in hips) / len(hips)
        ankles = [a for a in [la, ra] if a]
        ankles_near_knees = (
            bool(ankles) and
            abs(sum(a[1] for a in ankles)/len(ankles) - avg_knee_y) < 0.08
        )
        return avg_knee_y > avg_hip_y and (
            (torso_angle is not None and torso_angle < 55) or ankles_near_knees
        )

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    def _update_calibration(self, snapshot: PoseSnapshot):
        state = self.fall_state
        if (snapshot.center_of_mass is not None
                and not snapshot.is_horizontal_pose
                and snapshot.torso_angle is not None
                and snapshot.torso_angle < 30):
            state.calibration_samples.append(snapshot.center_of_mass[1])
            if len(state.calibration_samples) > config.STANDING_HEIGHT_FRAMES:
                state.calibration_samples = state.calibration_samples[-config.STANDING_HEIGHT_FRAMES:]
            if len(state.calibration_samples) >= 10:
                s = sorted(state.calibration_samples)
                state.standing_com_y = s[len(s)//2]
                state.standing_bbox_height = snapshot.bbox_height

            # Calibrate shoulder width (bone length reference)
            if (snapshot.shoulder_width is not None and
                    (state.standing_shoulder_width is None or
                     snapshot.shoulder_width > state.standing_shoulder_width)):
                state.standing_shoulder_width = snapshot.shoulder_width

            # Calibrate polygon area
            if (snapshot.polygon_area is not None and
                    (state.standing_polygon_area is None or
                     snapshot.polygon_area > state.standing_polygon_area)):
                state.standing_polygon_area = snapshot.polygon_area

    # -------------------------------------------------------------------------
    # Signal Scores
    # -------------------------------------------------------------------------

    def _compute_velocity_score(self, low_kp_mode: bool = False) -> float:
        """CoM vertical velocity. In low-kp mode, use wider time window."""
        if len(self.history) < 3:
            return 0.0
        window = config.HISTORY_WINDOW_SECONDS * (1.5 if low_kp_mode else 1.0)
        now = self.history[-1].timestamp
        cutoff = now - window
        positions = [(s.timestamp, s.center_of_mass[1])
                     for s in self.history
                     if s.timestamp >= cutoff and s.center_of_mass]
        if len(positions) < 3:
            positions = [(s.timestamp, s.bbox_center_y)
                         for s in self.history if s.timestamp >= cutoff]
        if len(positions) < 3:
            return 0.0
        max_velocity = 0.0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dt = positions[j][0] - positions[i][0]
                if 0.1 < dt < (1.0 if low_kp_mode else 0.8):
                    dy = positions[j][1] - positions[i][1]
                    pv = (dy / dt) * config.HEIGHT
                    if pv > max_velocity:
                        max_velocity = pv
        # In low-kp mode, lower the threshold (data is noisy, accept weaker signal)
        threshold = config.COM_VELOCITY_THRESHOLD * (0.7 if low_kp_mode else 1.0)
        return min(1.0, max_velocity / threshold)

    def _compute_pose_score(self, snapshot: PoseSnapshot) -> float:
        """Static pose score (side falls, horizontal bbox)."""
        score = 0.0
        if snapshot.is_horizontal_pose:
            score += 0.5
        if snapshot.hip_shoulder_proximity is not None:
            pt = config.KEYPOINT_PROXIMITY_THRESHOLD
            if snapshot.hip_shoulder_proximity < pt:
                score += 0.3 * (1.0 - snapshot.hip_shoulder_proximity / pt)
        if snapshot.torso_angle is not None:
            if snapshot.torso_angle > 60:
                score += 0.2
            elif snapshot.torso_angle > 40:
                score += 0.1
        return min(1.0, score)

    def _compute_sd_posture_score(self, snapshot: PoseSnapshot,
                                   timestamp: float,
                                   low_kp_mode: bool = False) -> float:
        """
        SD (Same Direction / foreshortening) posture score.

        Sub-signals:
        1. Torso Compression (R_compression) with 20-frame voting
        2. Nose Vertical Velocity (SG-smoothed)
        3. Polygon Area Z-proxy (nose-shoulder-hip area collapse)
        4. Bone Length Consistency (shoulder width shortening)
        5. Keypoint Collision (Y-axis clustering)
        6. Prone/Supine end-state patterns
        """
        score = 0.0
        compression_threshold = self.SD_COMPRESSION_THRESHOLD * (1.1 if low_kp_mode else 1.0)

        # ---------------------------------------------------------------
        # 1. Torso Compression with Temporal Voting
        # ---------------------------------------------------------------
        if len(self._sd_compression_buffer) >= 5:
            compressed = sum(1 for v in self._sd_compression_buffer
                             if v < compression_threshold)
            ratio = compressed / len(self._sd_compression_buffer)
            if ratio >= 0.4:
                avg_c = sum(self._sd_compression_buffer) / len(self._sd_compression_buffer)
                depth = max(0.0, 1.0 - avg_c / compression_threshold)
                score += 0.35 * depth * ratio
                if compressed >= self.SD_MIN_FRAMES:
                    score += 0.10  # Confirmed sustained compression

        # Current frame compression (immediate)
        if snapshot.torso_compression is not None:
            if snapshot.torso_compression < compression_threshold:
                score += 0.10 * (1.0 - snapshot.torso_compression / compression_threshold)

        # ---------------------------------------------------------------
        # 2. Nose Vertical Velocity (SG-smoothed)
        # ---------------------------------------------------------------
        nose_vel_score = self._compute_nose_velocity_score()
        score += 0.25 * nose_vel_score

        # ---------------------------------------------------------------
        # 3. Polygon Area Z-proxy
        # ---------------------------------------------------------------
        # Area of nose-shoulder-hip polygon collapses in depth-axis falls
        if (snapshot.polygon_area is not None and
                self.fall_state.standing_polygon_area is not None and
                self.fall_state.standing_polygon_area > 0.001):
            area_ratio = snapshot.polygon_area / self.fall_state.standing_polygon_area
            if area_ratio < 0.4:  # Area collapsed to < 40% of standing
                area_score = 1.0 - area_ratio / 0.4
                score += 0.20 * area_score
            elif area_ratio < 0.7:
                score += 0.05

        # Also check trend: rapid area decrease
        if len(self._polygon_area_buffer) >= 5:
            recent_areas = list(self._polygon_area_buffer)
            early_avg = sum(recent_areas[:5]) / 5
            late_avg = sum(recent_areas[-5:]) / 5
            if early_avg > 0.001 and late_avg / early_avg < 0.5:
                score += 0.10  # Rapid area collapse

        # ---------------------------------------------------------------
        # 4. Bone Length Consistency
        # ---------------------------------------------------------------
        # Shoulder width should be stable. Sudden drop = depth rotation
        if (snapshot.shoulder_width is not None and
                self.fall_state.standing_shoulder_width is not None and
                self.fall_state.standing_shoulder_width > 0.02):
            width_ratio = snapshot.shoulder_width / self.fall_state.standing_shoulder_width
            # If shoulder width drops > 50%, body is rotating in Z axis
            if width_ratio < 0.5:
                bone_score = 1.0 - width_ratio / 0.5
                score += 0.15 * bone_score
            elif width_ratio < 0.7:
                score += 0.05

        # ---------------------------------------------------------------
        # 5. Keypoint Collision (Y-axis clustering)
        # ---------------------------------------------------------------
        if snapshot.upper_lower_collision is not None:
            collision_threshold = 0.10 * (1.3 if low_kp_mode else 1.0)
            if snapshot.upper_lower_collision < collision_threshold:
                score += 0.15 * (1.0 - snapshot.upper_lower_collision / collision_threshold)

        if snapshot.nose_hip_spread is not None:
            if snapshot.nose_hip_spread < 0.08:
                score += 0.20  # Very strong: nose at same height as hips
            elif snapshot.nose_hip_spread < 0.15:
                score += 0.12
            elif snapshot.nose_hip_spread < 0.22:
                score += 0.05

        # ---------------------------------------------------------------
        # 6. Prone/Supine End-State Patterns
        # ---------------------------------------------------------------
        # Prone: lower kp missing + nose visible + compressed torso
        if (snapshot.lower_keypoints_missing and
                snapshot.nose_y is not None and
                snapshot.torso_compression is not None and
                snapshot.torso_compression < compression_threshold):
            score += 0.12

        # Supine: knee-ankle close + compressed torso
        if (snapshot.knee_ankle_proximity is not None and
                snapshot.knee_ankle_proximity < 0.12 and
                snapshot.torso_compression is not None and
                snapshot.torso_compression < compression_threshold):
            score += 0.08

        # ---------------------------------------------------------------
        # 7. Vertical Keypoint Spread
        # ---------------------------------------------------------------
        if snapshot.vertical_keypoint_spread is not None and snapshot.bbox_height > 0.05:
            norm_spread = snapshot.vertical_keypoint_spread / snapshot.bbox_height
            if norm_spread < 0.40:
                score += 0.08 * (1.0 - norm_spread / 0.40)

        return min(1.0, score)

    def _compute_nose_velocity_score(self) -> float:
        """
        Nose vertical velocity using SG-smoothed positions.
        Key signal for depth-axis falls.
        """
        if len(self._nose_smooth_buffer) < 5:
            return 0.0

        ref_height = self.fall_state.standing_bbox_height or 0.35
        nose_list = list(self._nose_smooth_buffer)
        max_score = 0.0

        for i in range(len(nose_list)):
            for j in range(i+1, len(nose_list)):
                dt = nose_list[j][0] - nose_list[i][0]
                if 0.05 < dt < 0.7:
                    dy = nose_list[j][1] - nose_list[i][1]
                    if dy > 0:
                        drop_ratio = dy / ref_height
                        if drop_ratio > 0.35:
                            max_score = max(max_score, min(1.0, drop_ratio / 0.35))
                        elif drop_ratio > 0.15:
                            max_score = max(max_score, drop_ratio / 0.35)

        # Total drop over recent 1.0s
        now = nose_list[-1][0]
        recent = [(t, y) for t, y in nose_list if t >= now - 1.0]
        if len(recent) >= 2:
            total_drop = recent[-1][1] - recent[0][1]
            drop_ratio = total_drop / ref_height
            if drop_ratio > 0.25:
                max_score = max(max_score, min(1.0, drop_ratio / 0.35))

        return max_score

    def _compute_descent_score(self, snapshot: PoseSnapshot) -> float:
        state = self.fall_state
        if state.standing_com_y is None:
            return 0.0
        current_y = (snapshot.center_of_mass[1] if snapshot.center_of_mass
                     else snapshot.bbox_center_y)
        descent = current_y - state.standing_com_y
        if descent <= 0:
            return 0.0
        ref_height = state.standing_bbox_height or 0.3
        descent_ratio = descent / max(ref_height, 0.01)
        threshold = config.MIN_FALL_DESCENT_RATIO
        if descent_ratio < threshold * 0.5:
            return 0.0
        return min(1.0, (descent_ratio - threshold * 0.5) / (threshold * 0.5))

    def _compute_stillness_score(self, timestamp: float) -> float:
        """Post-fall inactivity. Includes SD foreshortening poses as 'down'."""
        if len(self.history) < 10:
            return 0.0
        recent = list(self.history)[-10:]

        down_count = sum(1 for s in recent if (
            s.is_horizontal_pose or
            (s.torso_angle is not None and s.torso_angle > 50) or
            (s.torso_compression is not None and
             s.torso_compression < self.SD_COMPRESSION_THRESHOLD) or
            (s.upper_lower_collision is not None and s.upper_lower_collision < 0.12) or
            (s.nose_hip_spread is not None and s.nose_hip_spread < 0.20)
        ))

        if down_count < 6:
            self.fall_state.last_descent_detected = None
            return 0.0

        if self.fall_state.last_descent_detected is None:
            self.fall_state.last_descent_detected = timestamp
            return 0.0

        time_down = timestamp - self.fall_state.last_descent_detected

        # Stillness check via CoM variance
        com_positions = [s.center_of_mass for s in recent if s.center_of_mass]
        is_still = True
        if len(com_positions) >= 3:
            ys = [p[1] for p in com_positions]
            is_still = (max(ys) - min(ys)) < 0.05

        # Nose stillness (for SD falls)
        nose_ys = [s.nose_y for s in recent if s.nose_y is not None]
        if len(nose_ys) >= 3 and (max(nose_ys) - min(nose_ys)) < 0.04:
            is_still = True

        if not is_still:
            return min(0.5, time_down / config.POST_FALL_STILLNESS_TIME * 0.5)

        return min(1.0, time_down / config.POST_FALL_STILLNESS_TIME)

    def _compute_long_lie_boost(self, timestamp: float, raw_score: float) -> float:
        """
        NEW v3.2: Long-lie detection.

        Pattern: motion spike (any rapid movement) followed by
        sustained stillness (3-5s) with low R_compression.
        This distinguishes a fall from kneeling/bending (which involves recovery).

        Returns a score boost (0.0 - 0.3).
        """
        if len(self.history) < 15:
            return 0.0

        state = self.fall_state
        recent_20 = list(self.history)[-20:]
        recent_10 = recent_20[-10:]

        # Check if currently compressed (SD posture)
        compressed_count = sum(
            1 for s in recent_10
            if s.torso_compression is not None and
               s.torso_compression < self.SD_COMPRESSION_THRESHOLD
        )
        is_compressed = compressed_count >= 5

        # Check stillness in recent 10 frames
        nose_ys = [s.nose_y for s in recent_10 if s.nose_y is not None]
        com_ys = [s.center_of_mass[1] for s in recent_10 if s.center_of_mass]

        is_still_now = True
        if len(com_ys) >= 3:
            is_still_now = (max(com_ys) - min(com_ys)) < 0.04
        if len(nose_ys) >= 3:
            if (max(nose_ys) - min(nose_ys)) < 0.03:
                is_still_now = True

        if not is_still_now:
            # Not still - check if this is a motion spike to record
            if raw_score > 0.35:
                state.last_motion_spike_time = timestamp
            return 0.0

        # Currently still - check if there was a recent motion spike
        if state.last_motion_spike_time is None:
            return 0.0

        time_since_spike = timestamp - state.last_motion_spike_time
        still_duration = 0.0

        # Estimate how long we've been still
        for s in reversed(recent_20):
            if s.center_of_mass and com_ys:
                if abs(s.center_of_mass[1] - com_ys[-1]) < 0.04:
                    still_duration = timestamp - s.timestamp
                else:
                    break

        # Long-lie: spike happened, then 3+ seconds of stillness with compression
        if time_since_spike > 0.5 and still_duration >= 2.0 and is_compressed:
            # Boost proportional to how long they've been still
            boost = min(0.30, (still_duration - 2.0) / 5.0 * 0.30)
            return boost

        # Even without compression: spike + 4s stillness is suspicious
        if time_since_spike > 0.5 and still_duration >= 4.0:
            return min(0.15, (still_duration - 4.0) / 5.0 * 0.15)

        return 0.0

    # -------------------------------------------------------------------------
    # State Machine
    # -------------------------------------------------------------------------

    def _update_state(self, timestamp: float, raw_score: float):
        """
        Adaptive state machine.
        High score = faster alarm path.
        """
        state = self.fall_state

        if state.state_start_time is None:
            state.state_start_time = timestamp

        time_in_state = timestamp - state.state_start_time

        # Critical fall: skip VERIFICATION
        if raw_score > config.CRITICAL_SCORE:
            if state.state in ["MONITORING", "VERIFICATION"]:
                self._transition_to("PRE_ALARM", timestamp)
                return

        if state.state == "MONITORING":
            if raw_score > config.FALL_SCORE_THRESHOLD:
                self._transition_to("VERIFICATION", timestamp)

        elif state.state == "VERIFICATION":
            if raw_score < config.FALL_SCORE_THRESHOLD * 0.5:
                self._transition_to("MONITORING", timestamp)
            elif time_in_state >= config.VERIFICATION_TIME:
                self._transition_to("PRE_ALARM", timestamp)

        elif state.state == "PRE_ALARM":
            # Only cancel if score drops very low
            if raw_score < config.FALL_SCORE_THRESHOLD * 0.35:
                self._transition_to("MONITORING", timestamp)
            else:
                # Adaptive PRE_ALARM duration based on confidence
                if raw_score > 0.90:
                    effective = config.PRE_ALARM_TIME * 0.35
                elif raw_score > 0.80:
                    effective = config.PRE_ALARM_TIME * 0.55
                elif raw_score > 0.65:
                    effective = config.PRE_ALARM_TIME * 0.75
                else:
                    effective = config.PRE_ALARM_TIME

                if time_in_state >= effective:
                    self._transition_to("ALARM", timestamp)

        elif state.state == "ALARM":
            pass

    def _transition_to(self, new_state: str, timestamp: float):
        if self.fall_state.state != new_state:
            self.fall_state.state = new_state
            self.fall_state.state_start_time = timestamp
            if new_state == "ALARM":
                self.fall_state.alarm_active = True
            elif new_state == "MONITORING":
                self.fall_state.alarm_active = False

    # -------------------------------------------------------------------------
    # Result
    # -------------------------------------------------------------------------

    def _make_result(self, **details) -> dict:
        return {
            'score': round(self.fall_state.score, 3),
            'state': self.fall_state.state,
            'alarm_active': self.fall_state.alarm_active,
            'details': {k: round(v, 3) for k, v in details.items()},
        }
