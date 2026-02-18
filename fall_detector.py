"""
Fall Detector - Temporal Analysis Engine

Multi-stage fall detection using pose history, center-of-mass velocity,
torso angle tracking, and weighted scoring system.
"""

import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import cv2
import numpy as np
import json
import os
import config

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PoseSnapshot:
    """Single frame of pose data."""
    timestamp: float
    center_of_mass: Optional[Tuple[float, float]]  # (x, y) average of hips
    torso_angle: Optional[float]  # degrees from vertical (0=standing, 90=lying)
    bbox_ratio: float  # width / height
    bbox_center_y: float  # normalized (0-1) vertical center of bbox
    bbox_height: float  # normalized bbox height
    visible_keypoints: int  # count of confident keypoints
    is_horizontal_pose: bool  # static pose check
    hip_shoulder_proximity: Optional[float]  # normalized vertical distance
    torso_compression: Optional[float]  # Ratio of torso length / shoulder width
    is_kneeling: bool = False  # kneeling pose detected
    in_bed_zone: bool = False  # center of mass is inside a bed polygon


@dataclass
class FallState:
    """Current state of fall detection for one person."""
    score: float = 0.0
    state: str = "MONITORING"  # MONITORING / CAUTION / ALERT / ALARM
    alert_start_time: Optional[float] = None
    alarm_active: bool = False
    standing_com_y: Optional[float] = None  # calibrated "normal" standing height
    standing_bbox_height: Optional[float] = None
    calibration_samples: List[float] = field(default_factory=list)
    last_descent_detected: Optional[float] = None
    # Disappearance tracking
    person_visible: bool = False
    last_person_seen: Optional[float] = None
    person_gone_since: Optional[float] = None


# =============================================================================
# FallDetector
# =============================================================================

class FallDetector:
    """
    Temporal fall detection engine.

    Maintains a sliding window of pose snapshots and computes a
    weighted fall score (0.0 - 1.0) based on multiple signals.
    """

    def __init__(self):
        self.history: deque[PoseSnapshot] = deque(maxlen=120)  # ~4s at 30fps
        self.fall_state = FallState()
        self._fps_estimate = 30.0
        self._last_timestamp = None
        self._last_kp_data = None  # for kneeling check reuse
        self._zones = {'bed': [], 'door': []}
        self.load_zones()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def load_zones(self):
        """Load safe zones from JSON config. Auto-migrates legacy rect format."""
        try:
            with open('zones.json', 'r') as f:
                loaded_zones = json.load(f)
                
            # Migration check: Convert old [x1, y1, x2, y2] rects to polygons
            self._zones = {'bed': [], 'door': []}
            
            for key in ['bed', 'door']:
                if key in loaded_zones:
                    for z in loaded_zones[key]:
                        # Check if it's a flat list of 4 numbers (legacy rect)
                        if len(z) == 4 and isinstance(z[0], (int, float)):
                            # Convert to 4-point polygon
                            x1, y1, x2, y2 = z
                            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                            self._zones[key].append(poly)
                        else:
                            # Assume it's already a polygon (list of lists)
                            self._zones[key].append(z)
                            
            # Save back to update format
            self.save_zones(self._zones)
            
        except (FileNotFoundError, json.JSONDecodeError):
            self._zones = {'bed': [], 'door': []}

    def save_zones(self, zones):
        """Save safe zones to JSON config and reload."""
        with open('zones.json', 'w') as f:
            json.dump(zones, f)
        self._zones = zones

    def get_zones(self):
        """Return current zones."""
        return self._zones

    def update(self, bbox, keypoints, timestamp: Optional[float] = None) -> dict:
        """
        Process a new frame. Call this every frame.

        Args:
            bbox: Hailo bounding box object
            keypoints: List of Hailo keypoint objects (COCO 17-point format)
            timestamp: Frame timestamp (uses time.time() if None)

        Returns:
            dict with keys: score, state, alarm_active, details
        """
        if timestamp is None:
            timestamp = time.time()

        # Update FPS estimate
        if self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp
            if 0.001 < dt < 1.0:
                self._fps_estimate = 0.9 * self._fps_estimate + 0.1 * (1.0 / dt)
        self._last_timestamp = timestamp

        # Extract pose data
        snapshot = self._create_snapshot(bbox, keypoints, timestamp)
        self.history.append(snapshot)

        # Mark person as visible (for disappearance tracking)
        self.fall_state.person_visible = True
        self.fall_state.last_person_seen = timestamp
        self.fall_state.person_gone_since = None

        # Need at least some history for temporal analysis
        if len(self.history) < 5:
            return self._make_result()

        # Calibrate standing height from initial frames
        self._update_calibration(snapshot)

        # Compute individual signal scores (0.0 - 1.0 each)
        velocity_score = self._compute_velocity_score()
        pose_score = self._compute_pose_score(snapshot)
        descent_score = self._compute_descent_score(snapshot)
        stillness_score = self._compute_stillness_score(timestamp)

        # Bed Zone Logic: Suppress fall detection if in bed check
        if snapshot.in_bed_zone:
            # Person is sleeping/resting -> Force state to "RESTING" (no alarm)
            # We override the raw scores to 0
            velocity_score = 0.0
            pose_score = 0.0
            descent_score = 0.0
            stillness_score = 0.0
            # Also clear any pending alarm for this person
            self.fall_state.state = "RESTING"
            self.fall_state.score = 0.0
            self.fall_state.alarm_active = False
            self.fall_state.alert_start_time = None
            
            return self._make_result(
                velocity_score=0.0,
                pose_score=0.0,
                descent_score=0.0,
                stillness_score=0.0,
                kneeling=0.0,
                in_bed=1.0,
            )
        
        # If was RESTING but moved out, reset to MONITORING
        if self.fall_state.state == "RESTING":
            self.fall_state.state = "MONITORING"

        # Kneeling suppression: if kneeling, dampen pose and descent
        if snapshot.is_kneeling:
            pose_score *= 0.3
            descent_score *= 0.4

        # Weighted combination
        raw_score = (
            config.WEIGHT_VELOCITY * velocity_score +
            config.WEIGHT_POSE * pose_score +
            config.WEIGHT_DESCENT * descent_score +
            config.WEIGHT_STILLNESS * stillness_score
        )

        # Smooth the score (avoid flickering)
        self.fall_state.score = 0.7 * self.fall_state.score + 0.3 * raw_score

        # Update state machine
        self._update_state(timestamp)

        return self._make_result(
            velocity_score=velocity_score,
            pose_score=pose_score,
            descent_score=descent_score,
            stillness_score=stillness_score,
            kneeling=1.0 if snapshot.is_kneeling else 0.0,
            compression=snapshot.torso_compression if snapshot.torso_compression else 0.0,
        )

    def update_no_person(self, timestamp: Optional[float] = None) -> dict:
        """
        Call when NO person is detected in frame.
        Tracks disappearance and may escalate to CAUTION.
        """
        if timestamp is None:
            timestamp = time.time()

        state = self.fall_state

        # Was person previously visible?
        if state.person_visible and state.last_person_seen is not None:
            # Person just disappeared
            state.person_visible = False
            if state.person_gone_since is None:
                state.person_gone_since = timestamp

        if state.person_gone_since is not None:
            # Check if last seen position was in a Door Zone
            last_pos = None
            # Retrieve last valid center of mass or bbox center from history
            if self.history:
                last_snap = self.history[-1]
                if last_snap.center_of_mass:
                    last_pos = last_snap.center_of_mass
                else:
                    last_pos = (0.5, last_snap.bbox_center_y) # Approx x
            
            # Use FallState's last known position if history doesn't help
            # (Note: we should ideally store last pos in state, but let's check history first)
            
            # Simplified check: if recently seen in door zone, cancel alarm
            was_in_door = False
            if last_pos:
                was_in_door = self._is_point_in_zones(last_pos, self._zones.get('door', []))

            if was_in_door:
                # Person walked out the door - logical exit
                state.state = "MONITORING"
                state.person_gone_since = None
                return self._make_result(in_door=1.0)

            gone_duration = timestamp - state.person_gone_since

            # Escalation: CAUTION after threshold
            if gone_duration >= config.PERSON_GONE_CAUTION_TIME:
                if state.state == "MONITORING":
                    state.state = "CAUTION"

            # Escalation: ALERT after longer threshold
            if gone_duration >= config.PERSON_GONE_ALERT_TIME:
                if state.state in ("MONITORING", "CAUTION"):
                    state.state = "ALERT"
                    state.alert_start_time = state.alert_start_time or timestamp
                    state.score = max(state.score, config.FALL_SCORE_THRESHOLD)

                # Full alarm after sustained alert
                if (state.state == "ALERT" and state.alert_start_time
                        and (timestamp - state.alert_start_time) >= config.ALARM_DURATION_SECONDS):
                    state.state = "ALARM"
                    state.alarm_active = True

        return self._make_result()

    def reset(self):
        """Reset all state (e.g. when alarm is manually cleared)."""
        self.fall_state = FallState()
        self.history.clear()

    # -------------------------------------------------------------------------
    # Snapshot Creation
    # -------------------------------------------------------------------------

    def _create_snapshot(self, bbox, keypoints, timestamp: float) -> PoseSnapshot:
        """Extract all relevant data from a single frame."""

        # Bounding box metrics
        bbox_w = bbox.width()
        bbox_h = bbox.height()
        bbox_ratio = bbox_w / max(bbox_h, 0.001)
        bbox_center_y = bbox.ymin() + bbox_h / 2.0

        # Extract keypoints
        kp_data = self._extract_keypoints(bbox, keypoints)
        visible_count = sum(1 for k in kp_data.values() if k is not None)

        # Center of mass (average of hips, or shoulders if hips not visible)
        com = self._compute_com(kp_data)

        # Torso angle (spine line vs vertical)
        torso_angle = self._compute_torso_angle(kp_data)

        # Static horizontal pose check
        is_horizontal = bbox_ratio > config.FALL_ASPECT_RATIO

        # Hip-shoulder vertical proximity (normalized)
        hip_shoulder_prox = self._compute_hip_shoulder_proximity(kp_data)

        # Kneeling detection
        is_kneeling_pose = self._is_kneeling(kp_data, torso_angle)
        self._last_kp_data = kp_data

        # Zone Check
        # 5. Check for Foreshortening (End-on Fall)
        # Ratio of Vertical Torso Length / Horizontal Shoulder Width
        torso_compression = None
        ls, rs = kp_data.get('left_shoulder'), kp_data.get('right_shoulder')
        lh, rh = kp_data.get('left_hip'), kp_data.get('right_hip')

        if ls and rs and lh and rh:
            # Mid-points
            shoulder_y = (ls[1] + rs[1]) / 2
            hip_y = (lh[1] + rh[1]) / 2
            
            vertical_torso_len = abs(shoulder_y - hip_y)
            shoulder_width = abs(ls[0] - rs[0])
            
            # Avoid division by zero
            if shoulder_width > 0.05: # Minimum width to be considered "visible"
                torso_compression = vertical_torso_len / shoulder_width

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
        )

    def _extract_keypoints(self, bbox, keypoints) -> dict:
        """Extract named keypoints as normalized (0-1) coordinates."""
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
                    # Convert to absolute normalized coords (0-1 in frame)
                    x = kp.x() * bbox.width() + bbox.xmin()
                    y = kp.y() * bbox.height() + bbox.ymin()
                    result[name] = (x, y)
            except (IndexError, AttributeError):
                pass

        return result

    def _compute_com(self, kp_data: dict) -> Optional[Tuple[float, float]]:
        """Compute center of mass from available keypoints."""
        # Priority: hips > shoulders > bbox center
        points = []

        for name in ['left_hip', 'right_hip']:
            if kp_data.get(name) is not None:
                points.append(kp_data[name])

        if not points:
            for name in ['left_shoulder', 'right_shoulder']:
                if kp_data.get(name) is not None:
                    points.append(kp_data[name])

        if points:
            avg_x = sum(p[0] for p in points) / len(points)
            avg_y = sum(p[1] for p in points) / len(points)
            return (avg_x, avg_y)

        return None

    def _compute_torso_angle(self, kp_data: dict) -> Optional[float]:
        """Compute angle of spine relative to vertical (0=upright, 90=horizontal)."""
        # Spine = midpoint(shoulders) to midpoint(hips)
        ls, rs = kp_data.get('left_shoulder'), kp_data.get('right_shoulder')
        lh, rh = kp_data.get('left_hip'), kp_data.get('right_hip')

        if not (ls or rs) or not (lh or rh):
            return None

        # Midpoints (use available ones)
        if ls and rs:
            shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        else:
            shoulder_mid = ls or rs

        if lh and rh:
            hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        else:
            hip_mid = lh or rh

        # Angle from vertical
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]

        if abs(dy) < 0.001:
            return 90.0  # Perfectly horizontal

        angle_rad = math.atan2(abs(dx), abs(dy))
        return math.degrees(angle_rad)

    def _compute_hip_shoulder_proximity(self, kp_data: dict) -> Optional[float]:
        """Normalized vertical distance between hips and shoulders."""
        ls, rs = kp_data.get('left_shoulder'), kp_data.get('right_shoulder')
        lh, rh = kp_data.get('left_hip'), kp_data.get('right_hip')

        shoulders_y = []
        hips_y = []

        if ls:
            shoulders_y.append(ls[1])
        if rs:
            shoulders_y.append(rs[1])
        if lh:
            hips_y.append(lh[1])
        if rh:
            hips_y.append(rh[1])

        if shoulders_y and hips_y:
            avg_s = sum(shoulders_y) / len(shoulders_y)
            avg_h = sum(hips_y) / len(hips_y)
            return abs(avg_h - avg_s)

        return None

    def _is_point_in_zones(self, point: Tuple[float, float], zones: List[List[List[float]]]) -> bool:
        """
        Check if normalized (x,y) point is inside any of the given polygon zones.
        Zones are lists of vertices: [[x,y], [x,y], ...].
        """
        x, y = point
        # Convert normalized point to virtual pixel coordinates for standard OpenCV check
        # Use a virtual 1000x1000 grid for precision
        pt_virt = (int(x * 1000), int(y * 1000))

        for zone in zones:
            if len(zone) < 3:
                continue # Not a polygon
            
            # Convert normalized vertices to virtual grid
            vp_zone = np.array([(int(v[0] * 1000), int(v[1] * 1000)) for v in zone], dtype=np.int32)
            vp_zone = vp_zone.reshape((-1, 1, 2))

            # PointPolygonTest: >0 inside, =0 on edge, <0 outside
            dist = cv2.pointPolygonTest(vp_zone, pt_virt, False)
            if dist >= 0:
                return True

        return False

    def _is_kneeling(self, kp_data: dict, torso_angle: Optional[float]) -> bool:
        """
        Detect kneeling pose.
        Kneeling = knees visible & below hips & torso not fully horizontal.
        """
        # Need at least one knee and one hip
        lk = kp_data.get('left_knee')
        rk = kp_data.get('right_knee')
        lh = kp_data.get('left_hip')
        rh = kp_data.get('right_hip')

        knees = [k for k in [lk, rk] if k is not None]
        hips = [h for h in [lh, rh] if h is not None]

        if not knees or not hips:
            return False

        # Knees must be BELOW hips (higher Y = lower in image)
        avg_knee_y = sum(k[1] for k in knees) / len(knees)
        avg_hip_y = sum(h[1] for h in hips) / len(hips)

        knees_below_hips = avg_knee_y > avg_hip_y

        # Torso still somewhat upright (not fully collapsed)
        torso_upright = torso_angle is not None and torso_angle < 55

        # Check ankles - if visible and near knees, likely kneeling
        la = kp_data.get('left_ankle')
        ra = kp_data.get('right_ankle')
        ankles = [a for a in [la, ra] if a is not None]

        ankles_near_knees = False
        if ankles and knees:
            avg_ankle_y = sum(a[1] for a in ankles) / len(ankles)
            # Ankles close to knees vertically = legs folded = kneeling
            ankles_near_knees = abs(avg_ankle_y - avg_knee_y) < 0.08

        # Kneeling: knees below hips AND (torso upright OR ankles near knees)
        return knees_below_hips and (torso_upright or ankles_near_knees)

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    def _update_calibration(self, snapshot: PoseSnapshot):
        """Learn the person's standing CoM height from initial upright frames."""
        state = self.fall_state

        # Only calibrate from upright poses
        if (snapshot.center_of_mass is not None
                and not snapshot.is_horizontal_pose
                and snapshot.torso_angle is not None
                and snapshot.torso_angle < 30):

            state.calibration_samples.append(snapshot.center_of_mass[1])

            if len(state.calibration_samples) > config.STANDING_HEIGHT_FRAMES:
                state.calibration_samples = state.calibration_samples[-config.STANDING_HEIGHT_FRAMES:]

            if len(state.calibration_samples) >= 10:
                # Use median for robustness
                sorted_samples = sorted(state.calibration_samples)
                state.standing_com_y = sorted_samples[len(sorted_samples) // 2]
                state.standing_bbox_height = snapshot.bbox_height

    # -------------------------------------------------------------------------
    # Signal Scores (each returns 0.0 - 1.0)
    # -------------------------------------------------------------------------

    def _compute_velocity_score(self) -> float:
        """Score based on vertical velocity of CoM."""
        if len(self.history) < 3:
            return 0.0

        # Get CoM positions over recent window
        window_duration = config.HISTORY_WINDOW_SECONDS
        now = self.history[-1].timestamp
        cutoff = now - window_duration

        positions = []
        for snap in self.history:
            if snap.timestamp >= cutoff and snap.center_of_mass is not None:
                positions.append((snap.timestamp, snap.center_of_mass[1]))

        if len(positions) < 3:
            # Fallback to bbox center
            for snap in self.history:
                if snap.timestamp >= cutoff:
                    positions.append((snap.timestamp, snap.bbox_center_y))

        if len(positions) < 3:
            return 0.0

        # Compute max downward velocity over any 0.3s sub-window
        max_velocity = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dt = positions[j][0] - positions[i][0]
                if 0.1 < dt < 0.8:
                    # Downward = positive dy (y increases downward in image coords)
                    dy = positions[j][1] - positions[i][1]
                    velocity = dy / dt  # normalized units per second

                    # Convert to approximate pixel velocity
                    pixel_velocity = velocity * config.HEIGHT
                    if pixel_velocity > max_velocity:
                        max_velocity = pixel_velocity

        # Normalize: threshold = COM_VELOCITY_THRESHOLD
        threshold = config.COM_VELOCITY_THRESHOLD
        score = min(1.0, max_velocity / threshold) if threshold > 0 else 0.0
        return score

    def _compute_pose_score(self, snapshot: PoseSnapshot) -> float:
        """Score based on current static pose analysis."""
        score = 0.0

        # Horizontal bbox
        if snapshot.is_horizontal_pose:
            # Horizontal bbox or torso angle -> Strong indicator
            score += 0.4
        elif snapshot.torso_compression is not None and \
             snapshot.torso_compression < config.TORSO_COMPRESSION_THRESHOLD:
            # FORESHORTENED FALL (End-on view)
            # Body is "vertical" in 2D but torso is compressed (lying towards/away from camera)
            score += 0.5  # Strong indicator for this specific edge case
        elif snapshot.hip_shoulder_proximity is not None and \
             snapshot.hip_shoulder_proximity < 0.15:
             # Fallback: Hips and shoulders very close vertically
             score += 0.2
        
        # Hip-shoulder proximity (closer = more horizontal body)
        if snapshot.hip_shoulder_proximity is not None:
            proximity_threshold = config.KEYPOINT_PROXIMITY_THRESHOLD
            if snapshot.hip_shoulder_proximity < proximity_threshold:
                proximity_score = 1.0 - (snapshot.hip_shoulder_proximity / proximity_threshold)
                score += 0.3 * proximity_score

        # Torso angle (higher angle = more horizontal)
        if snapshot.torso_angle is not None:
            if snapshot.torso_angle > 60:
                score += 0.2
            elif snapshot.torso_angle > 40:
                score += 0.1

        return min(1.0, score)

    def _compute_descent_score(self, snapshot: PoseSnapshot) -> float:
        """Score based on how far CoM has dropped from standing height."""
        state = self.fall_state

        if state.standing_com_y is None:
            return 0.0

        current_com_y = None
        if snapshot.center_of_mass is not None:
            current_com_y = snapshot.center_of_mass[1]
        else:
            current_com_y = snapshot.bbox_center_y

        # How much has CoM dropped? (positive = downward in image)
        descent = current_com_y - state.standing_com_y

        if descent <= 0:
            return 0.0  # Person is at or above standing height

        # Normalize by standing bbox height (scale-independent)
        ref_height = state.standing_bbox_height or 0.3
        descent_ratio = descent / max(ref_height, 0.01)

        # Score: ramp from 0 at threshold/2 to 1.0 at threshold
        threshold = config.MIN_FALL_DESCENT_RATIO
        if descent_ratio < threshold * 0.5:
            return 0.0

        score = min(1.0, (descent_ratio - threshold * 0.5) / (threshold * 0.5))
        return score

    def _compute_stillness_score(self, timestamp: float) -> float:
        """Score based on post-descent inactivity (strongest signal)."""
        # Check if person is currently in a low/horizontal position
        if len(self.history) < 10:
            return 0.0

        recent = list(self.history)[-10:]

        # Is person currently "down"? (pose score > 0.3 in recent frames)
        down_count = sum(1 for s in recent if s.is_horizontal_pose or
                        (s.torso_angle is not None and s.torso_angle > 50))
        if down_count < 6:
            # Person is not consistently down
            self.fall_state.last_descent_detected = None
            return 0.0

        # How long have they been down?
        if self.fall_state.last_descent_detected is None:
            self.fall_state.last_descent_detected = timestamp
            return 0.0

        time_down = timestamp - self.fall_state.last_descent_detected
        stillness_threshold = config.POST_FALL_STILLNESS_TIME

        # Also check if they're actually still (low movement variance)
        com_positions = [s.center_of_mass for s in recent if s.center_of_mass is not None]
        is_still = True
        if len(com_positions) >= 3:
            ys = [p[1] for p in com_positions]
            y_variance = max(ys) - min(ys)
            # Low variance = very still (normalized coordinates, so small numbers)
            is_still = y_variance < 0.05

        if not is_still:
            # Person is down but moving - they might be getting up
            # Still give some score but reduced
            return min(0.5, time_down / stillness_threshold * 0.5)

        # Person is down AND still
        score = min(1.0, time_down / stillness_threshold)
        return score

    # -------------------------------------------------------------------------
    # State Machine
    # -------------------------------------------------------------------------

    def _update_state(self, timestamp: float):
        """Update MONITORING -> CAUTION -> ALERT -> ALARM state."""
        state = self.fall_state
        
        # If Resting, no alarm progression
        if state.state == "RESTING":
            return

        score = state.score

        if score >= config.CRITICAL_SCORE:
            # Instant alarm for very high scores
            state.state = "ALARM"
            state.alarm_active = True
            return

        if score >= config.FALL_SCORE_THRESHOLD:
            if state.state in ("MONITORING", "CAUTION"):
                state.state = "ALERT"
                state.alert_start_time = state.alert_start_time or timestamp

            elif state.state == "ALERT":
                alert_duration = timestamp - (state.alert_start_time or timestamp)
                if alert_duration >= config.ALARM_DURATION_SECONDS:
                    state.state = "ALARM"
                    state.alarm_active = True
        else:
            # Score dropped below threshold
            if state.state in ("ALERT", "CAUTION"):
                state.state = "MONITORING"
                state.alert_start_time = None
            elif state.state == "ALARM":
                if score < config.FALL_SCORE_THRESHOLD * 0.5:
                    state.state = "MONITORING"
                    state.alarm_active = False
                    state.alert_start_time = None
                    self.fall_state.last_descent_detected = None

    # -------------------------------------------------------------------------
    # Result
    # -------------------------------------------------------------------------

    def _make_result(self, **details) -> dict:
        """Create result dict for external consumption."""
        return {
            'score': round(self.fall_state.score, 3),
            'state': self.fall_state.state,
            'alarm_active': self.fall_state.alarm_active,
            'details': {k: round(v, 3) for k, v in details.items()},
        }
