"""
Configuration for Fall Detection App
"""

# Video Parameters (processing resolution)
WIDTH = 640
HEIGHT = 480

# =============================================================================
# Fall Detection - Keypoint Analysis
# =============================================================================
# Fall Detection Thresholds
FALL_SCORE_THRESHOLD_ALERT = 0.6  # Score to trigger ALERT state
FALL_SCORE_THRESHOLD_ALARM = 0.8  # Score to trigger ALARM state
FALL_ASPECT_RATIO = 1.2  # bbox width/height > 1.2 considered horizontal
VERTICAL_ANGLE_THRESHOLD = 45.0  # degrees from vertical
TORSO_COMPRESSION_THRESHOLD = 0.6 # Ratio of torso length / shoulder width (End-on fall) distance (normalized)

# Keypoint indices (COCO format)
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12

# =============================================================================
# Fall Detection - Temporal Analysis
# =============================================================================

HISTORY_WINDOW_SECONDS = 2.0       # Sliding window for velocity computation
COM_VELOCITY_THRESHOLD = 40        # pixels/sec - low to catch slow/gradual falls
TORSO_ANGLE_VELOCITY_THRESHOLD = 30  # degrees/sec
MIN_FALL_DESCENT_RATIO = 0.15     # Min CoM drop as fraction of standing bbox height
POST_FALL_STILLNESS_TIME = 2.0    # Seconds person must stay down
MIN_VISIBLE_KEYPOINTS = 4         # Minimum for full keypoint analysis
STANDING_HEIGHT_FRAMES = 90       # Frames to calibrate standing height

# Scoring weights (biased toward detection - never miss a real fall)
WEIGHT_VELOCITY = 0.25            # Rapid descent
WEIGHT_POSE = 0.20                # Horizontal/collapsed pose
WEIGHT_DESCENT = 0.25             # How far CoM dropped from standing
WEIGHT_STILLNESS = 0.30           # Post-fall inactivity (strongest signal)
FALL_SCORE_THRESHOLD = 0.5        # Score to trigger alert

# Alarm escalation
ALARM_DURATION_SECONDS = 2.0      # Sustained alert before full alarm
CRITICAL_SCORE = 0.8              # Instant alarm (no waiting period)

# Disappearance tracking
PERSON_GONE_CAUTION_TIME = 1.5    # Seconds before CAUTION (person vanished)
PERSON_GONE_ALERT_TIME = 5.0      # Seconds before escalating to ALERT

# =============================================================================
# Hailo Model
# =============================================================================

DEFAULT_HEF_PATH = "/usr/share/hailo-models/yolov8s_pose_h10.hef"
TAPPAS_POSTPROC_PATH = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes"

# =============================================================================
# Web Server
# =============================================================================

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
MJPEG_FPS = 15
MJPEG_QUALITY = 85
