"""
Configuration for Fall Detection App
"""

# Video Parameters (processing resolution)
# Input camera resolution might differ (will be resized)
# For ArduCAM IMX708 and RPi Cam, 640x480 balances performance/quality
WIDTH = 640
HEIGHT = 480

# Fall Detection Logic
SCORE_THRESHOLD = 0.3  # Minimum keypoint confidence
FALL_ASPECT_RATIO = 1.3  # BBox width/height ratio (horizontal = fall)
KEYPOINT_PROXIMITY_THRESHOLD = 0.15  # Hip-shoulder vertical distance threshold

# Alarms
ALARM_DURATION_SECONDS = 1.5  # Duration of fall to trigger alarm

# Hailo Model Paths
DEFAULT_HEF_PATH = "/usr/share/hailo-models/yolov8s_pose_h10.hef"

# Post-processing Library Path
TAPPAS_POSTPROC_PATH = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes"

# Keypoint indices (COCO format)
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12

# Web server settings
WEB_HOST = "0.0.0.0"  # Listen on all interfaces
WEB_PORT = 5000
MJPEG_FPS = 15  # Target FPS for web stream
MJPEG_QUALITY = 85  # JPEG compression quality (1-100)
