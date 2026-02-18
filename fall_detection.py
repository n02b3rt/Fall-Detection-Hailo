"""
Fall Detection Application using Hailo-10H AI Accelerator
Standalone version with local display (no web server).
"""
import os
import sys
import time

# Add hailo-apps to path
HAILO_APPS_PATH = os.path.join(os.path.dirname(__file__), "..", "hailo-apps")
sys.path.append(os.path.join(HAILO_APPS_PATH, "hailo_apps", "python"))

# Project imports
import config
from fall_detector import FallDetector
from utils.alarm import trigger_alarm, clear_alarm

# Environment setup
os.environ["TAPPAS_POSTPROC_PATH"] = config.TAPPAS_POSTPROC_PATH

# GStreamer imports
from core.gstreamer.gstreamer_app import app_callback_class
from core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE, USER_CALLBACK_PIPELINE, DISPLAY_PIPELINE,
)
from pipeline_apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp
import hailo


class FallDetectionLogic(app_callback_class):
    """Detection logic with temporal fall analysis."""
    def __init__(self):
        super().__init__()
        self.fall_detector = FallDetector()
        self.alarm_active = False


class FallDetectionApp(GStreamerPoseEstimationApp):
    """GStreamer application for fall detection."""
    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.options_menu.input,
            video_width=self.options_menu.width,
            video_height=self.options_menu.height,
        )
        infer_pipeline = INFERENCE_PIPELINE(
            hef_path=self.options_menu.hef_path,
            post_process_so=self.post_process_so,
            batch_size=self.options_menu.batch_size,
        )
        infer_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(infer_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(class_id=0)
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(
            video_sink="waylandsink",
            sync="false",
            show_fps="false",
        )
        return (
            f"{source_pipeline} ! "
            f"{infer_pipeline_wrapper} ! "
            f"{tracker_pipeline} ! "
            f"{user_callback_pipeline} ! "
            f"{display_pipeline}"
        )


def process_frame(element, buffer, user_data: FallDetectionLogic):
    """Per-frame callback with temporal fall detection."""
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    timestamp = time.time()
    best_result = None

    for detection in detections:
        if detection.get_label() != "person":
            continue

        bbox = detection.get_bbox()
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue

        keypoints = landmarks[0].get_points()
        result = user_data.fall_detector.update(bbox, keypoints, timestamp)

        if best_result is None or result['score'] > best_result['score']:
            best_result = result

    else:
        # No person detected - track disappearance
        best_result = user_data.fall_detector.update_no_person(timestamp)
        prev_alarm = user_data.alarm_active
        user_data.alarm_active = best_result['alarm_active']

        if best_result['alarm_active'] and not prev_alarm:
            trigger_alarm()
            print(f"[ALARM] Fall detected (disappearance)! Score: {best_result['score']:.2f}")

        if not best_result['alarm_active'] and prev_alarm:
            clear_alarm()
            print("[INFO] Alarm cleared")

    # Periodic status log - now references best_result which is guaranteed to be set
    state = best_result['state']
    score = best_result['score']
    if state != "MONITORING" or int(timestamp) % 5 == 0:
        details = best_result.get('details', {})
        print(f"[{state}] Score: {score:.2f} | {details}")

    return True


def main():
    """Main function."""
    os.environ["GST_DEBUG"] = "autovideosink:5,waylandsink:5"
    os.environ["DISPLAY"] = ":0"

    sys.argv = [
        "fall_detection.py",
        "--input", "rpi",
        "--hef-path", config.DEFAULT_HEF_PATH,
        "--width", str(config.WIDTH),
        "--height", str(config.HEIGHT),
        "--disable-sync",
    ]

    logic = FallDetectionLogic()
    app = FallDetectionApp(process_frame, logic)

    print("=" * 60)
    print("   FALL DETECTION - Standalone")
    print("   Temporal Analysis Engine v2.0")
    print("=" * 60)
    print(f"Alert threshold: {config.FALL_SCORE_THRESHOLD}")
    print(f"Alarm after: {config.ALARM_DURATION_SECONDS}s sustained")
    print(f"Instant alarm at score: {config.CRITICAL_SCORE}")
    print("Press Ctrl+C to quit\n")

    app.run()


if __name__ == "__main__":
    main()
