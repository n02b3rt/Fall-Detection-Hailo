"""
Fall Detection - Hybrid Web Application
GStreamer pipeline with Flask web server and temporal fall detection.
"""
import os
import sys
import time
import threading
import numpy as np

# Add hailo-apps to path
HAILO_APPS_PATH = os.path.join(os.path.dirname(__file__), "..", "hailo-apps")
sys.path.append(os.path.join(HAILO_APPS_PATH, "hailo_apps", "python"))

# Project imports
import config
from fall_detector import FallDetector
from utils.alarm import trigger_alarm, clear_alarm

# Environment setup
os.environ["TAPPAS_POSTPROC_PATH"] = config.TAPPAS_POSTPROC_PATH
os.environ["GST_DEBUG"] = "2"

# GStreamer imports
from core.gstreamer.gstreamer_app import app_callback_class
from pipeline_apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp
import hailo

# Flask imports
from flask import Flask, Response, jsonify, send_from_directory, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
from datetime import datetime
import json


# =============================================================================
# Shared State
# =============================================================================

class SharedFrameBuffer:
    """Thread-safe buffer for video frames."""
    def __init__(self):
        self.latest_frame = None
        self.lock = threading.Lock()

    def set_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy() if frame is not None else None

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

# Global buffer
frame_buffer = SharedFrameBuffer()


# =============================================================================
# Fall Detection Logic
# =============================================================================

class FallDetectionLogic(app_callback_class):
    """Detection logic with WebSocket support and temporal analysis."""
    def __init__(self, socketio_instance):
        super().__init__()
        self.socketio = socketio_instance
        self.fall_detector = FallDetector()
        self.alarm_active = False
        self.last_result = {
            'score': 0.0, 'state': 'MONITORING',
            'alarm_active': False, 'details': {}
        }
        self.last_update = datetime.now()
        self.lock = threading.Lock()

    def get_status(self):
        with self.lock:
            return {
                'fall_detected': self.last_result['state'] != 'MONITORING',
                'alarm_active': self.last_result['alarm_active'],
                'fall_score': self.last_result['score'],
                'fall_state': self.last_result['state'],
                'fall_duration': 0.0,  # Kept for API compat
                'details': self.last_result.get('details', {}),
                'timestamp': self.last_update.isoformat()
            }


def process_frame(element, buffer, user_data: FallDetectionLogic):
    """Per-frame callback: runs fall detector and captures frame for web."""
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    timestamp = time.time()
    best_result = None

    # Run temporal fall detector on each person
    for detection in detections:
        if detection.get_label() != "person":
            continue

        bbox = detection.get_bbox()
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue

        keypoints = landmarks[0].get_points()
        result = user_data.fall_detector.update(bbox, keypoints, timestamp)

        # Track the highest-score result
        if best_result is None or result['score'] > best_result['score']:
            best_result = result

    # Handle alarm state transitions
    with user_data.lock:
        if best_result:
            prev_alarm = user_data.alarm_active
            user_data.last_result = best_result
            user_data.alarm_active = best_result['alarm_active']

            # Alarm just triggered
            if best_result['alarm_active'] and not prev_alarm:
                trigger_alarm()
                user_data.socketio.emit('alarm_triggered', {
                    'timestamp': datetime.now().isoformat(),
                    'score': best_result['score'],
                })

            # Alarm just cleared
            if not best_result['alarm_active'] and prev_alarm:
                clear_alarm()
                user_data.socketio.emit('alarm_cleared', {
                    'timestamp': datetime.now().isoformat()
                })
        else:
            # No person detected - track disappearance
            result = user_data.fall_detector.update_no_person(timestamp)
            prev_alarm = user_data.alarm_active
            user_data.last_result = result
            user_data.alarm_active = result['alarm_active']

            if result['alarm_active'] and not prev_alarm:
                trigger_alarm()
                user_data.socketio.emit('alarm_triggered', {
                    'timestamp': datetime.now().isoformat(),
                    'score': result['score'],
                    'reason': 'person_disappeared',
                })

            if not result['alarm_active'] and prev_alarm:
                clear_alarm()
                user_data.socketio.emit('alarm_cleared', {
                    'timestamp': datetime.now().isoformat()
                })

        user_data.last_update = datetime.now()

    # Capture frame and draw overlays for web stream
    _capture_and_draw(buffer, detections, user_data)

    return True


def _capture_and_draw(buffer, detections, user_data):
    """Extract frame, draw pose skeleton and status overlay for web."""
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return

        try:
            frame = np.ndarray(
                shape=(config.HEIGHT, config.WIDTH, 3),
                dtype=np.uint8,
                buffer=map_info.data
            ).copy()

            # Draw safe zones (if any)
            _draw_zones(frame, user_data.fall_detector.get_zones())

            # Draw pose skeleton
            _draw_skeleton(frame, detections)

            # Draw fall detection status overlay
            _draw_status_overlay(frame, user_data)

            frame_buffer.set_frame(frame)
        finally:
            buffer.unmap(map_info)
    except Exception as e:
        print(f"[ERROR] Frame capture: {e}")


def _draw_skeleton(frame, detections):
    """Draw pose skeleton on frame."""
    skeleton_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6),                            # Shoulders
        (5, 7), (7, 9),                    # Left arm
        (6, 8), (8, 10),                   # Right arm
        (5, 11), (6, 12),                  # Torso
        (11, 12),                          # Hips
        (11, 13), (13, 15),                # Left leg
        (12, 14), (14, 16),                # Right leg
    ]

    for detection in detections:
        if detection.get_label() != "person":
            continue

        bbox = detection.get_bbox()
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue

        keypoints = landmarks[0].get_points()
        points = []

        for i in range(17):
            try:
                kp = keypoints[i]
                if kp.confidence() > config.SCORE_THRESHOLD:
                    x = int((kp.x() * bbox.width() + bbox.xmin()) * config.WIDTH)
                    y = int((kp.y() * bbox.height() + bbox.ymin()) * config.HEIGHT)
                    points.append((x, y))
                else:
                    points.append(None)
            except (IndexError, AttributeError):
                points.append(None)

        # Draw lines (cyan)
        for c0, c1 in skeleton_connections:
            if c0 < len(points) and c1 < len(points):
                pt1, pt2 = points[c0], points[c1]
                if pt1 and pt2:
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        # Draw keypoints (magenta with white border)
        for pt in points:
            if pt:
                cv2.circle(frame, pt, 4, (255, 0, 255), -1)
                cv2.circle(frame, pt, 5, (255, 255, 255), 1)


def _draw_zones(frame, zones):
    """Draw configured safe zones on frame."""
    h, w = frame.shape[:2]
    
    # Draw Bed Zones (Blue)
    for zone in zones.get('bed', []):
        if len(zone) == 4:
            x1, y1, x2, y2 = [int(v * s) for v, s in zip(zone, [w, h, w, h])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "BED", (x1 + 5, y1 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Draw Door Zones (Green)
    for zone in zones.get('door', []):
        if len(zone) == 4:
            x1, y1, x2, y2 = [int(v * s) for v, s in zip(zone, [w, h, w, h])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "DOOR", (x1 + 5, y1 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def _draw_status_overlay(frame, user_data):
    """Draw fall detection score and state on frame."""
    status = user_data.get_status()
    score = status['fall_score']
    state = status['fall_state']
    details = status.get('details', {})

    # State color
    if state == "ALARM":
        state_color = (0, 0, 255)  # Red
    elif state == "ALERT":
        state_color = (0, 165, 255)  # Orange
    elif state == "CAUTION":
        state_color = (0, 255, 255)  # Yellow
    elif state == "RESTING":
        state_color = (255, 0, 0)    # Blue
    else:
        state_color = (0, 200, 0)  # Green

    h = config.HEIGHT
    w = config.WIDTH

    # Top bar: state text
    cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(frame, f"{state}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

    # Score value on right
    score_text = f"Score: {score:.2f}"
    cv2.putText(frame, score_text, (w - 160, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Score bar at bottom
    bar_y = h - 20
    bar_h = 12
    cv2.rectangle(frame, (8, bar_y), (w - 8, bar_y + bar_h), (40, 40, 40), -1)

    # Fill based on score
    fill_w = int((w - 16) * min(score, 1.0))
    if fill_w > 0:
        # Color gradient: green -> yellow -> red
        if score < 0.3:
            bar_color = (0, 200, 0)
        elif score < 0.5:
            bar_color = (0, 200, 200)
        elif score < 0.8:
            bar_color = (0, 165, 255)
        else:
            bar_color = (0, 0, 255)
        cv2.rectangle(frame, (8, bar_y), (8 + fill_w, bar_y + bar_h), bar_color, -1)

    # Threshold line
    thresh_x = 8 + int((w - 16) * config.FALL_SCORE_THRESHOLD)
    cv2.line(frame, (thresh_x, bar_y), (thresh_x, bar_y + bar_h), (255, 255, 255), 1)

    # Detail scores (small text, bottom-left)
    if details:
        y_offset = h - 30
        for key in ['velocity_score', 'pose_score', 'descent_score', 'stillness_score']:
            val = details.get(key, 0)
            short_key = key.replace('_score', '')[0].upper()
            cv2.putText(frame, f"{short_key}:{val:.1f}", (8, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
            y_offset -= 14


# =============================================================================
# Flask Web Server
# =============================================================================

app = Flask(__name__, static_folder='static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

detection_logic = None


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    """Frame generator for MJPEG stream."""
    while True:
        frame = frame_buffer.get_frame()

        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret, buf = cv2.imencode('.jpg', frame_bgr,
                                     [cv2.IMWRITE_JPEG_QUALITY, config.MJPEG_QUALITY])
            if ret:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

        time.sleep(1.0 / config.MJPEG_FPS)


@app.route('/api/status')
def get_status():
    if detection_logic:
        return jsonify(detection_logic.get_status())
    return jsonify({'error': 'not ready'})


@app.route('/api/zones', methods=['GET'])
def get_zones():
    if detection_logic:
        return jsonify(detection_logic.fall_detector.get_zones())
    return jsonify({'bed': [], 'door': []})


@app.route('/api/zones', methods=['POST'])
def save_zones():
    if not detection_logic:
        return jsonify({'error': 'not ready'}), 503
    
    try:
        zones = request.json
        detection_logic.fall_detector.save_zones(zones)
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@socketio.on('connect')
def handle_connect():
    print('[INFO] Client connected')
    if detection_logic:
        emit('status', detection_logic.get_status())


@socketio.on('disconnect')
def handle_disconnect():
    print('[INFO] Client disconnected')


# =============================================================================
# Main
# =============================================================================

def main():
    global detection_logic

    print("=" * 60)
    print("   FALL DETECTION - HYBRID WEB APP")
    print("   Temporal Analysis Engine v2.0")
    print("=" * 60)
    print(f"Alert threshold: {config.FALL_SCORE_THRESHOLD}")
    print(f"Alarm after: {config.ALARM_DURATION_SECONDS}s sustained")
    print(f"Instant alarm at score: {config.CRITICAL_SCORE}")
    print(f"Web server: http://{config.WEB_HOST}:{config.WEB_PORT}")
    print("=" * 60)
    print()

    detection_logic = FallDetectionLogic(socketio)

    def run_flask():
        socketio.run(app,
                    host=config.WEB_HOST,
                    port=config.WEB_PORT,
                    debug=False,
                    allow_unsafe_werkzeug=True,
                    use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    print("[INFO] Flask started, launching GStreamer...")
    time.sleep(1)

    # Set sys.argv BEFORE creating GStreamer app
    sys.argv = [
        "fall_detection_web.py",
        "--input", "rpi",
        "--hef-path", "/usr/share/hailo-models/yolov8s_pose_h10.hef",
        "--width", "640",
        "--height", "480",
        "--disable-sync",
    ]

    gst_app = GStreamerPoseEstimationApp(process_frame, detection_logic)
    gst_app.run()


if __name__ == "__main__":
    main()
