"""
Fall Detection - Hybrid Version
GStreamer app z Flask web server
"""
import os
import sys
import time
import threading
import numpy as np

# Dodaj hailo-apps do path
HAILO_APPS_PATH = os.path.join(os.path.dirname(__file__), "..", "hailo-apps")
sys.path.append(os.path.join(HAILO_APPS_PATH, "hailo_apps", "python"))

# Import pozostałych modułów
import config
from utils.alarm import trigger_alarm, clear_alarm

# Ustaw środowisko
os.environ["TAPPAS_POSTPROC_PATH"] = config.TAPPAS_POSTPROC_PATH
os.environ["GST_DEBUG"] = "2"  # Mniej logów

# Import GStreamer
from core.gstreamer.gstreamer_app import app_callback_class
from pipeline_apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp
import hailo

# Flask imports
from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
from datetime import datetime


# ============================================================================
# Shared State - między GStreamer a Flask
# ============================================================================

class SharedFrameBuffer:
    """Thread-safe buffer dla klatek video."""
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


# ============================================================================
# Fall Detection Logic (z oryginalnego fall_detection.py)
# ============================================================================

def detect_fall(bbox, keypoints):
    """Wykrywa czy osoba upadła."""
    def get_keypoint(idx):
        point = keypoints[idx]
        if point.confidence() > config.SCORE_THRESHOLD:
            x = int((point.x() * bbox.width() + bbox.xmin()) * config.WIDTH)
            y = int((point.y() * bbox.height() + bbox.ymin()) * config.HEIGHT)
            return (x, y)
        return None

    bbox_width = bbox.width() * config.WIDTH
    bbox_height = bbox.height() * config.HEIGHT
    is_horizontal = bbox_width > bbox_height * config.FALL_ASPECT_RATIO

    keypoints_fall = False
    left_hip = get_keypoint(config.KEYPOINT_LEFT_HIP)
    right_hip = get_keypoint(config.KEYPOINT_RIGHT_HIP)
    left_shoulder = get_keypoint(config.KEYPOINT_LEFT_SHOULDER)
    right_shoulder = get_keypoint(config.KEYPOINT_RIGHT_SHOULDER)

    if left_hip and right_hip and left_shoulder and right_shoulder:
        avg_hip_y = (left_hip[1] + right_hip[1]) / 2
        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        y_difference = abs(avg_hip_y - avg_shoulder_y)
        keypoints_fall = y_difference < config.HEIGHT * config.KEYPOINT_PROXIMITY_THRESHOLD

    return is_horizontal or keypoints_fall


class FallDetectionLogic(app_callback_class):
    """Logika wykrywania upadków z WebSocket support."""
    def __init__(self, socketio_instance):
        super().__init__()
        self.fall_start_time = None
        self.alarm_active = False
        self.socketio = socketio_instance
        self.fall_detected = False
        self.fall_duration = 0.0
        self.last_update = datetime.now()
        self.lock = threading.Lock()
    
    def get_status(self):
        with self.lock:
            return {
                'fall_detected': self.fall_detected,
                'alarm_active': self.alarm_active,
                'fall_duration': round(self.fall_duration, 2),
                'timestamp': self.last_update.isoformat()
            }


def process_frame(element, buffer, user_data: FallDetectionLogic):
    """Callback dla każdej klatki - ZAPISUJE do frame_buffer."""
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    fall_detected_in_frame = False

    for detection in detections:
        if detection.get_label() != "person":
            continue

        bbox = detection.get_bbox()
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue

        keypoints = landmarks[0].get_points()

        if detect_fall(bbox, keypoints):
            fall_detected_in_frame = True
            break

    current_time = time.time()

    with user_data.lock:
        if fall_detected_in_frame:
            if user_data.fall_start_time is None:
                user_data.fall_start_time = current_time
                user_data.fall_detected = True
                print("[INFO] Możliwy upadek - obserwuję...")

            user_data.fall_duration = current_time - user_data.fall_start_time

            if user_data.fall_duration >= config.ALARM_DURATION_SECONDS and not user_data.alarm_active:
                user_data.alarm_active = True
                trigger_alarm()
                user_data.socketio.emit('alarm_triggered', {
                    'timestamp': datetime.now().isoformat(),
                    'duration': user_data.fall_duration
                })
        else:
            if user_data.alarm_active:
                clear_alarm()
                user_data.socketio.emit('alarm_cleared', {
                    'timestamp': datetime.now().isoformat()
                })
            user_data.fall_start_time = None
            user_data.alarm_active = False
            user_data.fall_detected = False
            user_data.fall_duration = 0.0
        
        user_data.last_update = datetime.now()
    
    # KEY STEP: Extract frame and DRAW pose skeleton for web
    try:
        import gi
        import cv2
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            try:
                # Extract raw frame
                frame_data = np.ndarray(
                    shape=(config.HEIGHT, config.WIDTH, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                frame = frame_data.copy()
                
                # DRAW POSE SKELETON on frame (same as hailooverlay)
                for detection in detections:
                    if detection.get_label() != "person":
                        continue

                    bbox = detection.get_bbox()
                    landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
                    if not landmarks:
                        continue

                    keypoints = landmarks[0].get_points()
                    
                    # COCO skeleton connections
                    skeleton = [
                        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                        (5, 6),  # Shoulders
                        (5, 7), (7, 9),  # Left arm
                        (6, 8), (8, 10),  # Right arm
                        (5, 11), (6, 12),  # Torso
                        (11, 12),  # Hips
                        (11, 13), (13, 15),  # Left leg
                        (12, 14), (14, 16),  # Right leg
                    ]
                    
                    # Collect points
                    points = []
                    for i in range(17):  # COCO format - 17 keypoints
                        try:
                            kp = keypoints[i]
                            if kp.confidence() > config.SCORE_THRESHOLD:
                                x = int((kp.x() * bbox.width() + bbox.xmin()) * config.WIDTH)
                                y = int((kp.y() * bbox.height() + bbox.ymin()) * config.HEIGHT)
                                points.append((x, y))
                            else:
                                points.append(None)
                        except:
                            points.append(None)
                    
                    # Draw lines (CYAN - like GStreamer)
                    # GStreamer frame is RGB, cv2 expects BGR, so Cyan = (0, 255, 255) in RGB
                    for conn in skeleton:
                        if conn[0] < len(points) and conn[1] < len(points):
                            pt1, pt2 = points[conn[0]], points[conn[1]]
                            if pt1 and pt2:
                                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)  # Cyan in RGB
                    
                    # Draw points (MAGENTA - like GStreamer)
                    # Magenta = (255, 0, 255) in RGB
                    for pt in points:
                        if pt:
                            cv2.circle(frame, pt, 4, (255, 0, 255), -1)  # Magenta in RGB
                            cv2.circle(frame, pt, 5, (255, 255, 255), 1)  # White border
                
                # Add status overlay
                status = user_data.get_status()
                if status['alarm_active']:
                    color = (0, 0, 255)  # Red
                    text = "ALARM! Fall detected!"
                elif status['fall_detected']:
                    color = (0, 165, 255)  # Orange
                    text = f"Observing... {status['fall_duration']}s"
                else:
                    color = (0, 255, 0)  # Green
                    text = "Monitoring active"
                
                cv2.rectangle(frame, (5, 5), (config.WIDTH - 5, 35), (0, 0, 0), -1)
                cv2.rectangle(frame, (5, 5), (config.WIDTH - 5, 35), color, 2)
                cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Zapisz do shared buffer (z nałożonym skeleton!)
                frame_buffer.set_frame(frame)
                
            finally:
                buffer.unmap(map_info)
    except Exception as e:
        print(f"[DEBUG] Exception in frame capture: {e}")

    return True


# ============================================================================
# Flask Web Server
# ============================================================================

app = Flask(__name__, static_folder='static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global detection state
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
    """Generator klatek z shared buffer."""
    frame_count = 0
    while True:
        frame = frame_buffer.get_frame()
        
        if frame is not None:
            frame_count += 1
            if frame_count % 30 == 0:  # Co sekundę
                print(f"[DEBUG] Sending frame {frame_count} to web client")
            
            # Konwertuj BGR -> RGB (GStreamer używa RGB, OpenCV BGR)
            # WAIT - GStreamer już jest RGB, cv2.imencode oczekuje BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            ret, buffer = cv2.imencode('.jpg', frame_bgr,
                                      [cv2.IMWRITE_JPEG_QUALITY, config.MJPEG_QUALITY])
            
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            if frame_count % 30 == 0:
                print("[DEBUG] No frame in buffer yet...")
        
        time.sleep(1.0 / config.MJPEG_FPS)


@app.route('/api/status')
def get_status():
    if detection_logic:
        return jsonify(detection_logic.get_status())
    return jsonify({'error': 'not ready'})


@socketio.on('connect')
def handle_connect():
    print('[INFO] Client connected')
    if detection_logic:
        emit('status', detection_logic.get_status())


@socketio.on('disconnect')
def handle_disconnect():
    print('[INFO] Client disconnected')


# ============================================================================
# Main
# ============================================================================

def main():
    global detection_logic
    
    print("=" * 60)
    print("   FALL DETECTION - HYBRID WEB APP")
    print("=" * 60)
    print(f"Alarm threshold: {config.ALARM_DURATION_SECONDS}s")
    print(f"Web server: http://{config.WEB_HOST}:{config.WEB_PORT}")
    print("=" * 60)
    print()
    
    # Inicjalizuj detection logic z socketio
    detection_logic = FallDetectionLogic(socketio)
    
    # Uruchom Flask w osobnym wątku
    def run_flask():
        socketio.run(app,
                    host=config.WEB_HOST,
                    port=config.WEB_PORT,
                    debug=False,
                    allow_unsafe_werkzeug=True,
                    use_reloader=False)
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    print("[INFO] Flask uruchomiony, startowanie GStreamer...")
    time.sleep(1)
    
    # KLUCZOWE: Ustaw sys.argv PRZED utworzeniem GStreamer app!
    sys.argv = [
        "fall_detection_web.py",
        "--input", "rpi",
        "--hef-path", "/usr/share/hailo-models/yolov8s_pose_h10.hef",
        "--width", "640",
        "--height", "480",
        "--disable-sync",
    ]
    
    # Uruchom GStreamer app (z lokalnym displayem + kamerą RPI)
    gst_app = GStreamerPoseEstimationApp(process_frame, detection_logic)
    gst_app.run()


if __name__ == "__main__":
    main()
