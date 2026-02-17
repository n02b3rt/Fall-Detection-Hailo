"""
Fall Detection Application using Hailo-10H AI Accelerator
Aplikacja do wykrywania upadków dla konkursu Hailo
"""
import os
import sys
import time

# Konfiguracja ścieżek do hailo-apps
HAILO_APPS_PATH = os.path.join(os.path.dirname(__file__), "..", "hailo-apps")
sys.path.append(os.path.join(HAILO_APPS_PATH, "hailo_apps", "python"))

# Import konfiguracji
import config
from utils.alarm import trigger_alarm, clear_alarm

# Ustawienie zmiennych środowiskowych
os.environ["TAPPAS_POSTPROC_PATH"] = config.TAPPAS_POSTPROC_PATH

# Importy z hailo-apps
from core.gstreamer.gstreamer_app import app_callback_class
from core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE, INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE, USER_CALLBACK_PIPELINE, DISPLAY_PIPELINE,
)
from pipeline_apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp
import hailo


class FallDetectionLogic(app_callback_class):
    """
    Logika aplikacji do wykrywania upadków.
    Śledzi stan upadku i zarządza alarmami.
    """
    def __init__(self):
        super().__init__()
        self.fall_start_time = None  # Timestamp pierwszego wykrycia upadku
        self.alarm_active = False    # Czy alarm jest aktywnie wyzwolony


class FallDetectionApp(GStreamerPoseEstimationApp):
    """
    Główna aplikacja GStreamer do wykrywania upadków.
    """
    def get_pipeline_string(self):
        """Buduje pipeline GStreamer dla detekcji pose i trackingu."""
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


def detect_fall(bbox, keypoints):
    """
    Wykrywa czy osoba upadła na podstawie bounding box i keypoints.
    
    Kryteria wykrywania:
    1. Bounding box jest poziomy (szerszy niż wyższy)
    2. Biodra są na podobnym poziomie co ramiona (mała różnica Y)
    
    Args:
        bbox: Hailo detection bounding box
        keypoints: Lista Hailo keypoints (COCO format)
        
    Returns:
        bool: True jeśli wykryto upadek
    """
    def get_keypoint(idx):
        """Pobiera współrzędne keypointa jeśli pewność > threshold."""
        point = keypoints[idx]
        if point.confidence() > config.SCORE_THRESHOLD:
            x = int((point.x() * bbox.width() + bbox.xmin()) * config.WIDTH)
            y = int((point.y() * bbox.height() + bbox.ymin()) * config.HEIGHT)
            return (x, y)
        return None

    # Kryterium 1: Proporcje bounding box
    bbox_width = bbox.width() * config.WIDTH
    bbox_height = bbox.height() * config.HEIGHT
    is_horizontal = bbox_width > bbox_height * config.FALL_ASPECT_RATIO

    # Kryterium 2: Biodra na poziomie ramion
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


def process_frame(element, buffer, user_data: FallDetectionLogic):
    """
    Callback wywoływany dla każdej klatki wideo.
    Analizuje detekcje i zarządza stanem alarmu.
    
    Args:
        element: GStreamer element
        buffer: GStreamer buffer z danymi
        user_data: Instancja FallDetectionLogic
        
    Returns:
        bool: True aby kontynuować przetwarzanie
    """
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    fall_detected_in_frame = False

    # Sprawdź każdą wykrytą osobę
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
            break  # Wystarczy jedna osoba w upadku

    current_time = time.time()

    if fall_detected_in_frame:
        if user_data.fall_start_time is None:
            # Pierwsze wykrycie - rozpocznij licznik
            user_data.fall_start_time = current_time
            print("[INFO] Możliwy upadek - obserwuję...")

        fall_duration = current_time - user_data.fall_start_time

        # Jeśli upadek trwa wystarczająco długo, wyzwól alarm
        if fall_duration >= config.ALARM_DURATION_SECONDS and not user_data.alarm_active:
            user_data.alarm_active = True
            trigger_alarm()

    else:
        # Brak upadku - resetuj stan
        if user_data.alarm_active:
            clear_alarm()
        user_data.fall_start_time = None
        user_data.alarm_active = False

    return True


def main():
    """Główna funkcja aplikacji."""
    # Włącz debugowanie GStreamer dla video sinków
    os.environ["GST_DEBUG"] = "autovideosink:5,waylandsink:5,ximagesink:5,kmssink:5"
    os.environ["DISPLAY"] = ":0"  # Wymuszenie display dla X11
    
    # Konfiguracja argumentów wiersza poleceń
    sys.argv = [
        "fall_detection.py",
        "--input", "rpi",
        "--hef-path", config.DEFAULT_HEF_PATH,
        "--width", str(config.WIDTH),
        "--height", str(config.HEIGHT),
        "--disable-sync",
    ]

    # Inicjalizacja i uruchomienie
    logic = FallDetectionLogic()
    app = FallDetectionApp(process_frame, logic)
    
    print("=" * 60)
    print("   FALL DETECTION - Aplikacja do wykrywania upadków")
    print("=" * 60)
    print(f"Alarm zostanie wyzwolony po {config.ALARM_DURATION_SECONDS}s ciągłego upadku")
    print("Naciśnij Ctrl+C aby zakończyć\n")
    
    # Wyświetl pipeline dla debugowania
    print("[DEBUG] Pipeline string:")
    print(app.get_pipeline_string())
    print()
    
    app.run()


if __name__ == "__main__":
    main()
