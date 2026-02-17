# Fall Detection System for Raspberry Pi 5 + Hailo AI HAT+

AI-powered fall detection system using pose estimation on Raspberry Pi 5 with Hailo-10H accelerator (40 TOPS). Features real-time monitoring with web interface and alarm notifications.

![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-C51A4A?logo=raspberrypi)
![AI](https://img.shields.io/badge/AI-Hailo--10H-00ADD8)
![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)

## 🎯 Features

- **Real-time pose estimation** using YOLOv8 Pose on Hailo-10H NPU (40 TOPS)
- **Fall detection algorithm** based on body orientation and keypoint analysis
- **Web interface** for remote monitoring (Flask + WebSocket)
- **Live video stream** with pose skeleton visualization (cyan lines + magenta keypoints)
- **Alarm system** with configurable duration threshold
- **Dual output**: Local display (Wayland) + browser stream (MJPEG)



## 📋 Prerequisites

### Hardware
- **Raspberry Pi 5** (4GB+ RAM recommended)
- **Raspberry Pi AI HAT+** (Hailo-10H, 40 TOPS variant)
- **Camera**:
  - Tested with: **ArduCAM 12MP IMX708 Wide-Angle** (102° FOV)
  - Compatible with: Any **Libcamera-supported** camera (e.g., RPi Camera v2, v3, HQ)
- Display for local monitoring (optional, Wayland required)

### Software
- **Raspberry Pi OS Bookworm** (64-bit)
- **Hailo firmware** and drivers installed
- **Python 3.13+**

## 🚀 Installation

### 1. Configure Hailo-10H

Follow the official Raspberry Pi documentation to set up Hailo:
- [Raspberry Pi AI Kit Documentation](https://www.raspberrypi.com/documentation/computers/ai.html)

Verify Hailo is working:
```bash
hailortcli fw-control identify
```

### 2. Install hailo-apps

This project depends on [hailo-apps](https://github.com/hailo-ai/hailo-apps) for pose estimation pipeline:

```bash
cd ~/Documents/hailoProjects
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
./install.sh
source setup_env.sh
```

Create and activate virtual environment:
```bash
python3 -m venv venv_hailo_apps
source venv_hailo_apps/bin/activate
pip install -r requirements.txt
```

### 3. Install Fall Detection App

Clone this repository:
```bash
cd ~/Documents/hailoProjects
git clone <your-repo-url> fall-detection-app
cd fall-detection-app
```

Install Python dependencies:
```bash
# Activate hailo-apps venv
source ../hailo-apps/venv_hailo_apps/bin/activate

# Install web dependencies
pip install flask flask-cors flask-socketio python-socketio opencv-python
```

### 4. Download Pose Model

The application uses YOLOv8s Pose model optimized for Hailo-10H:
```bash
# Model should be available at:
ls /usr/share/hailo-models/yolov8s_pose_h10.hef
```

If not present, download from Hailo Model Zoo.

## ⚙️ Configuration

Edit `config.py` to customize settings:

```python
# Camera resolution
WIDTH = 640
HEIGHT = 480

# Fall detection thresholds
ALARM_DURATION_SECONDS = 1.5  # Time before triggering alarm
FALL_ASPECT_RATIO = 1.3       # Width/height ratio threshold
KEYPOINT_PROXIMITY_THRESHOLD = 0.15  # Hip-shoulder vertical distance

# Web server
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
MJPEG_FPS = 30
MJPEG_QUALITY = 85

# Model path
DEFAULT_HEF_PATH = "/usr/share/hailo-models/yolov8s_pose_h10.hef"
```

## 🎮 Usage

### Web Application (Recommended)

Run the hybrid app with local display + web interface:

```bash
cd ~/Documents/hailoProjects/hailo-apps
source venv_hailo_apps/bin/activate
python3 ../fall-detection-app/fall_detection_web.py
```

Access web interface:
- Local: `http://localhost:5000`
- Network: `http://<raspberry-pi-ip>:5000`

### Desktop Application (Local Only)

Run GStreamer app with Wayland display:

```bash
cd ~/Documents/hailoProjects/hailo-apps
source venv_hailo_apps/bin/activate
python3 ../fall-detection-app/fall_detection.py
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Raspberry Pi Camera                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              GStreamer Pipeline (picamera2)             │
│  ┌─────────┐    ┌──────────┐    ┌────────────────────┐  │
│  │ Source  │ -> │ Hailo-10H│ -> │ Pose Estimation    │  |
│  │         │    │ Inference│    │ (YOLOv8 Pose)      │  │
│  └─────────┘    └──────────┘    └────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌──────────────────┐        ┌──────────────────┐
│  Local Display   │        │  Flask Callback  │
│  (hailooverlay)  │        │  • Draw skeleton │
│  • Wayland sink  │        │  • Detect falls  │
└──────────────────┘        │  • Save frames   │
                            └────────┬─────────┘
                                     │
                            ┌────────▼─────────┐
                            │  Web Interface   │
                            │  • MJPEG stream  │
                            │  • WebSocket     │
                            │  • Alarm UI      │
                            └──────────────────┘
```

##  Fall Detection Algorithm

The system detects falls using two criteria:

1. **Bounding Box Orientation**
   - Calculates width/height ratio of detected person
   - Fall detected if `width > height × FALL_ASPECT_RATIO`

2. **Keypoint Analysis**
   - Compares vertical position of hips vs shoulders
   - Fall detected if distance < `KEYPOINT_PROXIMITY_THRESHOLD`

Alarm triggers after fall persists for `ALARM_DURATION_SECONDS`.

## 🎨 Web Interface

- **Status Indicator**: Green (monitoring) / Orange (observing) / Red (alarm)
- **Live Video**: MJPEG stream with pose skeleton overlay
- **Real-time Notifications**: WebSocket-based alarm alerts
- **Responsive Design**: Works on desktop, tablet, mobile

## 🐛 Troubleshooting

### Camera not detected
```bash
# Check camera connection
libcamera-hello --list-cameras
```

### Hailo not working
```bash
# Verify firmware
hailortcli fw-control identify

# Check device
ls /dev/hailo*
```

### Missing environment variables
```bash
# Source hailo-apps environment
cd ~/Documents/hailoProjects/hailo-apps
source setup_env.sh
```

### Web stream shows test video
Ensure `--input rpi` is set in `sys.argv` (should be automatic in `fall_detection_web.py`)

## 📜 License

**This project**: Copyright © 2026 - All Rights Reserved  
See [LICENSE](LICENSE) file for details.

**Third-party components**: This project uses [hailo-apps](https://github.com/hailo-ai/hailo-apps) (MIT License).  
See [NOTICE](NOTICE) file for full third-party software attributions.

## 🙏 Acknowledgments

- **Hailo AI** for Hailo-10H NPU and hailo-apps framework
- **Raspberry Pi Foundation** for hardware platform and AI kit
- **Ultralytics** for YOLOv8 Pose model

## 📧 Support

For issues related to:
- **Hailo setup**: See [Raspberry Pi AI documentation](https://www.raspberrypi.com/documentation/computers/ai.html)
- **hailo-apps**: Visit [hailo-apps repository](https://github.com/hailo-ai/hailo-apps)
- **This project**: Open an issue in this repository

## 🚧 Development

### Running in development mode

```bash
# Enable debug logging
export GST_DEBUG=2

# Run with verbose output
python3 fall_detection_web.py
```

### Testing fall detection

Simulate a fall by:
1. Lying horizontally in camera view
2. Positioning body so hips are level with shoulders
3. Alarm should trigger after configured duration

---

**Built with ❤️ for Raspberry Pi 5 + Hailo-10H**
