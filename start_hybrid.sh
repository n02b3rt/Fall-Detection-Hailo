#!/bin/bash
# Start Hybrid Fall Detection Web App

cd ~/Documents/hailoProjects/fall-detection-app

echo "===================================="
echo "Fall Detection - Hybrid Web App"
echo "===================================="
echo ""

# Activate environment
source ../hailo-apps/venv_hailo_apps/bin/activate

# Set environment from hailo-apps
cd ../hailo-apps
export $(grep -v '^#' setup_env.sh | xargs)
cd ../fall-detection-app

# Run hybrid app (GStreamer + Flask)
python3 fall_detection_web.py
