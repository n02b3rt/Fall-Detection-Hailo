#!/bin/bash
# Quick Start Script for Fall Detection Web App

echo "================================"
echo "Fall Detection Web App - Setup"
echo "================================"
echo ""

# Activate hailo-apps virtual environment
echo "[1/3] Activating hailo-apps environment..."
source ~/Documents/hailoProjects/hailo-apps/venv_hailo_apps/bin/activate

# Install dependencies
echo "[2/3] Installing web dependencies..."
pip install flask flask-cors flask-socketio python-socketio

# Start web server
echo "[3/3] Starting web server..."
echo ""
echo "Web app will be available at: http://$(hostname -I | awk '{print $1}'):5000"
echo ""

python web_app.py
