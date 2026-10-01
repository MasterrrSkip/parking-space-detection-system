#!/usr/bin/env bash
echo "=========================================="
echo " Smart Parking Occupancy Detector (YOLOv8)"
echo "=========================================="
pip install -r requirements.txt
python3 smart_parking_yolo.py "$@"
