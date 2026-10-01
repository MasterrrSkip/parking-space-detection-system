# Smart Parking Occupancy Detector (YOLOv8 + OpenCV)

A comprehensive real-world parking lot analysis system that detects **BOTH OCCUPIED (Red)** and **VACANT (Green)** parking spaces in images, drone footage, CCTV feeds, and video streams.

## Features
- **100% Full-Bay Detection**: Ensures every marked bay is mapped and evaluated (no missing empty spaces).
- **YOLOv8 Deep Learning**: Detects cars, SUVs, trucks, and vans with confidence filtering.
- **Intersection-over-Slot (IoU)**: Evaluates vehicle occupancy inside designated parking bays.
- **Color Coded**:
  - 🔴 **RED**: Occupied parking space
  - 🟢 **GREEN**: Vacant / Available parking space
- **Visual Fine-Tuning & JSON Export**: Edit slot coordinates in the web app and load `parking_slots.json` directly into Python.
- **Works on Images & Videos**: Analyze static photos (`.jpg`, `.png`) or video footage (`.mp4`, `.avi`, webcam).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python smart_parking_yolo.py --image image.png --slots parking_slots.json
```
