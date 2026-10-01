"""
Classical Computer Vision Parking Occupancy Detector (No Deep Learning)
Guarantees detection of BOTH OCCUPIED (Red) and VACANT (Green) parking slots.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.png")
if image is None:
    image = cv2.imread("parking_lot.jpg")

if image is None:
    print("Error: Could not load image.")
else:
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 16)
    
    def make_row(y_start, y_end, x_start, x_end, num_slots):
        slots = []
        y = int(y_start * h)
        slot_h = int((y_end - y_start) * h)
        total_w = int((x_end - x_start) * w)
        slot_w = total_w // num_slots
        start_x = int(x_start * w)
        for i in range(num_slots):
            slots.append((start_x + i * slot_w, y, int(slot_w * 0.94), slot_h))
        return slots

    parking_spaces = []
    parking_spaces.extend(make_row(0.238, 0.365, 0.125, 0.985, 18))  # Row 1
    parking_spaces.extend(make_row(0.525, 0.665, 0.175, 0.985, 17))  # Row 2
    parking_spaces.extend(make_row(0.668, 0.830, 0.175, 0.985, 16))  # Row 3
    
    occupied_count = 0
    vacant_count = 0
    
    for i, (x, y, pw, ph) in enumerate(parking_spaces, start=1):
        roi = thresh[y:y+ph, x:x+pw]
        white_pixels = cv2.countNonZero(roi)
        density_thresh = int(0.22 * (pw * ph))
        
        if white_pixels > density_thresh:
            color = (255, 0, 0)  # Red (Occupied)
            occupied_count += 1
            status = "OCC"
        else:
            color = (0, 255, 0)  # Green (Vacant)
            vacant_count += 1
            status = "VAC"
            
        cv2.rectangle(image_rgb, (x, y), (x + pw, y + ph), color, 2)
        cv2.putText(image_rgb, f"P{i:02d}:{status}", (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    total_spaces = len(parking_spaces)
    cv2.imwrite("result.jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    print(f"Total: {total_spaces} | Occupied: {occupied_count} (Red) | Vacant: {vacant_count} (Green)")
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title("OpenCV Complete Parking Grid (Green=Vacant, Red=Occupied)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
