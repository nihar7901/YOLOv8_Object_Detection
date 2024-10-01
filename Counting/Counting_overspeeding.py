from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np

cap = cv2.VideoCapture("../Videos/cars.mp4")

model = YOLO("../YOLO-weights/yolov8n.pt")

classNames = ["Boy", "Girl", "car", "bicycle", "motorbike", "bus", "truck", "train", "boat", "aeroplane",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat"]

mask = cv2.imread("mask.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
totalCount = []

prev_positions = {}
fps = cap.get(cv2.CAP_PROP_FPS)  # Frame rate of the video
pixel_to_meter_ratio = 0.05  # Placeholder ratio

while True:
    success, img = cap.read()
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentclass = classNames[cls]

            if currentclass == "car" and conf > 0.4:
                cvzone.putTextRect(img, f'{currentclass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if id in prev_positions:
            prev_cx, prev_cy = prev_positions[id]
            pixel_distance = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
            real_world_distance = pixel_distance * pixel_to_meter_ratio
            speed_m_per_sec = real_world_distance * fps
            speed_kmh = speed_m_per_sec * 3.6
            cvzone.putTextRect(img, f'Speed: {int(speed_kmh)} km/h', (x1, y1 - 20), scale=1, thickness=1, offset=5)

            # Only count cars moving faster than 50 km/h
            if speed_kmh > 50:
                if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[1] + 10:
                    if totalCount.count(id) == 0:
                        totalCount.append(id)
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        prev_positions[id] = (cx, cy)

    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
