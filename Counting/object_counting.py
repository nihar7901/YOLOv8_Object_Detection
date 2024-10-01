from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/cars.mp4")

model = YOLO("../YOLO-weights/yolov8n.pt")

classNames = ["Boy", "Girl", "car", "bicycle", "motorbike", "bus", "truck", "train", "boat", "aeroplane",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat"
              ]

mask = cv2.imread("mask.png")
# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
#totalCount = 0
totalCount = []

while True:
    success, img = cap.read()
    #to make both(img and mask) the sizes same
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion,stream=True)

    detections = np.empty((0, 5))


    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            print(x1,y1,w,h)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)

            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)

            cls = int(box.cls[0])
            currentclass = classNames[cls]

            if currentclass == "car" and conf>0.4:
                cvzone.putTextRect(img, f'{currentclass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                   offset=5)
                #cvzone.cornerRect(img, (x1, y1, w, h), l=10,rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-10 < cy < limits[1]+10:
            #totalCount+=1
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    #cvzone.putTextRect(img, f' Count: {totalCount}', (50, 50))
    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))


    cv2.imshow("Image",img)
    #cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
