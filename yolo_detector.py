import cv2
from ultralytics import YOLO
from collections import deque

model = YOLO("yolov8n.pt")
video_path = "CatZoomies.mp4"
my_name = "Metasit Clicknext-Internship-2024"
points = deque(maxlen=30)

def draw_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        color = (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, "cat", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        points.append((cx, cy))
    
    return frame

def detect_object(frame):
    results = model.predict(frame, verbose=False, stream=True)
    cat_boxes = []

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 15:
                cat_boxes.append(box)

    for i in range(1, len(points)):
        if points[i-1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i-1], points[i], (0, 0, 255), 2)

    if cat_boxes:
        frame = draw_boxes(frame, cat_boxes)
        
    return frame

def add_name(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = cv2.getTextSize(my_name, font, 0.7, 2)[0]
    
    x = frame.shape[1] - size[0] - 10
    y = 50
    
    cv2.putText(frame, my_name, (x, y), font, 0.7, (0, 0, 255), 2)
    
    return frame

def main():
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open video")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_object(frame)
        frame = add_name(frame)

        cv2.imshow("Result", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()