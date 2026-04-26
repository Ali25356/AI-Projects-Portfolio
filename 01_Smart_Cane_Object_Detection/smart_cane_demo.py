import cv2
import time
import threading
import pyttsx3
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.45
CAMERA_INDEX = 0
SPEAK_EVERY_SECONDS = 3

class Speaker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.last_spoken = 0
        self.lock = threading.Lock()

    def speak(self, text):
        now = time.time()
        if now - self.last_spoken < SPEAK_EVERY_SECONDS:
            return
        self.last_spoken = now
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def _speak_thread(self, text):
        with self.lock:
            self.engine.say(text)
            self.engine.runAndWait()

def estimate_direction(x1, x2, frame_width):
    center = (x1 + x2) / 2
    if center < frame_width / 3:
        return "left"
    elif center > 2 * frame_width / 3:
        return "right"
    return "center"

def main():
    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)
    speaker = Speaker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera not found.")
        return

    print("Smart Cane demo started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        results = model(frame, conf=CONFIDENCE, verbose=False)
        frame_height, frame_width = frame.shape[:2]

        nearest_label = None
        nearest_area = 0
        nearest_direction = None

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                direction = estimate_direction(x1, x2, frame_width)

                if area > nearest_area:
                    nearest_area = area
                    nearest_label = label
                    nearest_direction = direction

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f} | {direction}"
                cv2.putText(frame, text, (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if nearest_label:
            warning = f"{nearest_label} detected in the {nearest_direction}"
            speaker.speak(warning)
            cv2.putText(frame, warning, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "No obstacle detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Smart Cane Object Detection", frame)

        if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
