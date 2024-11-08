import cv2
import numpy as np
import time
import Jetson.GPIO as GPIO
from ultralytics import YOLO
import threading
import queue

io0 = 18
io1 = 37  
io2 = 11  
io3 = 16
GPIO.setmode(GPIO.BOARD)
GPIO.setup(16, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(37, GPIO.OUT)
GPIO.setup(11, GPIO.OUT)
GPIO.output(18, GPIO.LOW)
GPIO.output(16, GPIO.LOW)
GPIO.output(11, GPIO.LOW)
GPIO.output(37, GPIO.LOW)

distance_value = 50
model = YOLO('./best.pt')  # or any other model path

# Object detection thresholds
CONF_THRESH = 0.1
IOU_THRESHOLD = 0.1


class DisplayThreadLeft(threading.Thread):
    def __init__(self, queue_left, queue_right):
        threading.Thread.__init__(self)
        self.queue_left = queue_left
        self.queue_right = queue_right

    def run(self):
        while True:
            try:
                # Get frames from both queues
                frame_left = self.queue_left.get(timeout=1)
                frame_right = self.queue_right.get(timeout=1)
                
                # Concatenate both frames horizontally (side by side)
                combined_frame = cv2.hconcat([frame_left, frame_right])

                # Show the combined frame
                cv2.imshow("Combined View", combined_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except queue.Empty:
                pass


class LeftThread(threading.Thread):
    def __init__(self, model, display_queue):
        threading.Thread.__init__(self)
        self.model = model
        self.display_queue = display_queue
        self.left_flag = 0
        self.right_flag = 0

    def run(self):
        cap = cv2.VideoCapture("1.mp4")  # Left camera (change index if needed)
        while True:
            
            _, frame = cap.read()
            start_time = time.time()
            results = self.model(frame, verbose=False)  # Inference
            annotated_frame = results[0].plot()
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Get boxes in [x1, y1, x2, y2] format
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            height, width, _ = frame.shape

            # Crosshair drawing
            center_x, center_y = width // 2, height // 2
            cv2.line(annotated_frame, (0, center_y), (width, center_y), (0, 255, 0), 2)  # Horizontal line
            cv2.line(annotated_frame, (center_x, 0), (center_x, height), (0, 255, 0), 2)  # Vertical line

            # Iterate over detected boxes and calculate distances to the crosshair
            for i in range(len(boxes)):
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                x1, y1, x2, y2 = box
                center_box_x = int((x1 + x2) / 2)
                center_box_y = int((y1 + y2) / 2)

                dist_x = center_x - center_box_x
                dist_y = abs(center_y - center_box_y)

                if dist_x > 0:
                    if np.abs(dist_y) < distance_value and self.left_flag == 0:
                        GPIO.output(io3, GPIO.HIGH)
                        self.left_flag = 1
                        print("Trigger GPIO 3 (left side)")
                    else:
                        GPIO.output(io3, GPIO.LOW)
                        print("GPIO 3 (left side) OFF")

                    if np.abs(dist_y) > distance_value and self.left_flag == 1:
                        self.left_flag = 0
                elif dist_x < 0:
                    if np.abs(dist_y) < distance_value and self.right_flag == 0:
                        self.right_flag = 1
                        GPIO.output(io0, GPIO.HIGH)
                        print("Trigger GPIO 0 (right side)")
                    else:
                        GPIO.output(io0, GPIO.LOW)
                        print("GPIO 0 (right side) OFF")

                    if np.abs(dist_y) > distance_value and self.right_flag == 1:
                        self.right_flag = 0

                # Draw lines from the box center to the crosshair center
                cv2.line(annotated_frame, (center_box_x, center_box_y), (center_x, center_y), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Dist X: {dist_x} px", (center_box_x + 10, center_box_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if len(boxes) == 0:
                GPIO.output(io0, GPIO.LOW)
                GPIO.output(io3, GPIO.LOW)
                self.right_flag = 0
                self.left_flag = 0

            img_out = cv2.putText(annotated_frame, "FPS= %.2f" % (1 / (time.time() - start_time)), (0, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.display_queue.put(img_out)


class RightThread(threading.Thread):
    def __init__(self, model, display_queue):
        threading.Thread.__init__(self)
        self.model = model
        self.display_queue = display_queue
        self.left_flag = 0
        self.right_flag = 0

    def run(self):
        cap = cv2.VideoCapture("1.mp4")  # Right camera (change index if needed)
        while True:
            start_time = time.time()
            _, frame = cap.read()
            results = self.model(frame, verbose=False)  # Inference
            annotated_frame = results[0].plot()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            height, width, _ = frame.shape

            # Crosshair drawing
            center_x, center_y = width // 2, height // 2
            cv2.line(annotated_frame, (0, center_y), (width, center_y), (0, 255, 0), 2)
            cv2.line(annotated_frame, (center_x, 0), (center_x, height), (0, 255, 0), 2)

            for i in range(len(boxes)):
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                x1, y1, x2, y2 = box
                center_box_x = int((x1 + x2) / 2)
                center_box_y = int((y1 + y2) / 2)

                dist_x = center_x - center_box_x
                dist_y = abs(center_y - center_box_y)

                if dist_x > 0:
                    if np.abs(dist_y) < distance_value and self.left_flag == 0:
                        GPIO.output(io1, GPIO.HIGH)
                        self.left_flag = 1
                        print("Trigger GPIO 1 (left side)")
                    else:
                        GPIO.output(io1, GPIO.LOW)
                        print("GPIO 1 (left side) OFF")

                    if np.abs(dist_y) > distance_value and self.left_flag == 1:
                        self.left_flag = 0
                elif dist_x < 0:
                    if np.abs(dist_y) < distance_value and self.right_flag == 0:
                        self.right_flag = 1
                        GPIO.output(io2, GPIO.HIGH)
                        print("Trigger GPIO 2 (right side)")
                    else:
                        GPIO.output(io2, GPIO.LOW)
                        print("GPIO 2 (right side) OFF")

                    if np.abs(dist_y) > distance_value and self.right_flag == 1:
                        self.right_flag = 0

                cv2.line(annotated_frame, (center_box_x, center_box_y), (center_x, center_y), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Dist X: {dist_x} px", (center_box_x + 10, center_box_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if len(boxes) == 0:
                GPIO.output(io1, GPIO.LOW)
                GPIO.output(io2, GPIO.LOW)
                self.right_flag = 0
                self.left_flag = 0

            img_out = cv2.putText(annotated_frame, "FPS= %.2f" % (1 / (time.time() - start_time)), (0, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.display_queue.put(img_out)


if __name__ == "__main__":
    display_queue_left = queue.Queue()
    display_queue_right = queue.Queue()

    left_thread = LeftThread(model, display_queue_left)
    right_thread = RightThread(model, display_queue_right)
    display_thread = DisplayThreadLeft(display_queue_left, display_queue_right)

    left_thread.start()
    right_thread.start()
    display_thread.start()

    left_thread.join()
    right_thread.join()
    display_thread.join()
