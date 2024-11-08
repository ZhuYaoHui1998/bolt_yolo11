#!/usr/bin/python3
import queue
import threading
import cv2
import numpy as np
# import Jetson.GPIO as GPIO
import time
from ultralytics import YOLO

# # GPIO setup
# x0 = 18
# x1 = 37
# x2 = 11
# x3 = 16
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(x0, GPIO.OUT)
# GPIO.setup(x1, GPIO.OUT)
# GPIO.setup(x2, GPIO.OUT)
# GPIO.setup(x3, GPIO.OUT)
# GPIO.output(x0, GPIO.LOW)
# GPIO.output(x1, GPIO.LOW)
# GPIO.output(x2, GPIO.LOW)
# GPIO.output(x3, GPIO.LOW)

# Object detection thresholds
CONF_THRESH = 0.1
IOU_THRESHOLD = 0.1

# Load YOLO model (replace with your desired model path or 'yolov5' or 'yolov8' if using the online model)
model = YOLO('./best.pt')  # or any other model path

class DisplayThreadLeft(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            try:
                image = self.queue.get(timeout=1)
                cv2.imshow("left", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except queue.Empty:
                pass

class LeftThread(threading.Thread):
    def __init__(self, model, display_queue):
        threading.Thread.__init__(self)
        self.model = model
        self.display_queue = display_queue

    def run(self):
        cap = cv2.VideoCapture("202414.mp4")  # Left camera (change index if needed)
        left_flag = 0
        right_flag = 0
        while True:
            start_time = time.time() 
            _, frame = cap.read()
            results = self.model(frame)  # Inference
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Get boxes in [x1, y1, x2, y2] format
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            # Control GPIO based on detected objects and distances
            if len(boxes) > 0:
                for i in range(len(boxes)):
                    # Here we assume that you are using the distances from the NMS filtering
                    box = boxes[i]
                    score = scores[i]
                    class_id = class_ids[i]

                    # Example of logic based on box distance (adjust this part to fit your logic)
                    # if abs(box[0] - box[2]) > 0:  # Example condition
                    #     GPIO.output(x3, GPIO.HIGH)
                    # else:
                    #     GPIO.output(x3, GPIO.LOW)

            else:
                # GPIO.output(x3, GPIO.LOW)
                pass

            img_out = cv2.putText(frame, "FPS= %.2f" % (1 / (time.time() - start_time)), (0, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.display_queue.put(img_out)

class RightThread(threading.Thread):
    def __init__(self, model, display_queue):
        threading.Thread.__init__(self)
        self.model = model
        self.display_queue = display_queue

    def run(self):
        cap1 = cv2.VideoCapture("202414.mp4")  # Right camera
        left_flag = 0
        right_flag = 0
        while True:
            start_time = time.time() 
            _, frame = cap1.read()
            results = self.model(frame)  # Inference
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Get boxes in [x1, y1, x2, y2] format
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            # Control GPIO based on detected objects and distances
            if len(boxes) > 0:
                for i in range(len(boxes)):
                    # Example logic based on box distance
                    box = boxes[i]
                    score = scores[i]
                    class_id = class_ids[i]

                    # if abs(box[0] - box[2]) > 0:  # Example condition
                    #     GPIO.output(x2, GPIO.HIGH)
                    # else:
                    #     GPIO.output(x2, GPIO.LOW)

            else:
                # GPIO.output(x2, GPIO.LOW)
                pass

            img_out = cv2.putText(frame, "FPS= %.2f" % (1 / (time.time() - start_time)), (0, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.display_queue.put(img_out)

if __name__ == "__main__":
    # Initialize display queue
    display_queue_left = queue.Queue()
    display_thread_left = DisplayThreadLeft(display_queue_left)
    display_thread_left.start()

    # Start the left and right threads for processing the cameras
    try:
        thread1 = LeftThread(model, display_queue_left)
        # thread2 = RightThread(model, display_queue_left)
        thread1.start()
        # thread2.start()
        thread1.join()
        # thread2.join()
    finally:
        pass
        # Cleanup GPIO pins
        # GPIO.cleanup()
