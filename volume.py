import cv2
import time
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker,HandLandmarkerOptions 
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VisionTaskRunningMode
 
 
base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")

latest_result= [None]

def on_result(result,output_image,timestamp_ms):
    latest_result[0] = result
    
options = HandLandmarkerOptions(base_options=base_options,running_mode = VisionTaskRunningMode.LIVE_STREAM, num_hands = 2, min_hand_detection_confidence = 0.5, result_callback = on_result)

landmarker = HandLandmarker.create_from_options(options)
print("HandLandMaker createdd successfully!!")

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame = cv2.flip(frame,1)
    
    h,w = frame.shape[:2]
    