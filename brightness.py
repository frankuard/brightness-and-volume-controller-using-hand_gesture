import cv2 
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VisionTaskRunningMode 


## ADDING REQUIRED MODEL FOR THIS PROJECT

base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.tasks")


latest_result = [None]

def on_result(result,output_image,timestamps_ms):
    latest_result = result[0]
    
options = HandLandmarkerOptions(base_options=base_options,running_mode=VisionTaskRunningMode.LIVE_STREAM,num_hands=2,min_hand_detection_confidence = 0.5, result_callback= on_result)

landmarker = HandLandmarker.create_from_options(options)
print("HandLandMaker created successfully")

