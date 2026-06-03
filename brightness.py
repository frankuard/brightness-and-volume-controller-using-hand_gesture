import cv2 
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VisionTaskRunningMode 


## ADDING REQUIRED MODEL FOR THIS PROJECT

base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.tasks")
