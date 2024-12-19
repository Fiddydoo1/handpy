
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 as cv
import pylab as plt
import threading

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
   hand_landmarks_list = detection_result.hand_landmarks
   handedness_list = detection_result.handedness
   annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
   for idx in range(len(hand_landmarks_list)):
     hand_landmarks = hand_landmarks_list[idx]
     handedness = handedness_list[idx]

    # Draw the hand landmarks.
     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
     hand_landmarks_proto.landmark.extend([
       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
     ])
     solutions.drawing_utils.draw_landmarks(
       annotated_image,
       hand_landmarks_proto,
       solutions.hands.HAND_CONNECTIONS,
       solutions.drawing_styles.get_default_hand_landmarks_style(),
       solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
     height, width, _ = annotated_image.shape
     x_coordinates = [landmark.x for landmark in hand_landmarks]
     y_coordinates = [landmark.y for landmark in hand_landmarks]
     text_x = int(min(x_coordinates) * width)
     text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
     cv.putText(annotated_image, f"{handedness[0].category_name}",
                 (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                 FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

   return annotated_image

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):

    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
    
    cv.imshow("that", annotated_image)
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        exit()
    
        
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='C:/Users/Kompyuter/Desktop/newhand/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
    )

#Maybe try if this works without the "with" keyword

landmarker = HandLandmarker.create_from_options(options)

cap = cv.VideoCapture(4, cv.CAP_DSHOW)
    
if not cap.isOpened():
        print("Cannot open camera")
        exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #cv.imshow("Feed", frame) 

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    timestamp_ms = int(time.time() * 1000)

    landmarker.detect_async(mp_image, timestamp_ms)
cap.release()
cv.destroyAllWindows
   
    
