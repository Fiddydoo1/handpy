import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2 as cv

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):

    print("Landmarks result: ", result.hand_landmarks)

    if result.hand_landmarks:

        np_image = np.array(output_image.numpy_view())

        print("hello number 1")
        for hand_landmarks in result.hand_landmarks:
            mp_drawing.draw_landmarks(np_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
        print("hello number 2")
        cv.imshow(output_image)

        print("hello number 3")
        if cv.waitKey(5) & 0xFF == 27:
                return        



options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='C:/Users/Kompyuter/Desktop/newhand/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
    )

#Maybe try if this shit works without the "with" keyword

with HandLandmarker.create_from_options(options) as landmarker:

    cap = cv.VideoCapture(4)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv.imshow("Feed", frame) 
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        timestamp_ms = int(time.time() * 1000)

        landmarker.detect_async(mp_image, timestamp_ms)

    cap.release()
    cv.destroyAllWindows()


