
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
      
      #So this is a loop for each individual thing in detection_result.hand_landmarks
     hand_landmarks = hand_landmarks_list[idx]
     handedness = handedness_list[idx]

    # Draw the hand landmarks.
      #Normalized Landmark represents a point in 3D space with x, y, z coordinates. 
      #x and y are normalized to [0.0, 1.0] by the image width and height respectively. 
      #z represents the landmark depth, and the smaller the value the closer the landmark 
      #is to the camera. 
     # The magnitude of z uses roughly the same scale as x.
     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      # The extend method here     V seems to come from something else I can maybe check
      #with visual code.
     hand_landmarks_proto.landmark.extend([
        #      where does this             V            V               V   come from? V here?
       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        #Then it could make sense how we 1. Go through the hand_landmarks_list = detection_result.hand_landmarks
        # 2. We create a for loop based on how many elements there are in hand_landmarks_list and assign it
        # to hand_landmarks.
        # 3. Then we create like the 3D points based on the values of landmark in hand_landmarks
        # By creating a "NormalizedLandmark".
     ])

      # These lines draw the hand_landmarks with a default style on the image with
      # the given coordinates.
      # I can try to look up these methods right here like draw_landmarks().
      # It seems to take in the original image, the hand_landmarks_proto which is the
      # variable that we assign the landmark locations to.
      # Then it contains three more lines with variables and methods I should look up.
      # solutions.hand.HAND_CONNECTIONS,
      # solutions.drawing_styles.get_default_hand_landmarks_style(),
      # solutions.drawing_styles.get_default_hand_connections_style())
     solutions.drawing_utils.draw_landmarks(
       annotated_image,
       hand_landmarks_proto,
       solutions.hands.HAND_CONNECTIONS,
       solutions.drawing_styles.get_default_hand_landmarks_style(),
       solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
      # annotated_image seems to have something called shape.
      # Since annotated_image is a np.copy(rgb_image) I can look up what that is
      # and if there's anything else.
     height, width, _ = annotated_image.shape

      # This looks like some python specific syntax V
     x_coordinates = [landmark.x for landmark in hand_landmarks]
     y_coordinates = [landmark.y for landmark in hand_landmarks]
      
      # Find out what this logic does and what min() is.
     text_x = int(min(x_coordinates) * width)
     text_y = int(min(y_coordinates) * height) - MARGIN
      # Draw handedness
      # Here we use a method from the opencv library to draw if the hand is a left or right one.
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

landmarker = HandLandmarker.create_from_options(options)

cap = cv.VideoCapture(4)

timestamp_ms = 0
    
if not cap.isOpened():
        print("Cannot open camera")
        exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    timestamp_ms += 1

   # I'm pretty sure the lag from running the program comes from how the detect_async function is used.

    landmarker.detect_async(mp_image, timestamp_ms)
cap.release()
cv.destroyAllWindows
   
    
