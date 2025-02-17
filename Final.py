import numpy as np
import cv2 as cv
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time
import os
from tkinter import Tk, Label
from PIL import ImageTk, Image
import random
from pathlib import Path

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
GestureRecognizer = mp.tasks.vision.GestureRecognizer

DESIRED_FPS = 30
CAMERA_ID = 5
MAX_QUEUE_SIZE = 2
SKIP_FRAMES = 1

class GestureRecognizerWrapper:
    def __init__(self):
        self.results_queue = deque(maxlen=5)
        self.latest_result = None
        self.frame_counter = 0


        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path='Z:/fromarch/newmodel.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.callback
            )

        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def callback(self, result, output_image, timestamp_ms):
        if result.gestures:
            self.latest_result = {
                    'gestures': result.gestures,
                    'hand_landmarks': result.hand_landmarks,
                    'timestamp': timestamp_ms
                }

    def process_frame(self, frame):
        self.frame_counter += 1
        if self.frame_counter % (SKIP_FRAMES + 1) != 0:
            return

        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=frame)
        self.recognizer.recognize_async(
                mp_image,
                int(time.time() * 1000)
        )

def draw_minimal_ui(frame, result, display, last_gesture_time, cooldown):
    """Optimized drawing function with only essential UI elements"""
    annotated_frame = frame.copy()
    height, width, _ = annotated_frame.shape

    current_time = time.time()

    if result['gestures'] and result['hand_landmarks']:
        try:
            # Get the most confident gesture for the first hand
            top_gesture = result['gestures'][0][0].category_name

            if display == True:
                pass

            elif top_gesture == "middle" and (current_time - last_gesture_time >= cooldown):
                print("this is current time: ", current_time)
                print("this is last_gesture_time: ", last_gesture_time)
                display = True 
                last_gesture_time = current_time
                print("in if statement")
                random_number = random.randint(1, 5)

                if random_number == 1:
                    #os.system('shutdown /s /t 0')
                    print("CHANCE HIT!")
                    

            landmarks = result['hand_landmarks'][0]

            # Convert normalized coordinates to pixel values
            x_coords = [landmark.x * width for landmark in landmarks]
            y_coords = [landmark.y * height for landmark in landmarks]

            # Calculate bounding box with safety checks
            x_min = max(0, int(min(x_coords)))
            x_max = min(width, int(max(x_coords)))
            y_min = max(0, int(min(y_coords)))
            y_max = min(height, int(max(y_coords)))

            # Draw bounding box
            cv.rectangle(annotated_frame, 
                        (x_min, y_min),
                        (x_max, y_max),
                        (0, 255, 0),  # Green color
                        2)            # Thickness

            # Prepare text and position
            gesture_text = f"{top_gesture.replace('_', ' ')}"
            text_scale = 0.7
            (text_width, text_height), _ = cv.getTextSize(
                gesture_text, 
                cv.FONT_HERSHEY_SIMPLEX, 
                text_scale, 2
            )

            # Position text above bounding box (or below if near top)
            text_y = y_min - 10 if y_min > text_height + 10 else y_max + text_height + 10
            text_y = max(0, min(height - text_height, text_y))

            cv.putText(annotated_frame, gesture_text,

                        (x_min, text_y),
                      cv.FONT_HERSHEY_SIMPLEX,
                      text_scale,
                      (0, 255, 0),  # Green color
                      2)            # Thickness

        except Exception as e:
            print(f"Drawing error: {str(e)}")
    
    return annotated_frame, last_gesture_time, display

def main():

    cooldown = 1
    last_gesture_time = 0

    lock = False 
    display = False

    recognizer = GestureRecognizerWrapper()
    cap = cv.VideoCapture(CAMERA_ID)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, DESIRED_FPS)

    last_display_time = time.time()

    try:
        while True:

            ret, frame = cap.read()
            if not ret:
                break
            
            recognizer.process_frame(frame)

            current_time = time.time()
            if current_time - last_display_time >= 1/DESIRED_FPS:
                display_frame = frame.copy()

                if recognizer.latest_result:

                    display_frame, last_gesture_time, display = draw_minimal_ui(display_frame, recognizer.latest_result, display, last_gesture_time, cooldown)

                cv.imshow('Window', display_frame)
                last_display_time = current_time

            
            if display == True:
                root = Tk()
                root.overrideredirect(True)  # Remove window borders
                root.geometry("1920x1080+0+0")
#                root.eval('tk::PlaceWindow . center')  # Center the window

                # Set a "transparent" color (e.g., gray15) for the window and label
                transparent_color = "gray15"
                root.configure(bg=transparent_color)
                root.wm_attributes("-transparentcolor", transparent_color)  # Key line

                # Load the image with Pillow (preserve alpha channel)
                img = Image.open("C:/Users/Kompyuter/Desktop/angryface2.png")#.resize((1000, 1000))
                img_tk = ImageTk.PhotoImage(img)

                # Create a label with the image and matching background color
                label = Label(root, image=img_tk, bg=transparent_color)
                label.pack()

                # Close after 3 seconds (or integrate with your main app)
                root.after(1000, root.destroy)
                root.mainloop()
                display = False


            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv.destroyAllWindows()
        recognizer.recognizer.close()

if __name__ == "__main__":
    main()
