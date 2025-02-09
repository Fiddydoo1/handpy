import cv2 as cv
import mediapipe as mp
import time
from collections import deque
from mediapipe.tasks.python import vision
from tkinter import Tk, Label
from PIL import ImageTk, Image
import os


# Configuration
DESIRED_FPS = 30
CAMERA_ID = 4
MAX_QUEUE_SIZE = 2  # Process only latest 2 frames
SKIP_FRAMES = 1     # Process every other frame

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
VisionRunningMode = mp.tasks.vision.RunningMode

display = False
number = 0

class GestureRecognizerWrapper:
    def __init__(self):
        self.results_queue = deque(maxlen=5)
        self.latest_result = None
        self.frame_counter = 0
        
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path='Z:/fromarch/newcreation.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.callback
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def callback(self, result, output_image, timestamp_ms):
        """Store the latest result for main thread rendering"""
        if result.gestures:
            self.latest_result = {
                'gestures': result.gestures,
                'hand_landmarks': result.hand_landmarks,
                'timestamp': timestamp_ms
            }

    def process_frame(self, frame):
        """Process frames with controlled rate"""
        self.frame_counter += 1
        if self.frame_counter % (SKIP_FRAMES + 1) != 0:
             return

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.recognizer.recognize_async(
            mp_image,
            int(time.time() * 1000)
        )

def draw_minimal_ui(frame, result):
    """Optimized drawing function with only essential UI elements"""
    annotated_frame = frame.copy()
    height, width, _ = annotated_frame.shape

    global display
    global number

    if result['gestures'] and result['hand_landmarks']:
        try:
            # Get the most confident gesture for the first hand
            top_gesture = result['gestures'][0][0].category_name

            if number >= 1:
                pass

            elif top_gesture == "middle":
                display = True
                number += 1


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
    
    return annotated_frame

def main():

    global display

    recognizer = GestureRecognizerWrapper()
    cap = cv.VideoCapture(CAMERA_ID)
    
    # Set camera parameters
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, DESIRED_FPS)
   # cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    last_display_time = time.time()
    
    try:
        while True:

            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame (with skipping)
            recognizer.process_frame(frame)

            # Display results at fixed FPS
            current_time = time.time()
            if current_time - last_display_time >= 1/DESIRED_FPS:
                display_frame = frame.copy()
                
                # Add latest results if available
                if recognizer.latest_result:

                    display_frame = draw_minimal_ui(display_frame, recognizer.latest_result)
                    # Implement your optimized drawing here
                    #gestures = recognizer.latest_result['gestures']
                    #print(f"Latest gesture: {gestures[0][0].category_name}")
                    
                cv.imshow('Gesture Control', display_frame)
                last_display_time = current_time

            print(display)

            if display == True:
                root = Tk()
                root.overrideredirect(True)  # Remove window borders
                root.geometry("1000x1000")
                root.eval('tk::PlaceWindow . center')  # Center the window

                # Set a "transparent" color (e.g., gray15) for the window and label
                transparent_color = "gray15"
                root.configure(bg=transparent_color)
                root.wm_attributes("-transparentcolor", transparent_color)  # Key line

                # Load the image with Pillow (preserve alpha channel)
                img = Image.open("Image").resize((300, 300))
                img_tk = ImageTk.PhotoImage(img)

                # Create a label with the image and matching background color
                label = Label(root, image=img_tk, bg=transparent_color)
                label.pack()

                # Close after 3 seconds (or integrate with your main app)
                root.after(3000, root.destroy)
                root.mainloop()

            # Exit on 'q'
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv.destroyAllWindows()
        recognizer.recognizer.close()

if __name__ == "__main__":
    main()
