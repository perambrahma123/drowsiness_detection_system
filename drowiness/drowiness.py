import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import os
import threading
import platform

# For sound handling
if platform.system() == "Windows":
    import winsound
else:
    import os

# Create directories for logs and clips
os.makedirs('logs', exist_ok=True)
os.makedirs('clips', exist_ok=True)

# Constants
EYE_ASPECT_RATIO_THRESHOLD = 0.25
EYE_ASPECT_RATIO_CONSEC_FRAMES = 20
FRAME_BUFFER_SIZE = 300  # 10 seconds at 30 fps

# MediaPipe eye landmarks indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize variables
cap = cv2.VideoCapture(0)
# Set camera resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

counter = 0
alert_active = False
alert_start_time = None
user_input = ""
frame_buffer = []
sound_thread = None
sound_playing = False

def calculate_ear(landmarks, eye_indices):
    """Calculate the eye aspect ratio (EAR) for given eye landmarks"""
    eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    vertical_dist1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical_dist2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

def log_event(event_type):
    """Log drowsiness events with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open('logs/drowsiness_events.log', 'a') as f:
        f.write(f"{timestamp}: {event_type}\n")
    return timestamp

def save_video_clip(frame_buffer, timestamp):
    """Save the video buffer when drowsiness is detected"""
    if frame_buffer:
        out_path = f'clips/drowsiness_{timestamp}.avi'
        height, width = frame_buffer[0].shape[:2]
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (width, height))
        for frame in frame_buffer:
            out.write(frame)
        out.release()

def play_alert_sound():
    """Play alert sound in a loop until drowsiness is addressed"""
    global sound_playing
    sound_playing = True
    while alert_active and sound_playing:
        try:
            if platform.system() == "Windows":
                winsound.Beep(1000, 500)  # Frequency = 1000Hz, Duration = 500ms
            else:
                os.system('play -nq -t alsa synth 0.5 sine 1000')  # For Linux
            time.sleep(0.5)
        except Exception as e:
            print(f"Error playing sound: {e}")
            break
    sound_playing = False

def start_alert():
    """Start the alert system including sound"""
    global alert_active, sound_thread
    if not alert_active:
        alert_active = True
        sound_thread = threading.Thread(target=play_alert_sound)
        sound_thread.daemon = True
        sound_thread.start()

def stop_alert():
    """Stop the alert system"""
    global alert_active, sound_playing
    alert_active = False
    sound_playing = False
    time.sleep(0.5)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Store frame in buffer
    frame_buffer.append(frame.copy())
    if len(frame_buffer) > FRAME_BUFFER_SIZE:
        frame_buffer.pop(0)

    # Process frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    display_frame = frame.copy()

    # Display EAR threshold and user input
    cv2.putText(display_frame, f"EAR Threshold: {EYE_ASPECT_RATIO_THRESHOLD:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Input: {user_input}", 
                (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate EAR
        left_ear = calculate_ear(landmarks, LEFT_EYE)
        right_ear = calculate_ear(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        
        # Display current EAR
        cv2.putText(display_frame, f"Current EAR: {ear:.2f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Drowsiness detection
        if ear < EYE_ASPECT_RATIO_THRESHOLD:
            counter += 1
            if counter >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                start_alert()
                if alert_start_time is None:
                    alert_start_time = time.time()
                    timestamp = log_event("Drowsiness Detected")
                    save_video_clip(frame_buffer, timestamp)
        else:
            counter = 0
        
        # Draw eye landmarks
        frame_height, frame_width = frame.shape[:2]
        for eye_points in [LEFT_EYE, RIGHT_EYE]:
            for point in eye_points:
                pos = landmarks[point]
                x = int(pos.x * frame_width)
                y = int(pos.y * frame_height)
                cv2.circle(display_frame, (x, y), 2, (0, 255, 0), -1)

    # Display warning if alert is active
    if alert_active:
        cv2.putText(display_frame, "WARNING: Drowsiness Detected!", 
                    (int(display_frame.shape[1]/4), int(display_frame.shape[0]/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(display_frame, "Type 'I am fine' to dismiss", 
                    (int(display_frame.shape[1]/4), int(display_frame.shape[0]/2) + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key != 255 and alert_active:
        if key == 8:  # Backspace
            user_input = user_input[:-1]
        elif key == 13:  # Enter key
            if user_input.lower() == "i am fine":
                stop_alert()
                alert_start_time = None
                counter = 0
                user_input = ""
        else:  # Regular character input
            user_input += chr(key)

    # Display the frame
    display_frame = cv2.resize(display_frame, (1280, 720))  # Resize to 1280x720
    cv2.imshow('Drowsiness Detection', display_frame)

# Cleanup
cap.release()
cv2.destroyAllWindows()
face_mesh.close()