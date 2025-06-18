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
EYE_ASPECT_RATIO_CONSEC_FRAMES = 10
FRAME_BUFFER_SIZE = 300  # 10 seconds at 30 fps
THUMBS_UP_CONSEC_FRAMES = 30  # Number of consecutive frames with thumbs up to confirm awake

# MediaPipe eye landmarks indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize variables
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

counter = 0
alert_active = False
alert_start_time = None
frame_buffer = []
sound_thread = None
sound_playing = False
thumbs_up_counter = 0

def calculate_ear(landmarks, eye_indices):
    """Calculate the eye aspect ratio (EAR) for given eye landmarks"""
    eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    vertical1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def detect_thumbs_up(hand_landmarks, frame_width, frame_height):
    """Enhanced thumb detection with multiple verification checks"""
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]
    index_tip = hand_landmarks.landmark[8]
    
    # 1. Thumb raised check (y-coordinate comparison)
    thumb_raised = thumb_tip.y < thumb_mcp.y  # Lower y = higher position
    
    # 2. Finger closure check
    fingers_closed = True
    for tip_idx, pip_idx in [(8,6), (12,10), (16,14), (20,18)]:  # tip, pip joints
        if hand_landmarks.landmark[tip_idx].y < hand_landmarks.landmark[pip_idx].y:
            fingers_closed = False
            break
    
    # 3. Thumb-index separation check
    thumb_vector = np.array([thumb_tip.x - thumb_mcp.x, thumb_tip.y - thumb_mcp.y])
    thumb_length = np.linalg.norm(thumb_vector)
    thumb_index_dist = np.linalg.norm([
        thumb_tip.x - index_tip.x,
        thumb_tip.y - index_tip.y
    ])
    well_separated = thumb_index_dist > thumb_length * 0.7
    
    return thumb_raised and fingers_closed and well_separated

def log_event(event_type):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open('logs/drowsiness_events.log', 'a') as f:
        f.write(f"{timestamp}: {event_type}\n")
    return timestamp

def save_video_clip(frame_buffer, timestamp):
    if frame_buffer:
        out_path = f'clips/drowsiness_{timestamp}.avi'
        height, width = frame_buffer[0].shape[:2]
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (width, height))
        for frame in frame_buffer:
            out.write(frame)
        out.release()

def play_alert_sound():
    global sound_playing
    sound_playing = True
    while alert_active and sound_playing:
        try:
            if platform.system() == "Windows":
                winsound.Beep(1000, 500)
            else:
                os.system('play -nq -t alsa synth 0.5 sine 1000')
            time.sleep(0.5)
        except Exception as e:
            print(f"Error playing sound: {e}")
            break
    sound_playing = False

def start_alert():
    global alert_active, sound_thread, thumbs_up_counter
    if not alert_active:
        alert_active = True
        thumbs_up_counter = 0
        sound_thread = threading.Thread(target=play_alert_sound)
        sound_thread.daemon = True
        sound_thread.start()

def stop_alert():
    global alert_active, sound_playing, thumbs_up_counter
    alert_active = False
    sound_playing = False
    thumbs_up_counter = 0
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
    frame_height, frame_width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)
    display_frame = frame.copy()

    # Display EAR threshold
    cv2.putText(display_frame, f"EAR Threshold: {EYE_ASPECT_RATIO_THRESHOLD:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Face mesh processing
    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        
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

    # Hand processing for thumbs up detection
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Check for thumbs up
            if detect_thumbs_up(hand_landmarks, frame_width, frame_height):
                cv2.putText(display_frame, "THUMBS UP DETECTED", 
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if alert_active:
                    thumbs_up_counter += 1
                    cv2.putText(display_frame, f"Confirmation: {thumbs_up_counter}/{THUMBS_UP_CONSEC_FRAMES}", 
                                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    if thumbs_up_counter >= THUMBS_UP_CONSEC_FRAMES:
                        stop_alert()
                        alert_start_time = None
                        counter = 0
                        log_event("Awake Confirmed")
            else:
                thumbs_up_counter = max(0, thumbs_up_counter - 1)

    # Display warning if alert is active
    if alert_active:
        cv2.putText(display_frame, "WARNING: DROWSINESS DETECTED!", 
                    (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(display_frame, "SHOW THUMBS UP TO CONFIRM YOU'RE AWAKE", 
                    (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if thumbs_up_counter > 0:
            cv2.putText(display_frame, f"Keep thumbs up! ({thumbs_up_counter}/{THUMBS_UP_CONSEC_FRAMES})", 
                        (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow('Drowsiness Detection with Thumb Recognition', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
hands.close()
