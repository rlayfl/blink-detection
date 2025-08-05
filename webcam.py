import cv2
import numpy as np
import dlib
from scipy.spatial import distance
import imutils
import time
import json
import dxcam
from datetime import datetime

# --- CONFIGURATION ---
USE_WEBCAM = False
USE_WEBCAM = True
monitor_region = {"top": 400, "left": 400, "width": 400, "height": 400}
# ---------------------

# Ask user for name
session_timestamp = str(int(time.time()))
json_filename = f"{session_timestamp}.json"

# Eye Aspect Ratio function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  # vertical
    B = distance.euclidean(eye[2], eye[4])  # vertical
    C = distance.euclidean(eye[0], eye[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return ear

# Constants
EAR_THRESHOLD = 0.2

# Initialise dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)  # Left eye landmarks
(rStart, rEnd) = (36, 42)  # Right eye landmarks

# Video capture setup
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
else:
    frame_width = monitor_region["width"]
    frame_height = monitor_region["height"]
    fps = 24  # reasonable default for screen capture

amount_of_blinks = 0
blinking = False
blink_log = []

if USE_WEBCAM:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        current_time_ms = int(time.time() * 1000)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            if ear < EAR_THRESHOLD:
                blinking = False

            elif ear > EAR_THRESHOLD and not blinking:
                amount_of_blinks += 1
                print("Blink detected", amount_of_blinks)
                blinking = True
                blink_log.append({
                    "timestamp_ms": current_time_ms
                })

            status = "Eyes Open" if ear > EAR_THRESHOLD else "Eyes Closed"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            time_text = f"Time: {current_time_ms} ms"
            cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Frame", imutils.resize(frame, width=450))
        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()

else:
    # Set up dxcam
    region = (
        monitor_region["left"],
        monitor_region["top"],
        monitor_region["left"] + monitor_region["width"],
        monitor_region["top"] + monitor_region["height"]
    )
    camera = dxcam.create()
    camera.start(region=region)

    while True:
        frame = camera.get_latest_frame()
        if frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        current_time_ms = int(time.time() * 1000)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            if ear < EAR_THRESHOLD:
                blinking = False

            elif ear > EAR_THRESHOLD and not blinking:
                amount_of_blinks += 1
                print("Blink detected", amount_of_blinks)
                blinking = True
                blink_log.append({
                    "timestamp_ms": current_time_ms
                })

            status = "Eyes Open" if ear > EAR_THRESHOLD else "Eyes Closed"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            time_text = f"Time: {current_time_ms} ms"
            cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Frame", imutils.resize(frame, width=450))
        if cv2.waitKey(1) == 27:  # ESC key
            break

with open(json_filename, "w") as f:
    json.dump({"session_timestamp": session_timestamp, "blink_log": blink_log}, f)
cv2.destroyAllWindows()