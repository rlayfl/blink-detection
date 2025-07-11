import cv2
import dlib
from scipy.spatial import distance
import imutils
import time
import json
from datetime import datetime

# Ask user for name
user_name = input("Enter your name: ").strip().replace(" ", "_")
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{user_name}_{session_timestamp}_blinks.json"

# Eye Aspect Ratio function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  # vertical
    B = distance.euclidean(eye[2], eye[4])  # vertical
    C = distance.euclidean(eye[0], eye[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return ear

# Constants
EAR_THRESHOLD = 0.25

# Initialise dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)  # Left eye landmarks
(rStart, rEnd) = (36, 42)  # Right eye landmarks

# Start webcam
cap = cv2.VideoCapture(0)

amount_of_blinks = 0
blinking = False
blink_log = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
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
                "timestamp_ms": current_time_ms,
                "blinking": True
            })

        status = "Eyes Open" if ear > EAR_THRESHOLD else "Eyes Closed"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display current time in ms
        time_text = f"Time: {current_time_ms} ms"
        cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

# Save blink log
with open(filename, "w") as f:
    json.dump(blink_log, f, indent=2)

print(f"Blink log saved to {filename}")
