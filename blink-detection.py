import cv2
import dlib
from scipy.spatial import distance
import imutils

# Eye Aspect Ratio function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  # vertical
    B = distance.euclidean(eye[2], eye[4])  # vertical
    C = distance.euclidean(eye[0], eye[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return ear

# Constants
EAR_THRESHOLD = 0.25  # This threshold works well if the eyes are clearly visible

# Initialise dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)  # Left eye landmarks
(rStart, rEnd) = (36, 42)  # Right eye landmarks

# Start webcam
cap = cv2.VideoCapture(0)

amount_of_blinks = 0
blinking = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        status = "Eyes Open - Attractiveness 10/10" if ear > EAR_THRESHOLD else "Eyes Closed - Attractiveness 10/10"

        if ear < EAR_THRESHOLD:
            blinking = False

        if ear > EAR_THRESHOLD and not blinking:            
            amount_of_blinks += 1
            print("Blink detected", amount_of_blinks)
            blinking = True

        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
