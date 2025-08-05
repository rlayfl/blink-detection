import cv2
import numpy as np
import mss
import time

# Load DNN face detector
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# User input
experiment_name = f"experiment_{int(time.time())}"
print(f"Experiment name set to: {experiment_name}")

# Define screen region (or full screen)
monitor_region = {"top": 100, "left": 100, "width": 800, "height": 600}

# Confidence threshold (increase to reduce false positives)
CONF_THRESHOLD = 0.3

with mss.mss() as sct:
    while True:
        # Capture screen frame
        screen_frame = sct.grab(monitor_region)

        # Convert to NumPy array
        frame = np.array(screen_frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Prepare input blob for DNN
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0), swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONF_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

        # Display frame
        cv2.imshow("Live Screen with DNN Face Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
