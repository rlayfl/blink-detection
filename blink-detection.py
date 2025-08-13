import argparse
import time
import json
import os
import cv2
import numpy as np
import mediapipe as mp

from mss import mss
from collections import deque

# ---- Eye landmark indices (MediaPipe Face Mesh, 468-landmark model) ----
LEFT_EYE = {
    "h": (33, 133),
    "v1": (160, 144),
    "v2": (158, 153),
}
RIGHT_EYE = {
    "h": (263, 362),
    "v1": (387, 373),
    "v2": (385, 380),
}

def euclidean(a, b):
    return np.linalg.norm(a - b)

def eye_aspect_ratio(landmarks, eye_indices):
    p_h1, p_h2 = landmarks[eye_indices["h"][0]], landmarks[eye_indices["h"][1]]
    p_v11, p_v12 = landmarks[eye_indices["v1"][0]], landmarks[eye_indices["v1"][1]]
    p_v21, p_v22 = landmarks[eye_indices["v2"][0]], landmarks[eye_indices["v2"][1]]

    horiz = euclidean(p_h1, p_h2)
    vert = euclidean(p_v11, p_v12) + euclidean(p_v21, p_v22)
    return (vert / (2.0 * horiz)) if horiz > 1e-6 else 0.0

# --- UPDATED: accept unix_time and show it on the overlay
def draw_overlay(frame, fps, ear_l, ear_r, blink_count, state, region_desc, unix_time):
    h, w = frame.shape[:2]
    pad = 10
    # increased height to fit the extra line for UNIX time
    cv2.rectangle(frame, (pad-5, pad-5), (300, 150), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (pad, 20 + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"EAR L: {ear_l:.3f}", (pad, 45 + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"EAR R: {ear_r:.3f}", (pad, 70 + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Blinks: {blink_count}", (pad, 95 + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
    # NEW: display UNIX time (seconds since epoch)
    cv2.putText(frame, f"UNIX: {unix_time}", (pad, 120 + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

    color = (0, 0, 255) if state == "CLOSED" else (0, 255, 0)
    cv2.putText(frame, f"Eyes: {state}", (w - 160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.putText(frame, region_desc, (pad, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

def parse_args():
    ap = argparse.ArgumentParser(description="Blink detection using MediaPipe Face Mesh from screen or webcam")
    # Input selection
    ap.add_argument("--input", choices=["screen", "webcam"], default="screen",
                    help="Choose frame source: screen or webcam")
    ap.add_argument("--camera-index", type=int, default=0,
                    help="Webcam device index (for --input webcam)")
    ap.add_argument("--flip", action="store_true",
                    help="Flip webcam horizontally (mirror)")

    # Screen capture
    ap.add_argument("--region", type=int, nargs=4, metavar=("X","Y","W","H"), default=[300, 300, 1300, 800],
                    help="Capture region (pixels). Overrides --monitor.")

    # Processing options
    ap.add_argument("--scale", type=float, default=0.75,
                    help="Downscale factor for processing speed (0.3–1.0)")
    ap.add_argument("--ear-thresh", type=float, default=0.21,
                    help="EAR threshold for eye closed")
    ap.add_argument("--min-frames", type=int, default=1,
                    help="Consecutive frames below threshold to count as a blink")

    # Multi-session
    ap.add_argument("--out", type=str, default=None,
                    help="Path to the JSON file. If omitted, an auto-unique name is used.")
    ap.add_argument("--title", type=str, default=None,
                    help="Window title / session name.")
    ap.add_argument("--no-gui", action="store_true",
                    help="Run headless (no preview window).")
    return ap.parse_args()

def main():
    args = parse_args()
    
    start_ts = time.time()      
    ts_ns = int(start_ts * 1e9)  
    safe_ts = int(start_ts)      

    if args.out:
        json_path = os.path.abspath(args.out)
    else:
        json_filename = f"{args.input}_{ts_ns}_{os.getpid()}.json"
        json_path = os.path.abspath(json_filename)

    log = {
        "source": args.input,                # "webcam" or "screen"
        "started_at": safe_ts,               # unix seconds at start
        "ear_threshold": float(args.ear_thresh),
        "min_frames": int(args.min_frames),
        "blinks": []                         # list of blink events
        # "total_blinks" will be added on exit
    }

    def write_log():
        # Ensure directory exists if user passed a path
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

    write_log()

    # ---- Init frame source
    cap = None
    sct = None
    region_desc = ""

    if args.input == "webcam":
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {args.camera_index}")
        region_desc = f"Webcam index {args.camera_index}"
    else:
        sct = mss()
        
        x, y, w, h = args.region
        monitor = {"left": x, "top": y, "width": w, "height": h}
        region_desc = f"Region {x},{y},{w}x{h}"

    # ---- MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        # This is hardcoded to 1 for my use case but could be expanded in future
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    blink_count = 0
    closed_streak = 0
    state = "OPEN"
    is_closed_latched = False   # count once per closed period

    t_last = time.time()
    fps_hist = deque(maxlen=30)

    window_title = args.title or f"Blink Detection ({'Webcam' if args.input=='webcam' else 'Screen Capture'}) #{os.getpid()}"

    try:
        while True:
            # ---- Grab a frame
            if args.input == "webcam":
                ok, frame = cap.read()
                if not ok:
                    break
                if args.flip:
                    frame = cv2.flip(frame, 1)
            else:
                frame = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)

            # ---- Optionally downscale for processing
            if 0.2 < args.scale < 1.0:
                frame_small = cv2.resize(frame, None, fx=args.scale, fy=args.scale,
                                         interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame

            rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            ear_l = ear_r = 0.0
            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0]
                h, w = frame_small.shape[:2]
                pts = np.array([(lm.x * w, lm.y * h) for lm in face.landmark], dtype=np.float32)

                ear_l = eye_aspect_ratio(pts, LEFT_EYE)
                ear_r = eye_aspect_ratio(pts, RIGHT_EYE)
                ear = (ear_l + ear_r) / 2.0

                if ear < args.ear_thresh:
                    closed_streak += 1
                    if not is_closed_latched and closed_streak >= args.min_frames:
                        blink_count += 1
                        is_closed_latched = True

                        # ---- Log blink event
                        event_ts = time.time()
                        log["blinks"].append({
                            "timestamp": event_ts,
                            "ear": float(ear),
                            "ear_l": float(ear_l),
                            "ear_r": float(ear_r)
                        })
                        write_log()
                    state = "CLOSED"
                else:
                    closed_streak = 0
                    is_closed_latched = False
                    state = "OPEN"
            else:
                state = "OPEN"
                closed_streak = 0
                is_closed_latched = False

            # ---- FPS
            now = time.time()
            fps = 1.0 / max(1e-6, (now - t_last))
            t_last = now
            fps_hist.append(fps)
            smoothed_fps = sum(fps_hist) / len(fps_hist)

            # ---- Draw + UI
            if not args.no_gui:
                current_unix = int(time.time())  # NEW: current UNIX time in seconds
                draw_overlay(frame, smoothed_fps, ear_l, ear_r, blink_count, state, region_desc, current_unix)
                cv2.imshow(window_title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
            else:
                # Headless: tiny sleep to avoid maxing CPU
                time.sleep(0.001)

    finally:        
        log["total_blinks"] = int(blink_count)
        # write once more on exit (in case nothing got logged yet)
        write_log()
        face_mesh.close()
        if not args.no_gui:
            cv2.destroyAllWindows()
        if cap is not None:
            cap.release()
        print(f"Blink log saved to: {json_path}")

if __name__ == "__main__":
    main()
