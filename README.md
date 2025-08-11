# Blink Detection
Richard Lay-Flurrie
University of Essex

## Introduction

This program automatically detects blinks in humans using a webcam or screen capture. Each time the person in shot closes their eyes, a blink is recorded. At the end of the session, the blinks are logged in unix time to a JSON file.

- The method is not flawless and may be affected by:
    - Eyes not clearly visible due to overly bright lighting
    - Reflective glasses
    - The person's gaze being too far from the centre of the footage
- Useful for collecting data from multiple sources simultaneously (e.g., multiple YouTube videos)
- Multi-source capture can be accessed by running `launch-multi.py`

## Startup Parameters

- `--input` screen/webcam
- `--camera-index` (int)
- `--flip` (store_true)
- `--region` (int, int, int, int)
- `--scale` (float)
- `--ear-thresh` (float)
- `--min-frames` (int)

For simultaneous multiple capture:

- `--out` (string)
- `--title` (string)
- `--no-gui` (store_true)