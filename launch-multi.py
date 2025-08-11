import sys
import subprocess
import uuid

PY = sys.executable
SCRIPT = "blink-detection.py"

def run_region(name, region, extra=None):
    extra = extra or []
    out = f"{name}_{uuid.uuid4().hex}.json"
    cmd = [
        PY, SCRIPT,
        "--input", "screen",
        "--region", str(region[0]), str(region[1]), str(region[2]), str(region[3]),
        "--title", name,
        "--out", out,
        "--scale", "0.75",
        "--ear-thresh", "0.21",
        "--min-frames", "1",
        "--max-faces", "1",
    ] + extra
    return subprocess.Popen(cmd)

if __name__ == "__main__":
    regions = [
        ("TopLeft",  (0,   0,   640, 480)),
        ("TopRight", (640, 0,   640, 480)),
        ("Bottom",   (0,   480, 1280, 540)),
    ]

    processes = [run_region(name, r) for name, r in regions]

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
