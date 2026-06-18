"""
Stage 1: Ping Pong Ball Detector (single phone, RTSP)
-----------------------------------------------------
Pulls an RTSP stream from a phone, detects the orange ball via HSV
thresholding, outputs the 2D centroid per frame.

Cross-platform (Windows / Mac). Requires: opencv-python, numpy
    pip install opencv-python numpy

Usage:
    python main.py --url rtsp://<phone-ip>:<port>/<path> --tune

Controls:  t = toggle tuner   s = print HSV bounds   q = quit
"""
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
import argparse, threading, time, cv2
import numpy as np


class RTSPStream:
    """Threaded RTSP reader. read() blocks, so we read in a background
    thread and always hand out the latest frame to stay near real-time."""

    def __init__(self, source, is_index=False):
        if is_index:
            self.cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")
        self.lock = threading.Lock()
        self.frame, self.stamp, self.running = None, None, True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01);
                continue
            ts = time.perf_counter()
            with self.lock:
                self.frame, self.stamp = frame, ts

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, None
            return self.frame.copy(), self.stamp

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


DEFAULT_LOWER = np.array([5, 120, 120])
DEFAULT_UPPER = np.array([20, 255, 255])
MIN_AREA = 30


def make_tuner():
    cv2.namedWindow("tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("tuner", 400, 300)
    for name, val, hi in [
        ("H low", DEFAULT_LOWER[0], 179), ("H high", DEFAULT_UPPER[0], 179),
        ("S low", DEFAULT_LOWER[1], 255), ("S high", DEFAULT_UPPER[1], 255),
        ("V low", DEFAULT_LOWER[2], 255), ("V high", DEFAULT_UPPER[2], 255),
    ]:
        cv2.createTrackbar(name, "tuner", int(val), hi, lambda x: None)


def read_tuner():
    g = lambda n: cv2.getTrackbarPos(n, "tuner")
    lower = np.array([g("H low"), g("S low"), g("V low")])
    upper = np.array([g("H high"), g("S high"), g("V high")])
    return lower, upper


def detect_ball(frame, lower, upper):
    """Return (cx, cy, radius, mask) or (None, None, None, mask)."""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, mask
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None, None, None, mask
    (x, y), radius = cv2.minEnclosingCircle(c)
    return float(x), float(y), float(radius), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="rtsp://... URL")
    ap.add_argument("--cam", type=int, help="camera index (Continuity Camera, e.g. 1)")
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()

    if args.cam is not None:
        stream = RTSPStream(args.cam, is_index=True)
    else:
        stream = RTSPStream(args.url, is_index=False)

    tuning = args.tune
    if tuning:
        make_tuner()
    lower, upper = DEFAULT_LOWER.copy(), DEFAULT_UPPER.copy()
    fps_t, fps_n, fps = time.perf_counter(), 0, 0.0

    print("Streaming. Keys: [t] tuner  [s] print HSV  [q] quit")
    try:
        while True:
            frame, stamp = stream.read()
            if frame is None:
                time.sleep(0.005);
                continue
            if tuning:
                lower, upper = read_tuner()

            cx, cy, r, mask = detect_ball(frame, lower, upper)
            if cx is not None:
                cv2.circle(frame, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                # >>> pipeline output: (cx, cy, stamp) -> triangulation stage
                cv2.putText(frame, f"({cx:.0f}, {cy:.0f})",
                            (int(cx) + 10, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            fps_n += 1
            if fps_n >= 15:
                now = time.perf_counter()
                fps = fps_n / (now - fps_t)
                fps_t, fps_n = now, 0
            cv2.putText(frame, f"{fps:.0f} FPS", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("detection", frame)
            if tuning:
                cv2.imshow("mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t"):
                tuning = not tuning
                if tuning:
                    make_tuner()
                else:
                    cv2.destroyWindow("tuner")
                    cv2.destroyWindow("mask")
            elif key == ord("s"):
                print(f"lower = {list(lower)}\nupper = {list(upper)}")
    finally:
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
