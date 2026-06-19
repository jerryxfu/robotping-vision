"""
fuse.py
-------
Triangulate the ball's 3D world position from two phone streams.
Loads intrinsics (hardcoded here) + extrinsics (from extrinsics.py),
detects the orange ball in each stream, triangulates to world xyz.

Output: (x, y, z) in mm, in the board-origin world frame -> for MATLAB.

    python fuse.py --url0 rtsp://172.20.10.1:554/stream \
                   --url1 rtsp://172.20.10.2:554/stream
"""

import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"

import argparse
import threading
import time
import cv2
import numpy as np
from extrinsics import R0, t0, R1, t1  # saved by the extrinsics script

# --- INTRINSICS (paste both cameras' real values) ---
K0 = np.array([[505.34254716, 0., 239.06364843],
               [0., 505.08527838, 321.69757632],
               [0., 0., 1.]])
dist0 = np.array([[2.20202595e-01, -1.32506763e+00, 1.05424188e-04,
                   1.42208007e-04, 2.08309395e+00]])
K1 = np.array([[510.76158671, 0., 240.76067209],
               [0., 510.4488106, 322.07845801],
               [0., 0., 1.]])
dist1 = np.array([[2.04017309e-01, -1.48397188e+00, 3.99659594e-04, 3.97122613e-05,
                   2.62261143e+00]])
# ----------------------------------------------------

# --- BALL HSV (from your tuned detector) ---
# slower = [np.int64(0), np.int64(169), np.int64(138)]
# upper = [np.int64(56), np.int64(255), np.int64(254)]

LOWER = np.array([0, 169, 138])  # TODO: your tuned values
UPPER = np.array([56, 255, 254])
MIN_AREA = 30
# -------------------------------------------

# Projection matrices: P = K [R|t], world -> image. Built once.
P0 = K0 @ np.hstack([R0, t0])
P1 = K1 @ np.hstack([R1, t1])


class Cam(threading.Thread):
    def __init__(self, url, name):
        super().__init__(daemon=True)
        self.url, self.name = url, name
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame, self.stamp, self.ok = None, None, self.cap.isOpened()
        self.running = True

    def run(self):
        while self.running:
            for _ in range(5):
                self.cap.grab()
            ok, f = self.cap.retrieve()
            if ok:
                self.frame, self.stamp = f, time.perf_counter()

    def stop(self):
        self.running = False
        self.cap.release()


def detect_ball(frame):
    """Return undistortable pixel (cx, cy) or None."""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)
    mask = cv2.erode(mask, None, 1)
    mask = cv2.dilate(mask, None, 2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None
    (x, y), _ = cv2.minEnclosingCircle(c)
    return np.array([[x, y]], dtype=np.float32)


def undistort_point(pt, K, dist):
    """Undistort a pixel and return it back in pixel coords."""
    und = cv2.undistortPoints(pt.reshape(1, 1, 2), K, dist, P=K)
    return und.reshape(2)


def triangulate(p0, p1):
    """Two undistorted pixels -> 3D world point (mm)."""
    pts4d = cv2.triangulatePoints(P0, P1,
                                  p0.reshape(2, 1), p1.reshape(2, 1))
    xyz = (pts4d[:3] / pts4d[3]).ravel()
    return xyz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url0", required=True)
    ap.add_argument("--url1", required=True)
    args = ap.parse_args()

    cam0 = Cam(args.url0, "cam0")
    cam1 = Cam(args.url1, "cam1")
    for c in (cam0, cam1):
        if not c.ok:
            raise RuntimeError(f"{c.name} failed to open")
        c.start()

    print("Fusing. Ball xyz in mm (world frame). q to quit.")
    try:
        while True:
            f0, f1 = cam0.frame, cam1.frame
            if f0 is None or f1 is None:
                time.sleep(0.005)
                continue

            d0 = detect_ball(f0)
            d1 = detect_ball(f1)

            if d0 is not None and d1 is not None:
                u0 = undistort_point(d0, K0, dist0)
                u1 = undistort_point(d1, K1, dist1)
                xyz = triangulate(u0, u1)
                # >>> THIS is the fused point for MATLAB: (x, y, z) mm + time
                print(f"xyz = ({xyz[0]:7.1f}, {xyz[1]:7.1f}, {xyz[2]:7.1f}) mm")

            # Optional: show both feeds with detection marked.
            for f, d, name in ((f0, d0, "cam0"), (f1, d1, "cam1")):
                disp = f.copy()
                if d is not None:
                    p = d.ravel().astype(int)
                    cv2.circle(disp, tuple(p), 6, (0, 255, 0), 2)
                cv2.imshow(name, disp)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam0.stop()
        cam1.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
