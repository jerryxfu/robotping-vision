"""
Triangulate the ball's 3D world position from two phone streams.
Loads intrinsics (hardcoded here) + extrinsics (from extrinsics.py),
detects the orange ball in each stream, triangulates to world xyz, and
shows a live 3D view (cameras, world origin, ball + trail).

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
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection

from extrinsics import R0, t0, R1, t1  # saved by the extrinsics script

# --- INTRINSICS ---
K0 = np.array([[492.49721348, 0., 238.35910301],
               [0., 492.63279911, 324.71922652],
               [0., 0., 1.]])

dist0 = np.array([[1.60704667e-01, -7.94847520e-01, 1.02455418e-03, -4.04626847e-04,
                   1.06560000e+00]])

K1 = np.array([[496.33199581, 0., 239.90805986],
               [0., 497.5526016, 326.47298346],
               [0., 0., 1.]])

dist1 = np.array([[2.57753368e-01, -1.78001331e+00, 1.19580866e-03, -4.28233049e-04,
                   3.31371106e+00]])

# ------------------

# --- BALL HSV (your tuned values) ---
# lower = [np.int64(69), np.int64(25), np.int64(90)]
# upper = [np.int64(146), np.int64(184), np.int64(205)]

LOWER = np.array([69, 25, 90])
UPPER = np.array([146, 184, 205])
MIN_AREA = 30
# -------------------------------------

# Projection matrices: P = K [R|t], world -> image. Built once.
P0 = K0 @ np.hstack([R0, t0])
P1 = K1 @ np.hstack([R1, t1])

# Camera world positions: C = -R^T t  (computed once, used by viewer).
C0 = (-R0.T @ t0).ravel()
C1 = (-R1.T @ t1).ravel()


def camera_forward(R):
    """Camera looks along +Z in its own frame; world dir = R^T @ [0,0,1]."""
    return R.T @ np.array([0, 0, 1.0])


class Live3D:
    """Live matplotlib 3D scene: origin axes, cameras, ball + trail.

    Must be created and updated from the MAIN thread (matplotlib rule).
    """

    def __init__(self, trail_len=40):
        plt.ion()
        self.fig = plt.figure(figsize=(8, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ball = None
        self.trail_artist = None
        self.trail = deque(maxlen=trail_len)
        self._draw_static()
        self.fig.canvas.draw()
        plt.show(block=False)

    def _draw_static(self):
        ax = self.ax
        L = 100  # axis length, mm
        ax.quiver(0, 0, 0, L, 0, 0, color="r", label="X")
        ax.quiver(0, 0, 0, 0, L, 0, color="g", label="Y")
        ax.quiver(0, 0, 0, 0, 0, L, color="b", label="Z")
        ax.scatter(0, 0, 0, c="k", s=50, marker="o")
        ax.text(0, 0, 0, "  origin")

        for C, R, name, col in ((C0, R0, "cam0", "purple"),
                                (C1, R1, "cam1", "orange")):
            ax.scatter(*C, c=col, s=80, marker="^")
            ax.text(*C, f"  {name}")
            fwd = camera_forward(R) * 80
            ax.quiver(C[0], C[1], C[2], fwd[0], fwd[1], fwd[2],
                      color=col, alpha=0.5)

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.legend(loc="upper left")
        self._equal_aspect()

    def _equal_aspect(self):
        pts = np.array([C0, C1, [0, 0, 0]])
        lo, hi = pts.min(0) - 100, pts.max(0) + 200
        mid = (lo + hi) / 2
        r = (hi - lo).max() / 2
        self.ax.set_xlim(mid[0] - r, mid[0] + r)
        self.ax.set_ylim(mid[1] - r, mid[1] + r)
        self.ax.set_zlim(mid[2] - r, mid[2] + r)

    def update(self, xyz):
        self.trail.append(xyz)
        if self.ball is not None:
            self.ball.remove()
        if self.trail_artist is not None:
            self.trail_artist.remove()
            self.trail_artist = None

        self.ball = self.ax.scatter(*xyz, c="red", s=120, marker="o")
        if len(self.trail) > 1:
            arr = np.array(self.trail)
            self.trail_artist = self.ax.plot(
                arr[:, 0], arr[:, 1], arr[:, 2],
                c="red", alpha=0.4, linewidth=1)[0]

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


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
    """Return pixel (cx, cy) as a (1,2) array, or None."""
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
    und = cv2.undistortPoints(pt.reshape(1, 1, 2), K, dist, P=K)
    return und.reshape(2)


def triangulate(p0, p1):
    pts4d = cv2.triangulatePoints(P0, P1, p0.reshape(2, 1), p1.reshape(2, 1))
    return (pts4d[:3] / pts4d[3]).ravel()


def to_world(xyz):
    """Correct the X<->Y swap from the board's coordinate convention.
    Swap X and Y; negate Z to keep the frame right-handed.
"""
    x, y, z = xyz
    return np.array([y, x, -z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url0", required=True)
    ap.add_argument("--url1", required=True)
    args = ap.parse_args()

    viewer = Live3D()

    cam0 = Cam(args.url0, "cam0")
    cam1 = Cam(args.url1, "cam1")
    for c in (cam0, cam1):
        if not c.ok:
            raise RuntimeError(f"{c.name} failed to open")
        c.start()

    print("Fusing. Ball xyz in mm (world frame). q in a CV window to quit.")
    last_3d = 0.0
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
                xyz = to_world(xyz)
                print(f"xyz = ({xyz[0]:7.1f}, {xyz[1]:7.1f}, {xyz[2]:7.1f}) mm")
                # Throttle 3D redraw to ~15 Hz so it doesn't choke the loop.
                now = time.perf_counter()
                if now - last_3d > 0.066:
                    viewer.update(xyz)
                    last_3d = now

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
        plt.close("all")


if __name__ == "__main__":
    main()
