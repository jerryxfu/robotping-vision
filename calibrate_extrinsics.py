"""
calibrate_extrinsics_charuco.py
-------------------------------
Recover each camera's pose (R, t) in a shared world frame defined by a
stationary ChArUco board. Run with BOTH phones mounted and pointed at the
same board placement. The board's origin becomes world origin.

Paste your per-camera intrinsics (K, dist) below first.

Usage:
    python calibrate_extrinsics_charuco.py \
        --url0 rtsp://172.20.10.1:554/stream \
        --url1 rtsp://172.20.10.2:554/stream

Place board where BOTH cameras see it, hold steady, press SPACE.
It solves both poses and prints R/t for each. q to quit.
"""

import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
import argparse
import cv2
import numpy as np

# --- BOARD (same as intrinsics) ---
SQUARES_X = 11
SQUARES_Y = 8
SQUARE_MM = 18.2  # use your measured on-screen size
MARKER_RATIO = 0.75
DICT = cv2.aruco.DICT_4X4_50
# -----------------------------------

# --- PASTE YOUR INTRINSICS ---
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
# ------------------------------

MARKER_MM = SQUARE_MM * MARKER_RATIO
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARE_MM, MARKER_MM, aruco_dict)
detector = cv2.aruco.CharucoDetector(board)


def grab(cap):
    ok, frame = cap.read()
    return frame if ok else None


def solve_pose(gray, K, dist):
    """Detect board and solvePnP -> (rvec, tvec, n_corners) or None."""
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 6:
        return None
    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    if obj_pts is None or len(obj_pts) < 6:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        return None
    return rvec, tvec, len(ch_ids), ch_corners, ch_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url0", required=True)
    ap.add_argument("--url1", required=True)
    args = ap.parse_args()

    cap0 = cv2.VideoCapture(args.url0, cv2.CAP_FFMPEG)
    cap1 = cv2.VideoCapture(args.url1, cv2.CAP_FFMPEG)
    for c in (cap0, cap1):
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not (cap0.isOpened() and cap1.isOpened()):
        raise RuntimeError("Could not open both streams")

    print("Place board so BOTH cameras see it. SPACE to solve. q to quit.")

    while True:
        f0, f1 = grab(cap0), grab(cap1)
        if f0 is None or f1 is None:
            continue
        g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

        # Live preview of detection on both.
        for f, g, name in ((f0, g0, "cam0"), (f1, g1, "cam1")):
            ch_c, ch_i, _, _ = detector.detectBoard(g)
            n = 0 if ch_i is None else len(ch_i)
            if n > 0:
                cv2.aruco.drawDetectedCornersCharuco(f, ch_c, ch_i)
            color = (0, 255, 0) if n >= 6 else (0, 0, 255)
            cv2.putText(f, f"{name}: {n} corners", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(name, f)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            r0 = solve_pose(g0, K0, dist0)
            r1 = solve_pose(g1, K1, dist1)
            if r0 is None or r1 is None:
                print("  board not clearly seen by BOTH - reposition")
                continue
            rvec0, tvec0, n0 = r0[0], r0[1], r0[2]
            rvec1, tvec1, n1 = r1[0], r1[1], r1[2]
            R0, _ = cv2.Rodrigues(rvec0)
            R1, _ = cv2.Rodrigues(rvec1)
            print(f"\n=== EXTRINSICS (cam0:{n0} cam1:{n1} corners) ===")
            print(f"R0 = np.array({np.array2string(R0, separator=', ')})")
            print(f"t0 = np.array({np.array2string(tvec0, separator=', ')})")
            print(f"R1 = np.array({np.array2string(R1, separator=', ')})")
            print(f"t1 = np.array({np.array2string(tvec1, separator=', ')})")
            # Sanity: camera positions in world (= -R^T t). Distance between
            # them should match your real baseline (mm).
            C0 = (-R0.T @ tvec0).ravel()
            C1 = (-R1.T @ tvec1).ravel()
            baseline = np.linalg.norm(C0 - C1)
            print(f"cam0 world pos: {C0.round(1)} mm")
            print(f"cam1 world pos: {C1.round(1)} mm")
            print(f"baseline (cam-to-cam distance): {baseline:.1f} mm")
            print("Compare baseline to the real distance between phones!")
            print("===============================================\n")

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
