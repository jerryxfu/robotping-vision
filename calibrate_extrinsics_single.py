"""
calibrate_extrinsics_single.py
------------------------------
Recover each camera's pose (R, t) in a shared world frame, ONE camera at
a time. Avoids the two-stream network lag.

CRITICAL: the board must NOT move between capturing cam0 and cam1. Lock it
down. Both cameras must be mounted in their final positions before you
start, and must not move either.

Run once per camera with its URL and an id:
    python calibrate_extrinsics_single.py --url rtsp://172.20.10.1:554/stream --cam 0
    python calibrate_extrinsics_single.py --url rtsp://172.20.10.2:554/stream --cam 1

Each run appends its result to extrinsics.py which the fusion script loads.
SPACE = solve & save pose   q = quit
"""

import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"

import argparse
import cv2
import numpy as np

# --- BOARD (same as intrinsics) ---
SQUARES_X = 11
SQUARES_Y = 8
SQUARE_MM = 15.0  # your measured on-screen square size
MARKER_RATIO = 0.75
DICT = cv2.aruco.DICT_4X4_50
# -----------------------------------

# --- PER-CAMERA INTRINSICS ---
# Fill in both. cam0 confirmed; paste cam1's real numbers.
INTRINSICS = {
    0: {
        "K": np.array([[505.34254716, 0., 239.06364843],
                       [0., 505.08527838, 321.69757632],
                       [0., 0., 1.]]),
        "dist": np.array([[2.20202595e-01, -1.32506763e+00, 1.05424188e-04,
                           1.42208007e-04, 2.08309395e+00]]),
    },
    1: {
        # TODO: paste camera 1's real K and dist here
        "K": np.array([[510.76158671, 0., 240.76067209],
                       [0., 510.4488106, 322.07845801],
                       [0., 0., 1.]]),
        "dist": np.array([[2.04017309e-01, -1.48397188e+00, 3.99659594e-04, 3.97122613e-05,
                           2.62261143e+00]]),
    },
}
# ------------------------------

MARKER_MM = SQUARE_MM * MARKER_RATIO
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARE_MM, MARKER_MM, aruco_dict)
detector = cv2.aruco.CharucoDetector(board)

OUT_FILE = "extrinsics.py"


def grab(cap):
    for _ in range(5):
        cap.grab()
    ok, frame = cap.retrieve()
    return frame if ok else None


def save_pose(cam_id, R, t):
    """Append this camera's pose to extrinsics.py as R{id}, t{id}."""
    header = "import numpy as np\n\n" if not os.path.exists(OUT_FILE) else ""
    with open(OUT_FILE, "a") as f:
        if header:
            f.write(header)
        f.write(f"R{cam_id} = np.array({np.array2string(R, separator=', ')})\n")
        f.write(f"t{cam_id} = np.array({np.array2string(t, separator=', ')})\n\n")
    print(f"  saved R{cam_id}, t{cam_id} to {OUT_FILE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--cam", type=int, required=True, choices=[0, 1])
    args = ap.parse_args()

    K = INTRINSICS[args.cam]["K"]
    dist = INTRINSICS[args.cam]["dist"]

    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.url}")

    print(f"Camera {args.cam}: aim at the FIXED board. SPACE to solve. q quit.")
    print("Board must stay put across BOTH cameras' captures!")

    last_solution = None
    while True:
        frame = grab(cap)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        n = 0 if ch_ids is None else len(ch_ids)

        disp = frame.copy()
        if n > 0:
            cv2.aruco.drawDetectedCornersCharuco(disp, ch_corners, ch_ids)
        color = (0, 255, 0) if n >= 6 else (0, 0, 255)
        cv2.putText(disp, f"cam{args.cam}: {n} corners", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if last_solution is not None:
            cv2.putText(disp, "SAVED - q to exit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(f"extrinsics cam{args.cam}", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if n < 6:
                print("  not enough corners - reposition / improve view")
                continue
            obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
            if obj_pts is None or len(obj_pts) < 6:
                print("  point matching failed - try again")
                continue
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
            if not ok:
                print("  solvePnP failed")
                continue
            R, _ = cv2.Rodrigues(rvec)
            C = (-R.T @ tvec).ravel()
            print(f"\n=== cam{args.cam} pose ({n} corners) ===")
            print(f"camera world position: {C.round(1)} mm")
            print(f"distance from board origin: {np.linalg.norm(C):.1f} mm")
            save_pose(args.cam, R, tvec)
            print("Run the OTHER camera now (board unmoved). q to exit.\n")
            last_solution = (R, tvec)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
