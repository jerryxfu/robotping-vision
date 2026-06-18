"""
calibrate_intrinsics_charuco.py
-------------------------------
Per-camera intrinsic calibration using a ChArUco board from a phone
RTSP stream. Modern OpenCV API (>= 4.7).

Board (from calib.io): 11 cols x 8 rows squares, DICT_4X4, 15mm checker.
MEASURE your on-screen square with a ruler and set SQUARE_MM below.

Usage:
    python calibrate_intrinsics_charuco.py --url rtsp://172.20.10.1:554/stream

SPACE = capture view (when board detected)
c     = compute & print K + dist
q     = quit
Aim for 15-20 views: vary angle, distance, screen position.
"""

import argparse
import cv2
import numpy as np

# --- CONFIGURE FROM YOUR calib.io BOARD ---
SQUARES_X = 11  # number of squares horizontally (columns)
SQUARES_Y = 8  # number of squares vertically (rows)
SQUARE_MM = 18.8  # MEASURE the on-screen square with a ruler! mm
MARKER_RATIO = 0.75  # marker size as fraction of square (calib.io default)
DICT = cv2.aruco.DICT_4X4_50  # try _100 / _250 if nothing detects
# ------------------------------------------

MARKER_MM = SQUARE_MM * MARKER_RATIO

aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARE_MM, MARKER_MM, aruco_dict)
detector = cv2.aruco.CharucoDetector(board)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.url}")

    all_charuco_corners = []
    all_charuco_ids = []
    all_object_points = []
    all_image_points = []
    img_size = None

    print("SPACE = capture   c = compute   q = quit")
    print("Get 15-20 varied views (angle, distance, position).")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        # Detect board: returns charuco corners/ids + raw marker corners/ids.
        ch_corners, ch_ids, m_corners, m_ids = detector.detectBoard(gray)

        disp = frame.copy()
        n = 0 if ch_ids is None else len(ch_ids)
        if n > 0:
            cv2.aruco.drawDetectedCornersCharuco(disp, ch_corners, ch_ids)
            cv2.putText(disp, f"DETECTED {n} corners - SPACE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "no board", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(disp, f"captured: {len(all_charuco_corners)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("intrinsics", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" ") and n >= 6:  # need enough corners to be useful
            # Convert detected charuco corners to object/image point pairs.
            obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
            if obj_pts is not None and len(obj_pts) >= 6:
                all_object_points.append(obj_pts)
                all_image_points.append(img_pts)
                all_charuco_corners.append(ch_corners)
                all_charuco_ids.append(ch_ids)
                print(f"  captured view {len(all_object_points)} "
                      f"({len(obj_pts)} corners)")
        elif key == ord("c"):
            if len(all_object_points) < 8:
                print(f"  need >=8 views, have {len(all_object_points)}")
                continue
            print("Computing...")
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                all_object_points, all_image_points, img_size, None, None)
            print("\n=== CALIBRATION RESULT ===")
            print(f"RMS reprojection error: {ret:.4f} px  "
                  f"(<0.5 great, <1.0 fine, >1.5 redo)")
            print(f"\nK = np.array({np.array2string(K, separator=', ')})")
            print(f"\ndist = np.array({np.array2string(dist, separator=', ')})")
            print(f"\nimage_size = {img_size}")
            print("==========================\n")
            print("Paste K and dist into the main script. q to exit.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
