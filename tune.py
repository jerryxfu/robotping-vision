import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
import argparse, cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
args = ap.parse_args()

cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("tuner", cv2.WINDOW_NORMAL)
for n, v, hi in [("H low", 5, 179), ("H high", 20, 179), ("S low", 120, 255),
                 ("S high", 255, 255), ("V low", 120, 255), ("V high", 255, 255)]:
    cv2.createTrackbar(n, "tuner", v, hi, lambda x: None)

g = lambda n: cv2.getTrackbarPos(n, "tuner")
print("Drag sliders so ONLY the ball is white. 's' to print, 'q' to quit.")
while True:
    for _ in range(5): cap.grab()
    ok, frame = cap.retrieve()
    if not ok: continue
    lower = np.array([g("H low"), g("S low"), g("V low")])
    upper = np.array([g("H high"), g("S high"), g("V high")])
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow("feed", frame)
    cv2.imshow("mask", mask)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    if k == ord("s"):
        print(f"LOWER = np.array({list(lower)})")
        print(f"UPPER = np.array({list(upper)})")
cap.release();
cv2.destroyAllWindows()
