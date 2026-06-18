import cv2

# Try 0 first; if it's the Mac's built-in camera, change to 1, then 2.
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Couldn't open camera at that index — try a different index.")
    raise SystemExit

print("Opened. Press q to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("No frame")
        break
    cv2.imshow("continuity test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
