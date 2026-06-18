import cv2, threading, time


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
            ok, f = self.cap.read()
            if ok:
                self.frame, self.stamp = f, time.perf_counter()

    def stop(self):
        self.running = False
        self.cap.release()


# put your two phone URLs here
urls = [
    ("rtsp://172.20.10.1:554/stream", "cam0"),
    ("rtsp://172.20.10.2:554/stream", "cam1"),
]

cams = [Cam(u, n) for u, n in urls]
for c in cams:
    if not c.ok:
        print(f"{c.name} FAILED to open {c.url}")
for c in cams:
    c.start()

print("Both running? Press q to quit.")
try:
    while True:
        for c in cams:
            if c.frame is not None:
                f = c.frame.copy()
                cv2.putText(f, f"{c.name} t={c.stamp:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(c.name, f)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    for c in cams:
        c.stop()
    cv2.destroyAllWindows()
