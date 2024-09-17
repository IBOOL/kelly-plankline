# importing required libraries
import cv2
import time
from threading import Thread  # library for implementing multi-threaded processing
from queue import Queue
from threading import Thread
import numpy as np

# defining a helper class for implementing multi-threaded processing
class Frame:
    def __init__(self, fpath, frame, n):
        self.fpath = fpath  # default is 0 for primary camera
        self.frame = frame
        self.n = n

    # method for returning latest read frame
    def read(self):
        return self.frame

    # method called to stop reading frames
    def get_n(self):
        return self.n


def do_stuff(q):
    frames_dir = "C:/Users/Tom Kelly/Desktop/plankline/raw/camera0/test/tmp/"
    while True:
        frame = q.get()
        n = frame.get_n()
        frame = frame.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.array(gray)
        field = np.quantile(gray, q=0.02, axis=0) + 1.
        gray = (gray / field.T * 255.0)
        gray = gray.clip(0, 255).astype(np.uint8)
        cv2.imwrite(f'{frames_dir}{n:06}.png', gray)
        q.task_done()



max_queue = 64
num_threads = 1
q = Queue(maxsize=0)

for i in range(num_threads):
    worker = Thread(target=do_stuff, args=(q,), daemon=True)
    worker.start()

fpath = "C:/Users/Tom Kelly/Desktop/plankline/raw/camera0/test/Camera3_VIPF-342-2022-07-22-22-00-05.292.avi"

start = time.time()
video = cv2.VideoCapture(fpath)
n = 1
while True:
    if q.qsize() < max_queue:
        ret, frame = video.read()
        if ret:
            q.put(Frame(fpath, frame, n))
            n += 1
        else:
            break

q.join()
worker.join(timeout=1)

end = time.time()
print(f"Elapsed time: {end-start:.2f} seconds.")