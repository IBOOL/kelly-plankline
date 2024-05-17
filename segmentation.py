# %%
import sys
import argparse
import configparser
from PIL import Image
import cv2  # still used to save images out
import os
import numpy as np
import csv
from multiprocessing import Process, Queue
import tqdm
import shutil

class Frame:
    def __init__(self, fpath, name, frame, n):
        self.fpath = fpath  # default is 0 for primary camera
        self.name = name
        self.frame = frame
        self.n = n

    # method for returning latest read frame
    def read(self):
        return self.frame

    # method called to stop reading frames
    def get_n(self):
        return self.n

    def get_name(self):
        return self.name


def bbox_area(bbox):
    res = []
    for p in bbox:
        res.append(abs(p[2]*p[3]))
    return res


def intersection(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

    return interArea


def process_frame(q, config): ## TODO: write metadata file
    """
    This function processes each frame (provided as cv2 image frame) for flatfielding and segmentation. The steps include
    1. Flatfield intensities as indicated
    2. Segment the image using cv2 MSER algorithmn.
    3. Remove strongly overlapping bounding boxes
    4. Save cropped targets.
    """

    while True:
        frame = q.get()
        if config['general']['dry_run'] == 'True':
            print('.')
            return
        
        ## Read img and flatfield
        gray = cv2.cvtColor(frame.read(), cv2.COLOR_BGR2GRAY)
        gray = np.array(gray)
        field = np.quantile(gray, q=float(config['segmentation']['flatfield_q']), axis=0)
        gray = (gray / field.T * 255.0)
        gray = gray.clip(0,255).astype(np.uint8)

        # Apply Otsu's threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        name = frame.get_name()
        n = frame.get_n()
        stats = []

        with open(f'{name}[:-1]statistics.csv', 'a', newline='\n') as outcsv:
            outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                if 2*w + 2*h > int(config['segmentation']['min_perimeter']):
                    size = max(w, h)
                    im = Image.fromarray(gray[y:(y+h), x:(x+w)])
                    im_padded = Image.new(im.mode, (size, size), (255))
                    
                    if (w > h):
                        left = 0
                        top = (size - h)//2
                    else:
                        left = (size - w)//2
                        top = 0
                    im_padded.paste(im, (left, top))
                    im_padded.save(f"{name}{n:05}-{i:05}.png")
                stats = [name, n, i, x + w/2, y + h/2, w, h, w*h]
                outwritter.writerow(stats)
                    

def process_avi(avi_path, segmentation_dir, config, q):
    """
    This function will take an avi filepath as input and perform the following steps:
    1. Create output file structures/directories
    2. Load each frame, pass it through flatfielding and sequentially save segmented targets
    """

    # segmentation_dir: /media/plankline/Data/analysis/segmentation/Camera1/segmentation/Transect1-REG
    _, filename = os.path.split(avi_path)
    output_path = segmentation_dir + os.path.sep + filename + os.path.sep
    os.makedirs(output_path, exist_ok=True)
    

    video = cv2.VideoCapture(avi_path)
    if not video.isOpened():
        return
    
    with open(f'{output_path[:-1]} statistics.csv', 'a', newline='\n') as outcsv:
        outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
        outwritter.writerow(['file', 'frame', 'crop', 'x', 'y', 'w', 'h', 'area'])

    n = 1
    while True:
        ret, frame = video.read()
        if ret:
            q.put(Frame(avi_path, output_path, frame, n), block = True)
            n += 1
        else:
            break


if __name__ == "__main__":
    
    directory = '../../raw/camera0/test1'

    config = {
        'general' : {
            'dir_permissions' : 511,
            'dry_run' : 'False'
        },
        'segmentation' : {
            'basename' : 'REG',
            'min_periemter' : 4*20,
            'flatfield_q' : 0.05
        }
    }

    v_string = "V2024.04.26"
    print(f"Starting Segmentation Script {v_string}")

    ## Determine directories
    raw_dir = os.path.abspath(directory) # /media/plankline/Data/raw/Camera0/test1
    segmentation_dir = raw_dir.replace("raw", "analysis") # /media/plankline/Data/analysis/Camera1/Transect1
    segmentation_dir = segmentation_dir.replace("camera0/", "camera0/segmentation/") # /media/plankline/Data/analysis/Camera1/Transect1
    segmentation_dir = segmentation_dir.replace("camera1/", "camera1/segmentation/") # /media/plankline/Data/analysis/Camera1/segmentation/Transect1
    segmentation_dir = segmentation_dir.replace("camera2/", "camera2/segmentation/") # /media/plankline/Data/analysis/Camera1/segmentation/Transect1
    segmentation_dir = segmentation_dir.replace("camera3/", "camera3/segmentation/") # /media/plankline/Data/analysis/Camera1/segmentation/Transect1
        
    segmentation_dir = segmentation_dir + f"-{config['segmentation']['basename']}" # /media/plankline/Data/analysis/segmentation/Camera1/segmentation/Transect1-REG
    os.makedirs(segmentation_dir, int(config['general']['dir_permissions']), exist_ok = True)


    avis = []
    avis = [os.path.join(raw_dir, avi) for avi in os.listdir(raw_dir) if avi.endswith(".avi")]
    print(f"Number of AVIs found: {len(avis)}")

    ## Prepare workers for receiving frames
    num_threads = os.cpu_count() - 1
    max_queue = num_threads * 4
    q = Queue(maxsize=int(max_queue))

    for i in range(num_threads):
        worker = Process(target=process_frame, args=(q, config,), daemon=True)
        worker.start()
        
    print(num_threads)

    for av in tqdm.tqdm(avis):
        process_avi(av, segmentation_dir, config, q)

    for av in avis:
        _, filename = os.path.split(av)
        output_path = segmentation_dir + os.path.sep + filename + os.path.sep
        shutil.make_archive(segmentation_dir + os.path.sep + filename, 'zip', output_path)
        shutil.rmtree(output_path, ignore_errors=True)

    print('Joining')
    worker.join(timeout=10)


