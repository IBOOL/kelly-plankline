# %%
import sys
import argparse
import configparser
from PIL import Image
import cv2  # still used to save images out
import os
import numpy as np
import csv
#from queue import Queue
#from threading import Thread
from multiprocessing import Process, Queue
import tqdm
import sqlite3

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

        # Detect regions
        mser = cv2.MSER_create(delta=int(config['segmentation']['delta']),
                               min_area=int(config['segmentation']['min_area']),
                                  max_area=int(config['segmentation']['max_area']),
                                    max_variation=config['segmentation']['max_variation'])
        regions, bboxes = mser.detectRegions(gray)
        area = bbox_area(bboxes)

        for x in range(len(bboxes)-1):
            for y in range(x+1, len(bboxes)):
                overlap = intersection([bboxes[x][0], bboxes[x][1], bboxes[x][0]+bboxes[x][2], bboxes[x][1] + bboxes[x][3]], [bboxes[y][0], bboxes[y][1], bboxes[y][0]+bboxes[y][2], bboxes[y][1] + bboxes[y][3]])
                if overlap * 1. / max(area[x], area[y]) > float(config['segmentation']['overlap']):
                    if area[x] > area[y]:
                        bboxes[y] = [0,0,0,0]
                    else:
                        bboxes[x] = [0,0,0,0]

        area = bbox_area(bboxes)
        name = frame.get_name()
        n = frame.get_n()
        with open(f'{name}statistics.csv', 'a', newline='\n') as outcsv:
            outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
            for i in range(len(bboxes)):
                if area[i] > 0:
                    x1 = bboxes[i][1]
                    x2 = bboxes[i][1] + bboxes[i][3]
                    y1 = bboxes[i][0]
                    y2 = bboxes[i][0] + bboxes[i][2]
                    size = max(bboxes[i][2:3])
                    
                    im = Image.fromarray(gray[x1:x2, y1:y2])
                    im_padded = Image.new(im.mode, (size, size), (255))
                    if bboxes[i][2] > bboxes[i][3]:
                        left = 0
                        top = (size - bboxes[i][3]) //2
                    else :
                        top = 0
                        left = (size - bboxes[i][2]) //2
                        
                    im_padded.paste(im, (left, top))
                    im_padded.save(f"{name}{n:05}-{i:05}.png")
                    stats = [name, n, i, bboxes[i][0] + bboxes[i][2]/2, bboxes[i][1] + bboxes[i][3]/2, bboxes[i][2], bboxes[i][3], area[i]]
                    outwritter.writerow(stats)


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
        with open(f'{name}statistics.csv', 'a', newline='\n') as outcsv:
            outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                if w*h > int(config['segmentation']['min_area']):
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
                    



# %%
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
    
    #con = sqlite3.connect(output_path + '/' + 'images.db')
    #con.execute("CREATE TABLE frame(ID INT PRIMARY KEY NOT NULL,frame INT, crop INT, image BLOB)")
    #con.commit()
    #con.close()

    video = cv2.VideoCapture(avi_path)
    #length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not video.isOpened():
        return
    
    with open(f'{output_path}statistics.csv', 'a', newline='\n') as outcsv:
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
    
    # Finish up by tarring all images:
    with tarfile.open(segmentation_dir + os.path.sep + filename + ".tar", "w:") as tar:
        tar.add(output_path)

directory = '../../raw/camera1/GAK_202207/'

config = {
    'general' : {
        'dir_permissions' : 511,
        'dry_run' : 'False'
    },
    'segmentation' : {
        'basename' : 'REG',
        'min_area' : 200,
        'flatfield_q' : 0.05
    },
    'classification' : {
        'model_name' : 'Alpha',
        'model_dir' : '../../model',
        'scnn_instances' : 1,
        'fast_scratch' : '/tmp',
        'batchsize' : 64,
        'image_size' : 128
    },
    'training' : {
        'scnn_dir' : '../../training/20231002',
        'model_name': 'Gamma',
        'model_path': '../../model/',
        'image_size': '128',
        'start' : 10,
        'stop' : 100,
        'validationSetRatio' : 0.2,
        'batchsize' : 16,
        'seed': 123
    }
}

v_string = "V2023.11.08"
print(f"Starting Segmentation Script {v_string}")

## Determine directories
raw_dir = os.path.abspath(directory) # /media/plankline/Data/raw/Camera1/Transect1
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
#num_threads = 2
max_queue = num_threads * 4
q = Queue(maxsize=int(max_queue))

for i in range(num_threads):
    worker = Process(target=process_frame, args=(q, config,), daemon=True)
    worker.start()
    
print(num_threads)

for av in tqdm.tqdm(avis):
    process_avi(av, segmentation_dir, config, q)

print('Joining')
worker.join(timeout=10)





