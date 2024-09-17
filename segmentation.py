#!/usr/bin/env python3
"""Segmentation script for UAF-Plankline

Usage:
    ./segmentation.py -c <config.ini> -d <project directory>

License:
    MIT License

    Copyright (c) 2023 Thomas Kelly

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

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
        
        #con = sqlite3.connect(frame.get_name() + '/' + 'images.db')
        
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
                                    max_variation=0.5,
                                      min_diversity=0.1)
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
                    #im = sqlite3.Binary(gray[bboxes[i][1]:(bboxes[i][1] + bboxes[i][3]), bboxes[i][0]:(bboxes[i][0] + bboxes[i][2])])
                    im = Image.fromarray(gray[bboxes[i][1]:(bboxes[i][1] + bboxes[i][3]), bboxes[i][0]:(bboxes[i][0] + bboxes[i][2])])
                    im.save(f"{name}{n:05}-{i:05}.png")
                    stats = [name, n, i, bboxes[i][0] + bboxes[i][2]/2, bboxes[i][1] + bboxes[i][3]/2, bboxes[i][2], bboxes[i][3], area[i]]
                    outwritter.writerow(stats)
                    #con.execute(f'INSERT INTO frame(frame,crop,image) VALUES ({n}, {i}, ?) [{im}]')


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



if __name__ == "__main__":
    """
    The main entry point and script for segmentation.
    """

    v_string = "V2023.11.08"
    print(f"Starting Segmentation Script {v_string}")

    # create a parser for command line arguments
    parser = argparse.ArgumentParser(description="Segmentation tool for the plankton pipeline. Uses ffmpeg and seg_ff to segment a video into crops of plankton")
    parser.add_argument("-c", "--config", required = True, help = "Configuration ini file.")
    parser.add_argument("-d", "--directory", required = True, help = "Input directory containing ./raw/")

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    ## Determine directories
    raw_dir = os.path.abspath(args.directory) # /media/plankline/Data/raw/Camera1/Transect1
    segmentation_dir = raw_dir.replace("raw", "analysis") # /media/plankline/Data/analysis/Camera1/Transect1
    segmentation_dir = segmentation_dir.replace("camera0/", "camera0/segmentation/") # /media/plankline/Data/analysis/Camera1/Transect1
    segmentation_dir = segmentation_dir.replace("camera1/", "camera1/segmentation/") # /media/plankline/Data/analysis/Camera1/segmentation/Transect1
    segmentation_dir = segmentation_dir.replace("camera2/", "camera2/segmentation/") # /media/plankline/Data/analysis/Camera1/segmentation/Transect1
    segmentation_dir = segmentation_dir.replace("camera3/", "camera3/segmentation/") # /media/plankline/Data/analysis/Camera1/segmentation/Transect1
    
    segmentation_dir = segmentation_dir + f"-{config['segmentation']['basename']}" # /media/plankline/Data/analysis/segmentation/Camera1/segmentation/Transect1-REG
    os.makedirs(segmentation_dir, int(config['general']['dir_permissions']), exist_ok = True)

    avis = []
    avis = [os.path.join(raw_dir, avi) for avi in os.listdir(raw_dir) if avi.endswith(".avi")]

    if (len(avis) == 0):
        #logger.error(f"No avi files found in machine_scratch/raw make sure avi files are in {raw_dir}.")
        sys.exit(f"No avi files found in machine_scratch/raw make sure avi files are in {raw_dir}.")

    ## Prepare workers for receiving frames
    num_threads = os.cpu_count() - 1
    #num_threads = 2
    max_queue = num_threads * 4
    q = Queue(maxsize=int(max_queue))

    for i in range(num_threads):
        worker = Process(target=process_frame, args=(q, config,), daemon=True)
        worker.start()

    for av in tqdm.tqdm(avis):
        process_avi(av, segmentation_dir, config, q)

    print('Joining')
    worker.join(timeout=10)
