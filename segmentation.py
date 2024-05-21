import sys
#import argparse
import configparser
from PIL import Image
import cv2
import os
import numpy as np
import csv
from multiprocessing import Process, Queue
import tqdm
import shutil
import random

class Frame:
    def __init__(self, fpath, name, frame, n):
        self.fpath = fpath  # default is 0 for primary camera
        self.name = name
        self.frame = frame
        self.n = n
        self.flatfield = False

    def read(self):
        return self.frame

    def get_n(self):
        return self.n

    def get_name(self):
        return self.name

    def update(self, newframe):
        self.frame = newframe

    def set_flatfield(self, state):
        self.flatfield = state

    def get_flatfield(self):
        return self.flatfield



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
        
        ## Read img and flatfield
        gray = cv2.cvtColor(frame.read(), cv2.COLOR_BGR2GRAY)
        gray = np.array(gray)
        if not frame.get_flatfield():
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

        if config['segmentation']['diagnostic']:
            cv2.imwrite(f'{name}{n:06}-qualtilefield.jpg', gray)
            cv2.imwrite(f'{name}{n:06}-threshold.jpg', thresh)

        with open(f'{name[:-1]} statistics.csv', 'a', newline='\n') as outcsv:
            outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
            for i in range(len(cnts)):
                x,y,w,h = cv2.boundingRect(cnts[i])
                if 2*w + 2*h > int(config['segmentation']['min_perimeter']) / 2: # Save information if perimenter is greater than half the minimum
                    stats = [n, i, x + w/2, y + h/2, w, h]
                    outwritter.writerow(stats)
                
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
                    im_padded.save(f"{name}{n:06}-{i:06}.png")
                

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
        outwritter.writerow(['frame', 'crop', 'x', 'y', 'w', 'h'])

    n = 1
    while True:
        ret, frame = video.read()
        if ret:
            q.put(Frame(avi_path, output_path, frame, n), block = True)
            n += 1
        else:
            break


def process_image_dir(img_path, segmentation_dir, config, q):
    """
    This function will take an image folder as input and perform the following steps:
    1. Create output file structures/directories
    2. Load each frame, pass it through flatfielding and sequentially save segmented targets
    """

    # segmentation_dir: /media/plankline/Data/analysis/segmentation/Camera1/segmentation/Transect1-REG
    _, filename = os.path.split(img_path)
    output_path = segmentation_dir + os.path.sep + filename + os.path.sep
    os.makedirs(output_path, exist_ok=True)

    with open(f'{output_path[:-1]} statistics.csv', 'a', newline='\n') as outcsv:
              outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
              outwritter.writerow(['frame', 'crop', 'x', 'y', 'w', 'h'])

    bkg = np.array(cv2.imread(config['segmentation']['calibration_image']))
        
    m = np.amin(bkg)
    M = np.amax(bkg)
    for f in os.listdir(img_path):
      if f.endswith(('.jpg', '.jpeg', '.png')):
          image = np.array(cv2.imread(img_path + os.path.sep + f))
          image = image / bkg * 255
          image = image.clip(0,255).astype(np.uint8)
          
          if config['segmentation']['diagnostic']:
              cv2.imwrite(output_path + os.path.sep + f'{f}-flatfield.jpg', image)
          
          ## Apply Mask (also invert so black masking becomes white background in shadowgraph)
          image = ~image
          mask = np.zeros(image.shape[:2], dtype="uint8")
          cv2.circle(mask, (image.shape[1]//2, image.shape[0]//2), 1100, 255, -1)
          image = cv2.bitwise_and(image, image, mask = mask) # Mask
          
          if config['segmentation']['diagnostic']:
              cv2.imwrite(output_path + os.path.sep + f'{f}-flatfield+crop.jpg', ~image)
          
          image = ~image # Invert back to shadowgraph-standard
          newframe = Frame(f, output_path, image, f)
          newframe.set_flatfield(True)
          q.put(newframe, block = True)


def generate_median_image(directory, output_dir):
    """
    
    """
    # Get a list of all image file names in the directory
    image_files = [file for file in os.listdir(directory) if file.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No image files found in the directory.")
        return

    image_files = random.sample(image_files, min([100, len(image_files)]))
    
    # Read the first image to get the dimensions
    first_image_path = os.path.join(directory, image_files[0])
    first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    height, width = first_image.shape

    # Initialize an array to store all images
    all_images = np.zeros((len(image_files), height, width), dtype=np.uint8)

    # Load all images into the array
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(directory, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #image = cv2.GaussianBlur(image, (5,5), 0) # TODO: Gaussian blur help at all?
        all_images[idx] = image

    # Compute the median image
    median_image = np.median(all_images, axis=0).astype(np.uint8)
    cv2.imwrite(output_dir + os.path.sep + 'median_image.jpg', median_image)
    print("New median (calibration) image saved as 'median_image.jpg'.")
    

if __name__ == "__main__":
    
    directory = '../../raw/camera0/shadowgraph'

    config = {
        'general' : {
            'dir_permissions' : 511
        },
        'segmentation' : {
            'diagnostic' : False,
            'basename' : 'REG',
            'min_perimeter' : 4*30,
            'flatfield_q' : 0.05,
            'calibration_image' : '../../raw/camera0/shadowgraph/median_image.jpg'
        }
    }

    v_string = "V2024.05.18"
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

    ## Find files to process:
    # AVI videos
    avis = []
    avis = [os.path.join(raw_dir, avi) for avi in os.listdir(raw_dir) if avi.endswith(".avi")]
    print(f"Number of AVIs found: {len(avis)}")

    # Subfolders of images(?)
    imgsets = []
    imgsets = [os.path.join(raw_dir, sub) for sub in next(os.walk(raw_dir))[1]]
    print(f"Number of possible image sets found: {len(imgsets)}")

    ## Prepare workers for receiving frames
    num_threads = os.cpu_count() - 1
    max_queue = num_threads * 4 # Prepare 4 frames per thread. TODO: test memory vs performance considerations. UPDATE: 4 still seems okay on my laptop.
    q = Queue(maxsize=int(max_queue))

    for i in range(num_threads):
        worker = Process(target=process_frame, args=(q, config,), daemon=True)
        worker.start()
        
    print(f'Spun up {num_threads} worker threads for processing.')

    if (len(avis)) > 0:
        print(f'Starting processing on {len(avis)} AVI files.')
        for av in tqdm.tqdm(avis):
            process_avi(av, segmentation_dir, config, q)

        for av in avis:
            _, filename = os.path.split(av)
            output_path = segmentation_dir + os.path.sep + filename + os.path.sep
            shutil.make_archive(segmentation_dir + os.path.sep + filename, 'zip', output_path)
            if not config['segmentation']['diagnostic']:
                shutil.rmtree(output_path, ignore_errors=True)

    if len(imgsets) > 0:
        print(f'Starting processing on {len(imgsets)} subfolders.')
        for im in tqdm.tqdm(imgsets):
            if not os.path.exists(config['segmentation']['calibration_image']):
                generate_median_image(im, config['segmentation']['calibration_image'])
            process_image_dir(im, segmentation_dir, config, q)
            
            _, filename = os.path.split(im)
            output_path = segmentation_dir + os.path.sep + filename + os.path.sep
            shutil.make_archive(segmentation_dir + os.path.sep + filename, 'zip', output_path)
            if not config['segmentation']['diagnostic']:
                shutil.rmtree(output_path, ignore_errors=True)
            
    print('Joining')
    worker.join(timeout=10)


