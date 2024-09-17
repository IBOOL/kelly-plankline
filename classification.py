#!/usr/bin/env python3
"""Classification script for UAF-Plankline

Usage:
    ./classification.py -c <config.ini> -d <project directory>

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

import os
import sys
import shutil
import argparse
import glob
import logging
import configparser
import logging.config
import tqdm
import subprocess
import datetime
from time import time
from multiprocessing import Pool, Queue
import psutil
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import csv
from PIL import Image
import os
import pandas as pd


def classify(model, input_dir, output_dir):
    images = []
    image_files = []
    for img in os.listdir(input_dir):
        if os.path.splitext(img)[1] == '.png':
            image_files.append(img)
            img = tf.keras.preprocessing.image.load_img(input_dir + '/' + img, target_size=(int(config['classification']['image_size']),int(config['classification']['image_size'])), color_mode='grayscale')
            img = np.expand_dims(img, axis=0)
            images.append(img)
    images = np.vstack(images)
    
    predictions = model.predict(images)
    prediction_labels = np.argmax(predictions, axis=-1)
    df = pd.DataFrame(predictions, index=image_files)
    df.to_csv(output_dir + '_' + 'prediction.csv', index=True, header=True, sep=',')
    

def pad(path, ratio=0.2):
    dirs = os.listdir(path)
    count = 0
    for item in dirs:
        if not os.path.isfile(path + '/' + item):
            dirs2 = os.listdir(path + '/' + item + "/")
            for item2 in dirs2:
                if os.path.isfile(path+'/'+item+"/"+item2):
                    if os.path.splitext(path+'/'+item+"/"+item2)[1] == '.png':
                        im = Image.open(path+'/'+item+"/"+item2)
                        width, height = im.size
                        if width > height * (1. + ratio):
                            left = 0
                            right = width
                            top = (width - height)//2
                            bottom = width
                            result = Image.new(im.mode, (right, bottom), (255))
                            result.paste(im, (left, top))
                            im.close()
                            result.save(path + '/'+item+"/"+item2)
                            count+=1
                        elif height > width * (1. + ratio):
                            left = (height - width)//2
                            right = height
                            top = 0
                            bottom = height
                            result = Image.new(im.mode, (right, bottom), (255))
                            result.paste(im, (left, top))
                            im.close()
                            result.save(path + '/'+item+"/"+item2)
                            count+=1
    count


if __name__ == "__main__":
    """Main entry point for classification.py"""
    
    v_string = "V2023.11.13"
    session_id = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")).replace(':', '')
    print(f"Starting Plankline Classification Script {v_string}")

    # create a parser for command line arguments
    parser = argparse.ArgumentParser(description="Classification tool for managing the isiis_scnn processes")
    parser.add_argument("-c", "--config", required = True, help = "Configuration ini file.")
    parser.add_argument("-d", "--directory", required = True)

    # read in the arguments
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    
    # Load model
    model_path = os.path.join(config['classification']['model_dir'], config['classification']['model_name'])
    
    if not os.path.isdir(model_path):
        sys.exit('Error, no model found at {model_path}')
    
    model = tf.keras.models.load_model(model_path)

    segmentation_dir = os.path.abspath(args.directory)  # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg
    classification_dir = segmentation_dir.replace('segmentation', 'classification')  # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg
    classification_dir = classification_dir + '-' + config["classification"]["model_name"] # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg-Plankton
    fast_scratch = config['classification']['fast_scratch'] + "/classify-" + session_id
    
    os.makedirs(classification_dir, int(config['general']['dir_permissions']), exist_ok = True)
    os.makedirs(fast_scratch, int(config['general']['dir_permissions']), exist_ok = True)
    
    root = os.listdir(segmentation_dir)
    
    if len(root) == 0:
        sys.exit(f"Error: No tars file in segmenation directory: {segmentation_dir}")
    
    n_pad = pad(segmentation_dir)
    print(f"Number of images padded: {n_pad}.")
    
    for r in root:
        classify(model, segmentation_dir + '/' + r, classification_dir + '/' + r)



