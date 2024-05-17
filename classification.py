# %%
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


directory = '../../analysis/camera1/segmentation/GAK_202207-REG/'

config = {
    'general' : {
        'dir_permissions' : 511
    },
    'segmentation' : {
        'basename' : 'REG',
        'segment_processes' : 1,
        'overlap' : 0.1,
        'max_area' : 400000,
        'min_area' : 200,
        'delta' : 4,
        'flatfield_q' : 0.02
    },
    'classification' : {
        'model_name' : 'Gamma',
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

v_string = "V2023.11.13"
session_id = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")).replace(':', '')
print(f"Starting Plankline Classification Script {v_string}")

# Load model
model_path = f"../../model/{config['classification']['model_name']}/"
model = tf.keras.models.load_model(model_path)

# ### Setup Folders and run classification on each segment output
segmentation_dir = os.path.abspath(directory)  # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg
classification_dir = segmentation_dir.replace('segmentation', 'classification')  # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg
classification_dir = classification_dir + '-' + config["classification"]["model_name"] # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg-Plankton
fast_scratch = config['classification']['fast_scratch'] + "/classify-" + session_id
    
os.makedirs(classification_dir, int(config['general']['dir_permissions']), exist_ok = True)
os.makedirs(fast_scratch, int(config['general']['dir_permissions']), exist_ok = True)

root = os.listdir(segmentation_dir)

print(f"Found {len(root)} subfolders.")

for r in tqdm.tqdm(root):
    images = []
    image_files = []
    for img in os.listdir(segmentation_dir + '/' + r):
        if os.path.splitext(img)[1] == '.png':
            image_files.append(img)
            img = tf.keras.preprocessing.image.load_img(segmentation_dir + '/' + r + '/' + img,
                                                        target_size=(int(config['classification']['image_size']),int(config['classification']['image_size'])),
                                                        color_mode='grayscale')
            img = np.expand_dims(img, axis=0)
            images.append(img)
    images = np.vstack(images)
    
    predictions = model.predict(images, verbose = 0)
    prediction_labels = np.argmax(predictions, axis=-1)
    df = pd.DataFrame(predictions, index=image_files)
    df.to_csv(classification_dir + '/' + r + '_' + 'prediction.csv', index=True, header=True, sep=',')


