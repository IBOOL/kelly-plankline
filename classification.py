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
import json

if __name__ == "__main__":
    directory = '../../analysis/camera0/segmentation/shadowgraph-REG'

    config = {
        'general' : {
            'dir_permissions' : 511
        },
        'classification' : {
            'model_name' : 'theta',
            'model_dir' : '../../model',
            'scnn_instances' : 1,
            'fast_scratch' : '/tmp',
            'batchsize' : 128,
            'image_size' : 128
        }
    }

    v_string = "V2024.05.20"
    session_id = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")).replace(':', '')
    print(f"Starting Plankline Classification Script {v_string}")

    # Load model
    model_path = f"../../model/{config['classification']['model_name']}.keras"
    label_path = f"../../model/{config['classification']['model_name']}.json"
    model = tf.keras.models.load_model(model_path)
    
    with open(label_path, 'r') as file:
        sidecar = json.load(file)

    print(f"Loaded keras model {config['classification']['model_name']} and sidecar JSON file.")
    
    # ### Setup Folders and run classification on each segment output
    segmentation_dir = os.path.abspath(directory)  # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg
    classification_dir = segmentation_dir.replace('segmentation', 'classification')  # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg
    classification_dir = classification_dir + '-' + config["classification"]["model_name"] # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg-Plankton
    fast_scratch = config['classification']['fast_scratch'] + "/classify-" + session_id
        
    os.makedirs(classification_dir, int(config['general']['dir_permissions']), exist_ok = True)
    os.makedirs(fast_scratch, int(config['general']['dir_permissions']), exist_ok = True)

    root = [z for z in os.listdir(segmentation_dir) if z.endswith('zip')]
    

    print(f"Found {len(root)} archives for potential processing.")

    for r in tqdm.tqdm(root):
        r2 = r.replace(".zip", "")
        shutil.unpack_archive(segmentation_dir + '/' + r, segmentation_dir + '/' + r2 + "/", 'zip')

        images = []
        image_files = []
        for img in os.listdir(segmentation_dir + '/' + r2):
            if img.endswith(('png', 'jpeg', 'jpg', 'tif', 'tiff')): 
                image_files.append(img)
                img = tf.keras.preprocessing.image.load_img(segmentation_dir + '/' + r2 + '/' + img,
                                                            target_size=(int(config['classification']['image_size']), int(config['classification']['image_size'])),
                                                            color_mode='grayscale')
                img = np.expand_dims(img, axis=0)
                images.append(img)
        images = np.vstack(images)
            
        predictions = model.predict(images, verbose = 0)
        prediction_labels = np.argmax(predictions, axis=-1)
        prediction_labels = [labels[i] for i in prediction_labels]
        df = pd.DataFrame(predictions, index=image_files)
        df_short = pd.DataFrame(prediction_labels, index=image_files)
            
        df.columns = sidecar['labels']
        df.to_csv(classification_dir + '/' + r2 + '_' + 'prediction.csv', index=True, header=True, sep=',')
        df_short.to_csv(classification_dir + '/' + r2 + '_' + 'predictionlist.csv', index=True, header=True, sep=',')
        shutil.rmtree(segmentation_dir + '/' + r2 + "/", ignore_errors=True)
