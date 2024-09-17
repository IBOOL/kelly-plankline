#!/usr/bin/env python3
"""Training script for UAF-Plankline
    This is the training script used to facilitate training of new SCNN models.
    Settings for this script come exclusively from the configuration ini file
    passed: 
        
        e.g. python3 train.py -c config.ini
    
    Importantly, teh script copies all data to a temporary scratch directory and 
    then copies results back onces completed. If there is a failure then no model
    epochs will be saved. The user is free to grab them from the scratch_dir.

Usage:
    ./train.py -c <config.ini>

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
import shutil
import argparse
import logging # TBK: logging module
import logging.config # TBK
import configparser # TBK: To read config file
import tqdm # TBK
from time import time
import psutil
from multiprocessing import Pool
import datetime
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
from PIL import Image
import pandas as pd


def pad(path, ratio=0.2):
    dirs = os.listdir(path)
    count = 0
    for item in dirs:
        if not os.path.isfile(path + item):
            dirs2 = os.listdir(path + item + "/")
            for item2 in dirs2:
                if os.path.isfile(path+item+"/"+item2):
                    im = Image.open(path+item+"/"+item2)
                    width, height = im.size
                    if width > height * (1. + ratio):
                        left = 0
                        right = width
                        top = (width - height)//2
                        bottom = width
                        result = Image.new(im.mode, (right, bottom), (255))
                        result.paste(im, (left, top))
                        im.close()
                        result.save(path+item+"/"+item2)
                        #print(f"({width}x{height}) -> ({right}x{bottom})")
                        count+=1
                    elif height > width * (1. + ratio):
                        left = (height - width)//2
                        right = height
                        top = 0
                        bottom = height
                        result = Image.new(im.mode, (right, bottom), (255))
                        result.paste(im, (left, top))
                        im.close()
                        result.save(path+item+"/"+item2,)
                        #print(f"({width}x{height}) -> ({right}x{bottom})")
                        count+=1
    count


def classify(model_file, input_dir):
    model = tf.keras.models.load_model(model_file)

    pad(input_dir)
    images = []
    image_files = []
    for img in os.listdir(input_dir):
        image_files.append(img)
        img = tf.keras.preprocessing.image.load_img(input_dir+img, target_size=(128, 128), color_mode='grayscale')
        #img = img.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)
    images = np.vstack(images)


    predictions = model.predict(images)
    prediction_labels = np.argmax(predictions, axis=-1)
    np.savetxt('prediction.csv', predictions, delimiter=',')

    with open('prediction_names.csv', newline='', mode='w') as csvfile:
        csvwritter = csv.writer(csvfile, delimiter='\n')
        csvwritter.writerow(image_files)



def init_model(num_classes, img_height, img_width):

    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
    model = ResNet18([img_height, img_width, 1], config['training']['model_name'], num_classes)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    model.summary()
    return(model)



def ResNet18(input_shape, name, num_classes):
    BN_AXIS = 3

    img_input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(-1. / 255, 1)(img_input)
    x = tf.keras.layers.RandomRotation(2)(x)
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)

    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = tf.keras.layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(axis=BN_AXIS, name='bn_conv1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = make_basic_block_layer(x, filter_num=64, blocks=2)
    x = make_basic_block_layer(x, filter_num=128, blocks=2, stride=2)
    x = make_basic_block_layer(x, filter_num=256, blocks=2, stride=2)
    x = make_basic_block_layer(x, filter_num=512, blocks=2, stride=2)


    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(img_input, x, name=name)
    return model


def make_basic_block_base(inputs, filter_num, stride=1):
    BN_AXIS = 3
    x = tf.keras.layers.Conv2D(filters=filter_num,
                                        kernel_size=(3, 3),
                                        strides=stride,
                                        kernel_initializer='he_normal',
                                        padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization(axis=BN_AXIS)(x)
    x = tf.keras.layers.Conv2D(filters=filter_num,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        kernel_initializer='he_normal',
                                        padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=BN_AXIS)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    shortcut = inputs
    if stride != 1:
        shortcut = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=stride,
                                            kernel_initializer='he_normal')(inputs)
        shortcut = tf.keras.layers.BatchNormalization(axis=BN_AXIS)(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x

def make_basic_block_layer(inputs, filter_num, blocks, stride=1):
    x = make_basic_block_base(inputs, filter_num, stride=stride)

    for _ in range(1, blocks):
        x = make_basic_block_base(x, filter_num, stride=1)

    return x



def load_model(config):
    #if int(config['training']['start']) > 0:
    #    return(tf.keras.models.load_model(config['training']['training_dir'], config))
    
    return(init_model(109, int(config['training']['image_size']), int(config['training']['image_size'])))
    

def train_model(model, config, train_ds, val_ds):
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=int(config['training']['stop'])-int(config['training']['start']),
                        initial_epoch=int(config['training']['start']),
                        batch_size = int(config['training']['batchsize']))
    
    
    return(model)


def init_ts(config):
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        config['training']['scnn_dir'] + '/data',
        interpolation='area',
        validation_split = 0.2,
        subset = "both",
        seed = 123,
        image_size = (int(config['training']['image_size']), int(config['training']['image_size'])),
        batch_size = int(config['training']['batchsize']),
        color_mode = 'grayscale')
    return(train_ds, val_ds)

if __name__ == "__main__":

    v_string = "V2023.10.09"
    print(f"Starting CNN Model Training Script {v_string}")
    
    # create a parser for command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config", required = True, help = "Configuration ini file.")
    args = parser.parse_args()

    if os.path.isfile(args.config) == False:
        print(f"No config file found called {args.config}. Aborting!")
        exit()

    config = configparser.ConfigParser()
    config.read(args.config)
    
    train_ds, val_ds = init_ts(config)
    model = load_model(config)
    model = train_model(model, config, train_ds, val_ds)
    model.save(config['training']['model_path'] + '/' + config['training']['model_name'])
    
    predictions = model.predict(val_ds)
    predictions = np.argmax(predictions, axis = -1)
    y = np.concatenate([y for x, y in val_ds], axis=0)

    confusion_matrix = tf.math.confusion_matrix(y, predictions)
    confusion_matrix = pd.DataFrame(confusion_matrix, index = train_ds.class_names, columns = train_ds.class_names)
    confusion_matrix.to_csv(config['training']['model_path'] + '/' + config['training']['model_name'] + ' confusion.csv')
    
    summary = {
        "file": val_ds.file_paths,
        "label": val_ds.class_names,
        "prediction": pd.Series(predictions),
    }
    
    summary = pd.DataFrame(summary)
    summary.to_csv(config['training']['model_path'] + '/' + config['training']['model_name'] + ' summary.csv')

