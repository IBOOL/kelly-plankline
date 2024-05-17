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
import json

def classify(model_file, input_dir):
    model = tf.keras.models.load_model(model_file)

    pad(input_dir)
    images = []
    image_files = []
    for img in os.listdir(input_dir):
        image_files.append(img)
        img = tf.keras.preprocessing.image.load_img(input_dir + img,
         target_size=(int(config['training']['image_size']),
         int(config['training']['image_size'])), color_mode='grayscale')
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


def load_model(config):
    if int(config['training']['start']) > 0:
        return(tf.keras.models.load_model(config['training']['scnn_dir'], config))
    
    return(init_model(61, int(config['training']['image_size']), int(config['training']['image_size'])))
   


def conv_block(x, growth_rate):
    x1 = tf.keras.layers.BatchNormalization()(x)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Conv2D(growth_rate, (3, 3), padding='same')(x1)
    x = tf.keras.layers.concatenate([x, x1], axis=-1)
    return x

def dense_block(x, num_layers, growth_rate):
    for i in range(num_layers):
        x = conv_block(x, growth_rate)
    return x

def transition_block(x, compression):
    num_filters = int(x.shape[-1] * compression)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, (1, 1))(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    return x

def DenseNet(input_shape, num_classes):

    ## Init and Augmentation
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(-1. / 255, 1)(inputs)
    x = tf.keras.layers.RandomRotation(1, fill_mode='constant', fill_value=0.0)(x)
    x = tf.keras.layers.RandomZoom(0.2, fill_value=0.0, fill_mode='constant')(x)
    x = tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=0.0)(x)
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    
    # Initial convolution layer
    x = tf.keras.layers.Conv2D(128, (7, 7), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Dense blocks
    x = dense_block(x, num_layers=4, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=8, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=16, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=32, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=16, growth_rate=32)
    
    # Final layers
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model
    

def init_model(num_classes, img_height, img_width):

    model = DenseNet([img_height, img_width, 1], num_classes)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    model.summary()
    return(model)


def train_model(model, config, train_ds, val_ds):
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=int(config['training']['stop'])-int(config['training']['start']),
                        initial_epoch=int(config['training']['start']),
                        batch_size = int(config['training']['batchsize']))
    
    
    return(model)


def init_ts(config):
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        config['training']['scnn_dir'],
        interpolation='area',
        validation_split = config['training']['validationSetRatio'],
        subset = "both",
        seed = int(config['training']['seed']),
        image_size = (int(config['training']['image_size']), int(config['training']['image_size'])),
        batch_size = int(config['training']['batchsize']),
        color_mode = 'grayscale')
    return(train_ds, val_ds)

if __name__ == "__main__":
    config = {
        'general' : {
            'dir_permissions' : 511,
            'dry_run' : 'False'
        },
        'training' : {
            'scnn_dir' : '../../training/training_set_20240123',
            'model_name': 'theta',
            'model_path': '../../model/',
            'image_size': 128,
            'start' : 0,
            'stop' : 100,
            'validationSetRatio' : 0.2,
            'batchsize' : 128,
            'seed': 123
        }
    }

    v_string = "V2024.05.03"
    print(f"Starting CNN Model Training Script {v_string}")

    # ## Load training and validation data sets

    train_ds, val_ds = init_ts(config)
    model = load_model(config)
    model = train_model(model, config, train_ds, val_ds)
    model.save(config['training']['model_path'] + '/' + config['training']['model_name'] + '.keras')
        
    predictions = model.predict(val_ds)
    predictions = np.argmax(predictions, axis = -1)
    y = np.concatenate([y for x, y in val_ds], axis = 0)

    confusion_matrix = tf.math.confusion_matrix(y, predictions)
    confusion_matrix = pd.DataFrame(confusion_matrix, index = train_ds.class_names, columns = train_ds.class_names)
    #confusion_matrix.to_excel(config['training']['model_path'] + '/' + config['training']['model_name'] + ' confusion.xlsxthe s')
    confusion_matrix.to_csv(config['training']['model_path'] + '/' + config['training']['model_name'] + ' confusion.csv')
    
    json_object = json.dumps(train_ds.class_names, indent=4)
 
    # Writing to sample.json
    with open(config['training']['model_path'] + '/' + config['training']['model_name'] + ' classes.json', "w") as outfile:
        outfile.write(json_object)
        
    #summary = {
    #    "file": val_ds.file_paths,
    #    "label": val_ds.class_names,
    #    "prediction": pd.Series(predictions),
    #}
        
    #summary = pd.DataFrame(summary)
    #summary.to_csv(config['training']['model_path'] + '/' + config['training']['model_name'] + ' summary.csv')


