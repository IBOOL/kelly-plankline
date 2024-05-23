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
import platform
import time


class TrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, filename):
        super(TrainingLogger, self).__init__()
        self.filename = filename
        self.file = open(self.filename, 'w')
        self.csv_writer = None

    def on_train_begin(self, logs=None):
        self.csv_writer = csv.writer(self.file)
        self.csv_writer.writerow(['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: loss={logs['loss']}, accuracy={logs['accuracy']}, val_loss={logs['val_loss']}, val_accuracy={logs['val_accuracy']}")
        self.csv_writer.writerow([epoch+1, logs['loss'], logs['accuracy'], logs['val_loss'], logs['val_accuracy']])

    def on_train_end(self, logs=None):
        self.file.close()


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


def DenseNet121(input_shape, num_classes):

    ## Init and Augmentation
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(-1. / 255, 1)(inputs) # Invert shadowgraph image (white vs black)
    x = tf.keras.layers.RandomRotation(1, fill_mode='constant', fill_value=0.0)(x)
    x = tf.keras.layers.RandomZoom(0.2, fill_value=0.0, fill_mode='constant')(x)
    x = tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode='constant', fill_value=0.0)(x)
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    x = tf.keras.layers.RandomBrightness(0.2, value_range=(0.0, 1.0))(x)
    
    # Initial convolution layer
    x = tf.keras.layers.Conv2D(128, (7, 7), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    ## DenseNet121
    x = dense_block(x, num_layers=6, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=12, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=24, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=16, growth_rate=32)

    # Final layers
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model


def DenseNet169(input_shape, num_classes):

    ## Init and Augmentation
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(-1. / 255, 1)(inputs) # Invert shadowgraph image (white vs black)
    x = tf.keras.layers.RandomRotation(1, fill_mode='constant', fill_value=0.0)(x)
    x = tf.keras.layers.RandomZoom(0.2, fill_value=0.0, fill_mode='constant')(x)
    x = tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode='constant', fill_value=0.0)(x)
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    x = tf.keras.layers.RandomBrightness(0.2, value_range=(0.0, 1.0))(x)
    
    # Initial convolution layer
    x = tf.keras.layers.Conv2D(128, (7, 7), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    ## DenseNet169
    x = dense_block(x, num_layers=6, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=12, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=32, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=32, growth_rate=32)

    # Final layers
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model


def DenseNet201(input_shape, num_classes):

    ## Init and Augmentation
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(-1. / 255, 1)(inputs) # Invert shadowgraph image (white vs black)
    x = tf.keras.layers.RandomRotation(1, fill_mode='constant', fill_value=0.0)(x)
    x = tf.keras.layers.RandomZoom(0.2, fill_value=0.0, fill_mode='constant')(x)
    x = tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode='constant', fill_value=0.0)(x)
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    x = tf.keras.layers.RandomBrightness(0.2, value_range=(0.0, 1.0))(x)
    
    # Initial convolution layer
    x = tf.keras.layers.Conv2D(128, (7, 7), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    ## DenseNet201
    x = dense_block(x, num_layers=6, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=12, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=48, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=32, growth_rate=32)

    # Final layers
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model


def DenseNet264(input_shape, num_classes):

    ## Init and Augmentation
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(-1. / 255, 1)(inputs) # Invert shadowgraph image (white vs black)
    x = tf.keras.layers.RandomRotation(1, fill_mode='constant', fill_value=0.0)(x)
    x = tf.keras.layers.RandomZoom(0.2, fill_value=0.0, fill_mode='constant')(x)
    x = tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode='constant', fill_value=0.0)(x)
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    x = tf.keras.layers.RandomBrightness(0.2, value_range=(0.0, 1.0))(x)
    
    # Initial convolution layer
    x = tf.keras.layers.Conv2D(128, (7, 7), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    ## DenseNet264
    x = dense_block(x, num_layers=6, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=12, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=64, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=48, growth_rate=32)

    # Final layers
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model

def init_model(num_classes, img_height, img_width):

    ## Generate new model:
    model = DenseNet121([img_height, img_width, 1], num_classes)
    #model = DenseNet169([img_height, img_width, 1], num_classes)
    #model = DenseNet102([img_height, img_width, 1], num_classes)
    #model = DenseNet264([img_height, img_width, 1], num_classes)
    
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    model.summary()
    return(model)


def train_model(model, config, train_ds, val_ds):
    custom_logger = TrainingLogger(config['training']['model_path'] + '/' + config['training']['model_name'] + '.log')

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=int(config['training']['stop'])-int(config['training']['start']),
                        initial_epoch=int(config['training']['start']),
                        batch_size = int(config['training']['batchsize']),
                        callbacks=[custom_logger])
    
    return(model, history)


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
    with open('config.json', 'r') as f:
        config = json.load(f)

    v_string = "V2024.05.03"
    session_id = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")).replace(':', '')
    
    print(f"Starting CNN Model Training Script {v_string}")
    timer = {'init' : time.time()}

    # ## Load training and validation data sets
    train_ds, val_ds = init_ts(config)
    timer['model_load_start'] = time.time()
    model = load_model(config)
    timer['model_load_end'] = time.time()
    
    timer['model_train_start'] = time.time()
    model, history = train_model(model, config, train_ds, val_ds)
    timer['model_train_end'] = time.time()
    
    timer['model_save_start'] = time.time()
    model.save(config['training']['model_path'] + '/' + config['training']['model_name'] + '.keras')
    timer['model_save_end'] = time.time()
        
    predictions = model.predict(val_ds)
    predictions = np.argmax(predictions, axis = -1)
    y = np.concatenate([y for x, y in val_ds], axis = 0)

    confusion_matrix = tf.math.confusion_matrix(y, predictions)
    confusion_matrix = pd.DataFrame(confusion_matrix, index = train_ds.class_names, columns = train_ds.class_names)
    confusion_matrix.to_csv(config['training']['model_path'] + '/' + config['training']['model_name'] + ' confusion.csv')
    
    timer['close'] = time.time()
    
    ## Generate sidecar dictionary:
    sidecar = {
        'model_name' : config['training']['model_name'],
        'model_type' : config['training']['model_type'],
        'labels' : train_ds.class_names,
        'script_version' : v_string,
        'config' : config,
        'system_info' : {
            'System' : platform.system(),
            'Node' : platform.node(),
            'Release' : platform.release(),
            'Version' : platform.version(),
            'Machine' : platform.machine(),
            'Processor' : platform.processor()
        },
        'timings' : timer,
        'history' : history
    }
    
    json_object = json.dumps(sidecar, indent=4)
    with open(config['training']['model_path'] + '/' + config['training']['model_name'] + '.json', "w") as outfile:
        outfile.write(json_object)

