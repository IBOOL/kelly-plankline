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
import sys
import shutil
import argparse
import logging # TBK: logging module
import logging.config # TBK
import configparser # TBK: To read config file
import tqdm # TBK
import subprocess
from time import time
import psutil
from multiprocessing import Pool
import datetime



if __name__ == "__main__":

    v_string = "V2023.09.13"
    print(f"Starting Plankline Training Script {v_string}")
    
    # create a parser for command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config", required = True, help = "Configuration ini file.")

    # read in the arguments
    args = parser.parse_args()

    if os.path.isfile(args.config) == False:
        print(f"No config file found called {args.config}. Aborting!")
        exit()

    config = configparser.ConfigParser()
    if os.path.exists('default.ini'):
        config.read('default.ini')
    config.read(args.config)

    if config.has_option('logging', 'config') == False:
        print(f"No logging:config specified in {args.config}. Aborting!")
        exit()


    # Variables read in from the config file:
    config_version = config['general']['config_version']
    model_dir = config['training']['model_dir']
    scnn_cmd = config['training']['scnn_cmd']
    start = config['training']['start']
    stop = config['training']['stop']
    batchsize = config['training']['batchsize']
    basename = config['training']['basename']
    vsp = config['training']['validationSetRatio']
    lrd = config['training']['learningRateDecay']
    ilr = config['training']['initialLearningRate']
    permis = int(config['general']['dir_permissions'])
    
    ## Setup logger
    logging.config.fileConfig(config['logging']['config'], defaults={'date':datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),'path':model_dir,'name':'train'})
    logger = logging.getLogger('sLogger')

    logger.info(f"Starting training script {v_string}")


    # Print config options to screen (TBK)
    print(f"Configuration file: {args.config} (Version {config_version})")
    print(f"Performing training for {model_dir}")
    print(f"Starting Epoch:        {start}")
    print(f"Stopping Epoch:        {stop}")
    print(f"Initial Learning Rate: {ilr}")
    print(f"Learning Rate Decay:   {lrd}")
    print(f"Validation Ratio:      {vsp}")
    
    cp_file = model_dir + '/' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + ' ' + args.config
    logger.debug(f"Copying ini file to training directory {model_dir}")
    logger.info(f"Copy config to {cp_file}")
    shutil.copy2(args.config, cp_file)

    logger.info("Setting up directories.")
    
    timer_train = time()
    for i in list(range(int(start), int(stop)+1)):
        logger.info(f"Current ram usage (GB): {psutil.virtual_memory()[3]/1000000000:.2f}")
        logger.info(f"Current cpu usage (%): {psutil.cpu_percent(4):.1f}")
    
        ## Format training call:
        train = f'\"{scnn_cmd}\" -project {model_dir} -start {i} -stop {i+1} -batchSize {batchsize} -basename {basename} -vsp {vsp} -lrd {lrd} -ilr {ilr} -cD 0'
        train_call = [scnn_cmd, '-project', model_dir, '-start', str(i), '-stop', str(i+1), '-batchSize', batchsize, '-basename', basename, '-vsp', vsp, '-lrd', lrd, '-ilr', ilr, '-cD', '0']
        logger.info("Training call: " + train)
        print(f"Starting training epoch {i}.")

        result = subprocess.run(train_call, stdout = subprocess.PIPE)
        result = result.stdout.decode('utf-8')
        with open(model_dir + '/weights/' + basename + '_epoch-' + str(i) + '.log', 'a') as f:
            f.write(result)

        result = result.split('\n')

        if len(result) > 75:
            print(*result[76:(len(result)-1)], sep = '\n')
            logger.debug(result[76:(len(result)-1)])
        else:
            logger.debug(result)
            print(result)
    
    timer_train = time() - timer_train

    logger.debug(f"Training finished in {timer_train:.3f} s.")
    print(f"Finished training in {timer_train:.1f} seconds.")

    logger.debug("Done.")
    
