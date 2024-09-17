#!/usr/bin/env python3
"""Cleanup script for UAF-Plankline

Usage:
    ./cleanup.py -c <config.ini> -d <project directory>

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
import configparser # TBK: To read config file
import glob

if __name__ == "__main__":
    """Main entry point for cleanup.py"""
    v_string = "V2023.09s.05"

    # create a parser for command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--directory", required = False, help = "Input directory containing ./raw/")
    parser.add_argument("-f", action = 'store_true')

    # read in the arguments
    args = parser.parse_args()

    ## Read in config options:
    if args.directory is None:
        skip_working = True
    else:
        skip_working = False
    
    if not skip_working:
        working_dir = os.path.abspath(args.directory)
    
    seg_tmp = glob.glob('/tmp/segment*')
    class_tmp = glob.glob('/tmp/class*')
    train_tmp = glob.glob('/tmp/train*')

    if args.f is True:
        if not skip_working:
            if os.path.isdir(working_dir + '/segmentation/'):
                shutil.rmtree(working_dir + '/segmentation/', ignore_errors = True)

            if os.path.isdir(working_dir + '/classification/'):
                shutil.rmtree(working_dir + '/classification/', ignore_errors = True)

            if os.path.isdir(working_dir + '/R/'):
                shutil.rmtree(working_dir + '/R/', ignore_errors = True)

        if len(seg_tmp) > 0:
            for d in seg_tmp:
                shutil.rmtree(d, ignore_errors = True)
        
        if len(class_tmp) > 0:
            for d in class_tmp:
                shutil.rmtree(d, ignore_errors = True)
                
        if len(train_tmp) > 0:
            for d in train_tmp:
                shutil.rmtree(d, ignore_errors = True)
                
    else:
        def get_size(start_path = '.'):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(start_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    # skip if it is symbolic link
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)

            return total_size

        print("No -f flag provided. Rerun with -f flag if you want to delete the following directories:")
        
        if not skip_working:
            if os.path.isdir(working_dir + '/segmentation/'):
                print(f"Directory: {working_dir + '/segmentation/':30} containing {(get_size(working_dir + '/segmentation/')/10**9):.1f} GB")
            else:
                print(f"Directory: {working_dir + '/segmentation/':30} not found.")
            
            if os.path.isdir(working_dir + '/classification/'):
                print(f"Directory: {working_dir + '/classification/':30} containing {(get_size(working_dir + '/classification/')/10**9):.1f} GB")
            else:
                print(f"Directory: {working_dir + '/classification/':30} not found.")
            
            if os.path.isdir(working_dir + '/R/'):
                print(f"Directory: {working_dir + '/R/':30} containing {(get_size(working_dir + '/R/')/10**9):.1f} GB")
            else:
                print(f"Directory: {working_dir + '/R/':30} not found.")
        
        if len(seg_tmp) > 0:
            for d in seg_tmp:
                print(f"Directory: {d:30} containing {(get_size(d)/10**9):.1f} GB")
        
        if len(class_tmp) > 0:
            for d in class_tmp:
                print(f"Directory: {d:30} containing {(get_size(d)/10**9):.1f} GB")
                
        if len(train_tmp) > 0:
            for d in train_tmp:
                print(f"Directory: {d:30} containing {(get_size(d)/10**9):.1f} GB")

    print('Done.')