#!/usr/bin/env python3
"""Image Pulling script for UAF-Plankline

Usage:
    ./pull_all.py -d <project directory>

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
import glob
import argparse
import csv # to parse all of the csv files
import tarfile # untar the images
import re
import shutil # for copying files

# arg types
def directory(arg):
    if os.path.isdir(arg):
        return arg
    else:
        raise argparse.ArgumentTypeError("Not a valid directory path")

def csv_type(arg):
    if not os.path.isfile(arg):
        raise argparse.ArgumentTypeError("Not a file")
    if not arg.endswith(".csv"):
        raise argparse.ArgumentTypeError("Not a csv file")
    else:
         return arg
def float01(arg):
    if not float(arg):
        raise argparse.ArgumentTypeError("Not a floating point number")
    arg = float(arg)
    if not (arg < 1.0 and arg > 0.0):
        raise argparse.ArgumentTypeError("Not a floating point value between 0 and 1")
    else:
        return arg

# check to make sure the image doesn't already exist. Some images in different tars are given the same name
def add_unique_postfix(fn):
    if not os.path.exists(fn): # if the file name doesn't exist, don't change the name
        return fn

    path, name = os.path.split(fn)
    name, ext = os.path.splitext(name)

    make_fn = lambda i: os.path.join(path, '%s(%d)%s' % (name, i, ext))

    for i in range(2, 10000): # finds a number that is not used to put as the suffix
        uni_fn = make_fn(i)
        if not os.path.exists(uni_fn):
            return uni_fn

    return None
def get_parser(): # get command line options
    parser = argparse.ArgumentParser(description="Tool that was created to help expand the data for the taxon that we do not have many images of by utilizing the data from the images that were already classified.")
    # you should do many taxon at once since a lot of the time is lost unzipping files so it is only slightly less efficient
    parser.add_argument("-t", "--taxon", required=False, type=str, nargs='+', help="The taxons that you are looking to find from the csv files. This should be given as a list of taxon separated by a space.")

    selectgroup = parser.add_mutually_exclusive_group(required=True)
    selectgroup.add_argument("-p", "--probability_taxon", type=float01, help="The probability that a taxon needs to be over for the program to extract image extract the image, this should be a value between (0,1)")
    selectgroup.add_argument("-b", "--best_taxon", action="store_true", help="Extracts the images if the image was classified as a certain taxa, that is one of the taxons that you are trying to find.")

    inputgroup = parser.add_mutually_exclusive_group(required=True)
    inputgroup.add_argument("-d", "--input_drive", type=directory, help="The directory that you want to be pulling csv files from to look for images")
    inputgroup.add_argument("-c", "--input_csv", type=csv_type, help="The csv file that you want to look for taxon images.")

    parser.add_argument("-o", "--output", required=True, type=directory, help="The directory that you want to put the taxon folders of images")
    parser.add_argument("-r", "--raw_tars", required=True, type=directory, help="The directory of all of the tar images that correspond to the csv data")
    parser.add_argument("-m", "--min_images", type=int, default=0, help="The minimum number of images you need to have in a tar.gz in order to unzip it, this defaults to 0")

    parser.add_argument("-dc", "--different_columns", action="store_true", default=False, help="Checks all of the csvs and tar.gz files. This should be used if the csv files do not follow the same pattern.")
    parser.add_argument("-s", "--strict_subclasses", action="store_true", default=False, help="Doesn't find any substrings, only uses the taxon names that you input exactly")

    return parser

def validate_args(parser): # verify the command line options that the user plugged in and print them out
    args = parser.parse_args()
    tars_dir = args.raw_tars

    csvs = []
    file_dict = {}
    if(args.input_drive): # several csv files
        csvs = glob.glob(args.input_drive + "/*.csv")# do something here so that program goes through and gets all of the csv files
        if(len(csvs) == 0):
            parser.error("-d, --input_drive Directory does not contain any csv files")
        print("Input Drive: %s" % (args.input_drive))
        print("Number of Input CSVs: %d" % (len(csvs)))

        # create dictionary linking between tar files and csvs
        file_dict = {}
        for tar in os.listdir(tars_dir):
            unique_name = os.path.basename(tar).replace(".tar","")
            matches = glob.glob(str(args.input_drive) + "/*" + str(unique_name) + "*.csv")
            if (len(matches) >= 1):
                csv_filename = matches[0]
                if (csv_filename not in file_dict):
                    file_dict[csv_filename] = os.path.join(tars_dir, tar)
                else:
                    print("Error: More than one tar file have been matched with a csv file.")
                    exit()

    else: # a single csv files
        print("Input CSV:", args.input_csv)
        for tar in os.listdir(tars_dir):
            unique_name = os.path.basename(tar).replace(".tar","")
            if (unique_name in os.path.basename(args.input_csv)):

                if (args.input_csv not in file_dict):
                    file_dict[args.input_csv] = os.path.join(tars_dir, tar)
                else:
                    print("Error: More than one tar file have been matched with the given csv file.")
                    exit()

        csvs.append(args.input_csv)

    for csv_file in csvs:
        if (csv_file not in file_dict):
            print(file_dict)
            parser.error("r, --raw_tars does not have a corresponding tar for the csv file %s" % csv_file)

    print("Raw tar Directory:", tars_dir)

    if(args.output):
        out_dir = args.output
        if(len(glob.glob(out_dir + "/*")) > 0):
            parser.error("-o, --output Directory can not contain any any files or directories so that we can avoid overwriting existing files.")
        print("Output Directory:", out_dir)

    min_images = args.min_images
    if(min_images < 0):
        parser.error("-m, --min_images The minimum numbers of images you extract needs to be a positive integer.")
    print("Minimum images needed to unzip:", min_images)

    if(args.best_taxon):
        print("\n----- Finding chosen taxon where they are the top probability -----")
    else:
        probability = args.probability_taxon
        print("\n----- Finding chosen taxon with probabilities above %s -----" % probability)

    return csvs, tars_dir, out_dir, file_dict, min_images, args.best_taxon, args.probability_taxon, args.different_columns, args.strict_subclasses

def find_all_taxon(csv): # NOTE: this is assuming the csv's collumns are all the same.
    taxon_exist = next(csv)[1:]
    return taxon_exist


def invalid_taxon(parser, taxon_exist, all_taxon): # see if any of the taxon were not substrings of the column names in the csv files.
    args = parser.parse_args()
    numInvalid = 0 # store the number of invalid taxon
    for i, taxon in enumerate(all_taxon):
        if(taxon not in taxon_exist):
            if(args.different_columns):
                numInvalid += 1
                print("\"%s\" was ignored since it isn't a valid substring of a taxon name for this csv file" % (taxon))
            else:
                parser.error("-t, --taxon Taxon \"%s\" is not a valid substring of a taxon name" % (taxon))
    if numInvalid == len(all_taxon): # determine if there are any valid taxon that were inputted
        return True
    return False

# print out all of the taxon this program is searching for
def print_taxon(taxon_locations):
    print("Finding the sub_taxon:")
    for i, (taxon, sub_taxon) in enumerate(taxon_locations.values()):
        print("%d. %s" % (i, sub_taxon))

def find_top(csv, taxon_locations):
    matched_images = dict()
    for row in csv:
        top_col = 1
        for col, prob in enumerate(row):
            if col < 2: continue # skip the image name row
            # TODO maybe add quickselect here to get the top k elements
            if float(prob) > float(row[top_col]):
                 top_col = col
        if top_col in taxon_locations: # check if top_col is in the dictionary
            image = os.path.basename(row[0])
            matched_images[image] = [taxon_locations[top_col]] # add the image to the matched images dict
    return matched_images

def find_above_prob(csv, taxon_locations, probability):
    matched_images = dict()
    for row in csv:
        for i in range(len(taxon_locations)):
            if float(row[i+1]) > probability: # if one is found this tar.gz now needs to be unzipped
                image = os.path.basename(row[0])
                if image not in matched_images:
                     matched_images[image] = [taxon_locations[i]]
                else:
                     matched_images[image].append(taxon_locations[i]) # add taxon to hashmap for the image
    return matched_images

def untar(tar, out_dir):
    print("Unzipping %s into %s" % (tar, out_dir))
    t = tarfile.open(tar, 'r') # set up the file pointer for the tar file
    t.extractall(out_dir) # all images into the out_dir

def build_structure(path_array, taxon_locations):
    for i in enumerate(path_array): # shouldn't ever be that many, dependant on your minimum detection probability.
        dir_structure = "/" + taxon_locations + "/"
        os.makedirs(out_dir + dir_structure, exist_ok = True) # TODO: maybe move this to another place so that it is not done in a loop
        if(len(path_array) > 1): # then we will need to copy the image to multiple directories
            if(i < len(path_array) - 1):
                shutil.copy2(img, out_dir + dir_structure)
            else: # move the last one instead of copying it
                shutil.move(img, out_dir + dir_structure)
        else:  # move to one directory
            image_unique = add_unique_postfix(out_dir + dir_structure + os.path.basename(img)) # make sure there is a unique image name
            shutil.move(img, image_unique)

if __name__ == "__main__":
    """Main entry to pull_all.py"""
    
    v_string = "V2023.09.05"
    parser = get_parser() # get command line arguments
    csvs, tars_dir, out_dir, file_dict, min_images, best, probability, different, strict = validate_args(parser) # error check arguments further
    print(v_string)
    print(file_dict)

    f = open(csvs[0])
    _csv = csv.reader(f)
    e_taxon = find_all_taxon(_csv) # read the taxon from the first csv file
    f.close()

    # this is the main processing part of the program
    for csv_file in csvs: # loop through all of the csv files
        print("\nFinding images from", csv_file)

        f = open(csv_file)
        _csv = csv.reader(f)
        name_line = next(_csv) # remove nameline
        tar_file = file_dict[csv_file]
        untar(tar_file, out_dir)

        for taxon_locations in e_taxon:
            matched_images = dict()
            matched_images = find_above_prob(_csv, taxon_locations, probability)

            print(f"Found {len(matched_images)} images in {csv_file} for {taxon_locations}")
            print("Building file structure")
            for img in glob.glob(out_dir + f'/{taxon_locations}*.jpg'): # loop through all the images in the untared directory
                path_array = matched_images[os.path.basename(img)]
                build_structure(path_array, taxon_locations) # move all of the images to the correct sub-directory
        f.close()

    print("----- Done -----")
