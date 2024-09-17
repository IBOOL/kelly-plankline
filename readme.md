# Plankline Scripts
Scripts were originally from the Plankline project controlled by OSU (citation's to be included). I have heavily modified and repurposed the code to suit our architecture and to improve logging and performance. These scripts are available as is and I make no claim to ownership or control of any potentially sensitive code. These are intended for in-house use at UAF.


## Setup and Documentation

The Plankline suite of processing scripts is split into three main modules: (1) segmentation, (2) classification, (3) analysis; and is unified with the scripts in this repository. 

If you are new to linux and setting up a Plankline instance for the first time, please start with the [Linues Setup](Notes/Linux.md) and the [Plankline Setup](Notes/General%20Setup.md) guides.

For documentation on how Plankline works, please see the [Plankline](Notes/Plankline.md) reference.

Optionally we also have a preliminary [Morphocluster](Notes/Morphocluster.md) setup guide for those that are interested.


## Quick start

To run plankline with a specific configuration file and input directory:

    python3 segmentation.py -c <ini> -d <dir>
    python3 classification.py -c <ini> -d <dir>

So for example:

    python3 segmentation.py -c osu_test_config.ini -d /media/plankline/data/test_data

