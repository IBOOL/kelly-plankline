This is an _actively_ developed set of documentation for the setup and use of the UAF version of the planktline processing pipeline developed by OSU. Please contact Thomas Kelly <tbkelly@alaska.edu> for more information.

Updated 2023-08-11

## 1. Segmentation

On the first install, 

    cd /opt
    sudo git clone https://github.com/tbrycekelly/UAF-Segmentation
    sudo chown -R plankline:plankline /opt/UAF-Segmentation
    sudo chmod 776 -R /opt/UAF-Segmentation

To update the executable with a new version:

    cd /opt/UAF-Segmentation
    git clean -f
    git pull

To build the segmentation executable:

    cd /opt/UAF-Segmentation/build
    cmake ../
    make

Test it and copy it to final directory (if it works):

    ./segment
    cp ./segment ..


## 2. SCNN

For the first install,

    cd /opt
    sudo git clone https://github.com/tbrycekelly/UAF-SCNN.git
    sudo chown -R plankline:plankline /opt/UAF-SCNN
    sudo chmod 776 -R /opt/UAF-SCNN

To update from github,

    cd /opt/UAF-SCNN
    git clean -f
    git pull

May need to run `git config --global --add safe.directory /opt/UAF-SCNN` if permissions are not right.

To build SCNN:

    cd /opt/UAF-SCNN/build
    make clean
    make wp2

If there are undefined refences on the make, check that the LIBS line in ./build/Makefile is the output of `pkg-config opencv --cflags --libs` and inclues `-lcublas`.

Test it and copy it to final directory (if it works):

    ./wp2
    cp ./wp2 ../scnn

### 2.1 Training a Neural Network

When setting up a training library of curated images, we recommend to aim for more than ~500 images per category yielding. This will yield >100 validation images when training is performed with a 20% calidation data set.

**Overview of Traing Steps**
1. Place training set into /media/plankline/Data/Training folder
    - In particular, the folder structure of this training set includes the ./Data (location of classified images) and ./weights (initially empty) subdirectories.

TODO: Check if /weights/ folder is created automatically. 
TODO: Make the folder names not be uppercase

2. SSH into Plankline, then navigate to /media/plankline/Data/Training
3. Run `./classList.sh` and give it the name of the path to the training images. This will print off the number of images per each subfolder. It will also write this information to _classList_, a text file in `./<trainingset>/Data`. An example call will look like:

    ./classList.sh ./training_set_20231002/Data

    
4. Navigate to the UAF-Plankline script folder, for example: `cd /media/plankline/Data/Scripts/UAF-Plankline`
5. Edit _default.ini_ as needed, specifically the **training** section. We recommended running the training script for a couple epochs (1-2) first, to make sure it works, then run for a larger number. So for example `start = 0` and `end = 1` within _default.ini_.
6. We can run train.py:

    python3 ./train.py -c default.ini


__NB:__ If the command line is disconnected, then the thread will stop, so we recommend using the command `screen` to allow for easy detaching and reataching of command sessions. To use, run `screen` in your current terminal window. This will spawn a new virtual session that you can later detach. Run any/all commands you would like, for example the above _training.py_ call. To allow the process to continue even if your local computer disconnects from the server, press `Ctrl+a` then `d` to detach. You can now exit or close the window and the current session will persist in the background on the server.

To later reatach the session, simply run `screen -r`. If there are multiple screen sessions available then you can reatch a sepcific one with `screen -r PID` where _PID_ is the process ID shown on screen. To close (i.e. terminate) a screen session, either reattach the session then `Ctrl+a` then `\` or from outside screen run `killall -i screen` to termiante the processes directly.




## 3. Plankline Scripts
This can be performed by any/every user on a system and can be placed in any folder. So far we have been placing this folder, /UAF-Plankline/ in Tom's documents folder. Typical locations that make sense include ~ (home directory), ~/Desktop, ~/Documents.

    cd ~
    git clone https://github.com/tbrycekelly/UAF-Plankline.git

To update an existing folder (will remove the config files present):

    cd ~/UAF-Plankline
    git clean -f
    git pull

To run plankline with a specific configuration file (required):

    python3 segmentation.py -c <ini> -d <dir>
    python3 classification.py -c <ini> -d <dir>

So for example:

    python3 segmentation.py -c osu_test_config.ini -d /media/plankline/data/test_data

The segmentation.py script will read in a project directory containing a _raw_ subfolder of AVIs and will create a _segmentation_ subfolder if not already existing. Each AVI will be processed, flatfielded, and cropped to a dedicated TAR file inside _segmentation_. If using the optional _compress = True_ flag then the file will be a TAR.GZ file. Simiarly the _classification.py_ script will read in a project directory and for every file in _segmentation_ produce a classification results file in a _classification_ subfolder. Both scripts will place a copy of the configuration file used into their respective subfolders for archival purposes.

These scripts require the segmentation executable and SCNN executable to be available (_see 1 and 2_).


