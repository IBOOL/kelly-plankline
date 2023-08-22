# Documentation for UAF-Plankline
N.B. This version of plankline was forked and adapted significantly from the [original OSU version](https://zenodo.org/record/4641158)(DOI: 10.5281/zenodo.4641158)[[1]](#1). The following is a high-level summary of the changes applied in no particular order.

1. The segmentation and SCNN executables have been relocated to /opt/Threshold-MSER and /opt/SCNN, respectively. This allows all users to use these utilities without having to change/fix the paths.

2. All settings for segmentation and classification are now set within the .ini file itself. This is the only file that should be modified between projects/runs/etc. I am still working on the inline documentation in the config file. Eventually will make a "default.ini" to make copies of.

3. A new set of logging utilities are built in to allow for detailed and configurable logging routines to be used. All log entries are timestamped and given a status level: INFO, DEBUG, WARN, ERROR, CRITICAL. Default log files are places into the working directory (e.g. /media/plankline/Data/Data/projectX/) and timestamped. Logs for the executables are located in their respective folders.

4. Only some messages are printed to the screen by default (except warnings and errors), including only key information. A progress bar is shown for the majority of the processing.

5. Both scripts are now run by calling the python file directly.

6. Currently I have not implemented/adapted the previous directory/file verification functions that plankline used. So both scripts will rewrite all existing files and does not check/verify if output exists.

---

# Installation

Please see [Plankline Setup](Plankline Setup.md) for installation instructions.


# Segmentation (MSER)

The segmentation algorithm used in the _Plankline_ processing suite is nomally a __Maximally Stable Extremal Region__ approach as implemented in _opencv_, but there are caveats. Depending onthe signal to noise ratio hyperparamter (__SNR__), one of three possible segmentation routines are called. SNR is calculated as follows (imageProcessing.cpp:497).

    float SNR(const cv::Mat& img) {
        // perform histogram equalization
        cv::Mat imgHeq;
        cv::equalizeHist(img, imgHeq);

        // Calculate Signal To Noise Ratio (SNR)
        cv::Mat imgClean, imgNoise;
        cv::medianBlur(imgHeq, imgClean, 3);
        imgNoise = imgHeq - imgClean;
        double SNR = 20*( cv::log(cv::norm(imgClean,cv::NORM_L2) / cv::norm(imgNoise,cv::NORM_L2)) );

        return SNR;

When the SNR of an image is greater than that provided, then the MSER algorithm is called immediately after flatfielding and preprocessing (with a `3x3` kernel) [[2]](#2). If the SNR is greater than 75% of the hyperparamter value, then the image is flatfielded, proprocessed (with a `17x17` kernel), and then thresholded based on the __threshold__ hyperparameter. A contouring algorithm is then applied to find specific ROIs. Finally, if the image SNR is below 75% of the hyperparamter value, then the image is flatfielded, proprocessed using the same `17x17` kernel and otherwise processed identically to the previous processing (but in a standalone function). 



# Training


# Classification



## References

<a id="1">[1]</a> Schmid, Moritz S, Daprano, Dominic, Jacobson, Kyler M, Sullivan, Christopher, Briseño-Avena, Christian, Luo, Jessica Y, & Cowen, Robert K. (2021). A Convolutional Neural Network based high-throughput image classification pipeline - code and documentation to process plankton underwater imagery using local HPC infrastructure and NSF's XSEDE (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.4641158

<a id='2'>[2]</a> Preprocessing here entails an erosion and dialation step conducted by _opencv_. The erosion kernel size is `2*size+1` and thus is a `3x3` or `17x17`
