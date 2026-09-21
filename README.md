# exam
This repository include science data and python scripts for visualizing, reducing and analyzing a time series of astronomical images and plot the lightcurve of a variable star.

Folder "2026" contains two subfolders:
- "sci" is filled with a time series of 314 fits images of the variable star AE-UMa;
- "calib" includes dark, flat and bias exposures to perform reduction of raw data into data viable for scientific measurements.

The code is organized in four python scripts:
- "reader.py" performs visualization of a single fits file of choice, printing key infos from the header and the image;
- "masters.py" creates calibration fits files from data in "calib" folder;
- "reduction.py" performs raw data reduction using the calibration files and creates a Multi-Extension FITS (mef) file for each frame, including the processed image and the associated error map for error propagation;
- "photometry.py" performs aperture photometry on AE-UMa processed frames, using a centroiding algorithm to find the exact position of the star and calculating the aperture radius as a function of the Signal-to-Noise Ration (SNR). Then, using a reference star with known magnitude, calculates the differential magnitude of AE-UMa and plots the lightcurve for the entire period of observation. 

Before executing the code make sure to have a working installation of Python3 in your system and install the following packages: numpy, scipy, matplotlib, photutils, ccdproc, astropy, pandas. Here's how to do it (Linux):
- Create the virtual environment (let's call it exam):
> python3 -m venv exam

- Activate the environment:
> source exam/bin/activate

- Install the required packages:
> python3 -m pip install numpy scipy matplotlib photutils ccdproc astropy pandas

Now you can visualize one frame at time executing "reader.py" (the choice of the file is editable in the script) and build the lightcurve executing in order "masters.py", "reduction.py" and "photometry.py".
