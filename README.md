# Real-time FER Classifier

## Overview
This repository contains driver code that helped me perform real-time testing on my best performing FER classification model.

The repository is comprised of five folders and five Python scripts:
- data:
    - this folder contains all the test images and videos that are used as input data for the model
- haarcascade classifiers:
    - this folder contains a HaarCascade frontal-face classifier used for face detection
- models:
    - this folder contains the saved model in a H5 and JSON file format
- saved-frames:
    - this folder contains the frames extracted from the live feed after detecting an emotion
- training:
    - this folder contains the notebook that was used for exploring the dataset, training and tuning the model
- Python Scripts:
    - Main.py
    - HaarCascade.py
    - MediaPipe.py
    - Images.py
    - Videos.py


## Requirements
- Python 3
- OpenCV
- TensorFlow 
- Keras


## How to test the model on Images
- Select an image file path of an emotion in the Images.py file
- Select one of the following face detection classifiers in Main.py (line 29): HaarCascade or MediaPipe
- Uncomment line 16 on the Main.py script
- Run the Main.py file

## How to test the model on Videos
- Select a video file path of an emotion in the Videos.py file
- Select one of the following face detection classifiers in Main.py (line 29): HaarCascade or MediaPipe
- Uncomment line 17 on the Main.py script
- Run the Main.py file

## How to test the model on a Webcam
- Select one of the following face detection classifiers in Main.py (line 29): HaarCascade or MediaPipe
- Uncomment line 18 on the Main.py script
- Run the Main.py file


## Acknowledgements
- Both HaarCascade.py and MediaPipe.py incorporate code that do not belong to me, they were merely used and repurposed to achieve the objective of real-time emotion classification. Credits are due to the following sources:
    - OpenCV: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
    - Advanced Computer Vision with Python - Full Course: https://youtu.be/01sAkU_NvOY?t=5307


## System Limitations
- MediaPipe face detection causes the program to crash due to less RAM and CPU cores. In this case, a GPU is advisable, however it isn't required
- If your face is out of bounds from the live webcam, the program will crash
    - This is because MediaPipe requires coloured frames as an input for face detection, while the model requires grayscale frames as an input for emotion classification. Therefore, your face must always be within the bounds of the window to prevent the program from crashing
