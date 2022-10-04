clever-parking
==============================

## Project description

Service for car license plate number, type, color, moving direction automatic detection
The service is based on yolo7 model and EasyOCR project.

## Dataset description

We use open source [Russian plate Dataset](https://universe.roboflow.com/plate-tsusp/russian-plate) for the Object Detection task.

The dataset contains 848 images and was provided by Roboflow.

Roboflow is an end-to-end computer vision platform that helps you:
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Folder to store loaded data for the training/validation/inference
    │   └── result   <- Folder to store inference results
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, experiment runs
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── scripts            <- .sh scripts for the fast .py scripts running
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment`
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download data
    │   │   └── load_dataset.py
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make predictions
    │      ├── predict_image_model.py
    │      └── predict_video_model.py
    ├── yolov7             <- yolov7 project code as git submodule
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## How to run

First of all, you need to install all project requirements: 
```
pip install -r requirements.txt
pip install -r yolov7/requirements.txt
```

the next step is run one of the .sh scripts:
- to train model you can use `scripts/train.sh`
- to test model on the test data you can use `scripts/test.sh`
- to predict for the image you can use `scripts/predict_for_image.sh`
- to train model you can use `scripts/predict_for_video.sh`

You can edit these scripts if you need.
