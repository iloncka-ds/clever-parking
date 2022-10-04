clever-parking
==============================

## Project description

Service for car license plate number, type, color, moving direction automatic detection
The service is based on yolo7 model and EasyOCR project.

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
    │      ├── predict_model.py
    │      └── train_model.py
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