clever-parking
==============================

## Project description

Service for car license plate number automatic detection
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

## Experiments

## Metrics

- **Precision** measures how accurate is your predictions. i.e. the percentage of your predictions are correct.
- **IoU** measures the overlap between 2 boundaries. We use that to measure how much our predicted boundary overlaps with the ground truth (the real object boundary).
- **Recall** measures how good you find all the positives. For example, we can find 80% of the possible positive cases in our top K predictions. 
- **AP** is the average over multiple IoU (the minimum IoU to consider a positive match). AP@[.5:.95] corresponds to the average AP for IoU from 0.5 to 0.95 with a step size of 0.05. **mAP** (mean average precision) is the average of AP. 

### Experiments setup

- Hardware
    - CPU count: 1
    - GPU count: 1
    - GPU type: Tesla T4
- Software:
    - Python version: 3.7.14
    - OS: Linux-5.10.133+-x86_64-with-Ubuntu-18.04-bionic
- Training params:
    - Batch size: 16
    - Image size: 640x640
    - Epochs: 55
    - Weight decay: 0.0005
    - Warmup momentum: 0.8
    - Warmup epochs: 3

| Model  | Precision | Recall | mAP@[.5] | mAP@[.5:.95] |
| ------------- | ------------- |------------- |------------- |------------- |
|  yolov7-e6   |  0.2246 | 0.3684 | 0.1329 | 0.04663 |
|  yolov7-w6   |  0.9885 | 0.05263 | 0.05659 | 0.005891 |
|  yolov7x   |  0.2692 | 0.2632 | 0.1553 | 0.04348 |
|  yolov7   |  0.9999  | 1 | 0.9952 | 0.7518 |

[Weights&Biases Experiments report](https://wandb.ai/dl-learning/YOLOR/reports/---VmlldzoyNzQ0Mzk1?accessToken=1rma5zlp2ee59b9uxart20401fiofe9d3q0jml03mkozgwbktvdjmav20tideqdz)

###


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

Main project parameters are settled in [params.yaml](https://github.com/iloncka-ds/clever-parking/blob/main/params.yaml) so you can edit this file too
