import os

import click
import logging

from load_dataset import load_dataset

model_weights = 'yolov7.pt'
dataset_location = load_dataset()

@click.command()
@click.argument('model_weights', type=click.STRING)
@click.argument('dataset_location', type=click.Path(exists=True))
def train_model(model_weights, dataset_location):
    """ Train model from pretrained model_weights on custom dataset"""
    
    logger = logging.getLogger(__name__)
    logger.info('train model')
    # get model_weights weights
    os.system('cmd /c "wget -P /yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{model_weights}"')
    os.system('cmd /c "cd /yolov7"')
    os.system('cmd /c "python train.py --batch 16 --cfg cfg/training/yolov7.yaml --epochs 55 --data {dataset_location}/data.yaml --weights {model_weights} --device 0"') 
    