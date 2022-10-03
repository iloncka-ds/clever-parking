import os

import click
import logging




@click.command()
@click.argument('confidence', type=click.FLOAT)
@click.argument('test_dataset_location', type=click.Path(exists=True))
def test_model(confidence=0.45, test_dataset_location='/test/images'):
    """ Get predictions on test set"""
    
    logger = logging.getLogger(__name__)
    logger.info('get predictions on test')
    os.system('cmd /c "cd /yolov7"')
    os.system('cmd /c "python detect.py --weights yolov7/runs/train/exp/weights/best.pt --conf {confidence} --source {test_dataset_location}"')
    
