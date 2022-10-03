
from roboflow import Roboflow
from settings import ROBOFLOW_KEY

def load_dataset():
    rf = Roboflow(api_key=ROBOFLOW_KEY)
    project = rf.workspace("plate-tsusp").project("russian-plate")
    dataset = project.version(3).download("yolov7")
    return dataset.location