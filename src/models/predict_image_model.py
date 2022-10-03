import logging

import click
import cv2
import easyocr
import numpy as np
import torch
import yaml
from numpy import random

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import (check_img_size,
                                  non_max_suppression,
                                  scale_coords, set_logging)
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import (select_device, time_synchronized)
from .utils import letterbox

reader = easyocr.Reader(["en"], gpu=True)
classes_to_filter = None

#TODO: refactor, manual testing

@click.command()
@click.option(
    "-cf",
    "--config_path",
    type=click.Path(exists=True),
    help="Path to config file",
    required=True,
)
def predict(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    with torch.no_grad():
        weights, imgsz = config["model"]["trained_weights"], config["img_size"]
        set_logging()
        device = select_device(config["device"])
        half = device.type != "cpu"
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()

        names = model.module.names if hasattr(model, "module") else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if device.type != "cpu":
            model(
                torch.zeros(1, 3, imgsz, imgsz)
                .to(device)
                .type_as(next(model.parameters()))
            )

        img0 = cv2.imread(source_image_path)
        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        classes = None
        if config["classes"]:
            classes = []
            for class_name in config["classes"]:
                classes.append(config["classes"].index(class_name))

        pred = non_max_suppression(
            pred,
            config["conf_threshold"],
            config["iou-thres"],
            classes=classes,
            agnostic=False,
        )

        t2 = time_synchronized()
        for i, det in enumerate(pred):
            s = ""
            s += "%gx%g " % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):

                    label = f"{names[int(cls)]} {conf:.2f}"
                    plot_one_box(
                        xyxy,
                        img0,
                        label=label,
                        color=colors[int(cls)],
                        line_thickness=3,
                    )

                    img_cropped = img0[
                        int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2])
                    ]
                    gray = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2GRAY)
                    ocr_result = reader.readtext(gray)
                    text_predicted = ocr_result[0][1]
                    confidence_text_pred = ocr_result[0][2]
                    cv2.putText(
                        img0,
                        text_predicted + ", conf: " + str(confidence_text_pred),
                        (50, 70),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        1,
                        (50, 50, 255),
                        2,
                    )
                    cv2.imwrite(config["output_path"], img0)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    predict()
