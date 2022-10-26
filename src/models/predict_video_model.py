"""import logging
import sys
from pathlib import Path

import click
import cv2
import easyocr
import numpy as np
import torch
import yaml
from numpy import random

sys.path.append(str(Path(__file__).resolve().parents[2]))


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device


def setup_model(device, half, image_size, weights):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    image_size = check_img_size(image_size, s=stride)  # check img_size
    if half:
        model.half()
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != "cpu":
        model(torch.zeros(1, 3, image_size, image_size).to(device).type_as(next(model.parameters())))
    return colors, image_size, model, names, stride


@click.command()
@click.option(
    "-cf",
    "--config_path",
    type=click.Path(exists=True),
    help="Path to config file",
    required=True,
)
@click.option(
    "-vf",
    "--video_path",
    type=click.Path(exists=True),
    help="Path to image",
    default="data/sample_video.mp4",
)
def predict(config_path, video_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    video = cv2.VideoCapture(video_path)
    video_path = Path(video_path).name
    result_path = str(Path(__file__).resolve().parents[2] / Path(config["predict"]["output_path"]) / video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    output = cv2.VideoWriter(
        result_path,
        cv2.VideoWriter_fourcc(*"DIVX"),
        fps,
        (w, h),
    )
    torch.cuda.empty_cache()

    with torch.no_grad():
        weights, image_size = (
            config["model"]["trained_weights"],
            config["model"]["img_size"],
        )
        set_logging()
        device = select_device(config["model"]["device"])
        half = device.type != "cpu"
        colors, image_size, model, names, stride = setup_model(device, half, image_size, weights)

        classes = None
        if config["model"]["classes"]:
            classes = []
            for class_name in config["model"]["classes"]:
                classes.append(config["model"]["classes"].index(class_name))

        for j in range(n_frames):
            ret, img_0 = video.read()
            if ret:
                img = process_img(device, half, image_size, img_0, stride)
                prediction = model(img, augment=False)[0]
                prediction = non_max_suppression(
                    prediction,
                    config["model"]["conf_threshold"],
                    config["model"]["iou_threshold"],
                    classes=classes,
                    agnostic=False,
                )

                for i, det in enumerate(prediction):
                    s = "%gx%g " % img.shape[2:]
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_0.shape).round()

                        for c in det[:, -1].unique():
                            # detections per class
                            n = (det[:, -1] == c).sum()
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                        has_number = False
                        for *xyxy, conf, cls in reversed(det):
                            label = f"{names[int(cls)]} {conf:.2f}"
                            plot_one_box(
                                xyxy,
                                img_0,
                                label=label,
                                color=colors[int(cls)],
                                line_thickness=3,
                            )
                            img_cropped = img_0[int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2])]
                            gray = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2GRAY)
                            ocr_result = reader.readtext(gray)
                            if len(ocr_result) == 0:
                                continue
                            text_predicted = ocr_result[0][1]
                            print(f"Detected number: {text_predicted}")
                            confidence_text_prediction = str(ocr_result[0][2])
                            if has_number:
                                continue
                            cv2.putText(
                                img=img_0,
                                text=f"{text_predicted}",
                                org=(50, 70),
                                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=1,
                                color=(50, 50, 255),
                                thickness=2,
                            )
                            has_number = True
                print(f"{j+1}/{n_frames} frames processed")
                output.write(img_0)
            else:
                break
    output.release()
    video.release()


def process_img(device, half, image_size, img_0, stride):
    img = letterbox(img_0, image_size, stride=stride)[0]

    # BGR to RGB, to 3x416x416
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)

    # uint8 to fp16/32
    img = img.half() if half else img.float()

    # 0 - 255 to 0.0 - 1.0
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    reader = easyocr.Reader(["en"], gpu=True)
    predict()
"""
