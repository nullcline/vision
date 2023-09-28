import sys
sys.path.append('./yolov7')

import cv2
import torch
import numpy as np
import os
from numpy import random
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import (check_img_size,
                           non_max_suppression,
                           scale_coords,
                           set_logging,
                        )


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class DiskDetector:
    def __init__(self):
        classes_to_filter = None  # Example ['bob','stewart'] Note that capital matters. Match class to labelled image name exactly.

        self.opt = {
            # Path to weights file default weights are for nano model
            "weights": os.getcwd() + r'\best.pt',
            "yaml": os.getcwd() + r'\opt.yaml',
            "img-size": 640,  # default image size
            "conf-thres": 0.40,  # confidence threshold for inference.
            "iou-thres": 0.45,  # NMS IoU threshold for inference.
            "device": "0",  # device to run our model i.e. 0 or 0,1,2,3 or cpu
            "classes": classes_to_filter  # list of classes to filter or None

        }

    @torch.no_grad()
    def setup(self):
        weights, self.imgsz = self.opt['weights'], self.opt['img-size']
        set_logging()
        self.device = select_device(self.opt['device'])
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        if self.half:
            self.model.half()

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))


    @torch.no_grad()
    def predict(self, img0):
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=False)[0]

        # Apply NMS
        classes = None
        if self.opt['classes']:
            classes = []
            for class_name in self.opt['classes']:
                classes.append(self.opt['classes'].index(class_name))

        pred = non_max_suppression(pred, self.opt['conf-thres'], self.opt['iou-thres'], classes=classes, agnostic=False)

        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    yield int((xyxy[0].item() + int(xyxy[2].item()))/2), int((xyxy[1].item() + int(xyxy[3].item()))/2)


if __name__ == "__main__":
    # example usage
    ytf = DiskDetector()
    # run whenever you want to load the model
    print("Loading model")

    ytf.setup()

    img_path = os.getcwd() + r'\data\grass.png'
    img = cv2.imread(img_path)
    top_left, bottom_right = ytf.predict(img)
    print(top_left, bottom_right)