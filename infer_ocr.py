import argparse

from paddleocr.tools.infer.predict_det import TextDetector
import paddleocr.tools.infer.utility as utility

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


from PIL import Image
import utility_func
from preprocess import pdf2Image
import os
import cv2

import time
import json
from uuid import uuid4
from math import dist

args = utility.parse_args()
infer_args = {
    'use_onnx': True,
    'det_model_dir': "./onnx/detection.onnx",
    "use_gpu": False,
}

args.__dict__.update(**infer_args)


def sorted_boxes(dt_boxes):
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def create_image_url(path):
    name = path.split('/')[-1]
    return f'/data/local-files/?d=imgs/{name}'

def create_annotations(img, dtboxes, rec_res, path):
    # "/data/local-files/?d=imageData/{name}.jpg"
    results  = []
    h, w = img.shape[:2]
    for bbox, txt in zip(dtboxes, rec_res): # for each box
        id_gen = str(uuid4())[:10]
        x = bbox[0][0] # x of point bot left
        y = bbox[0][1] # y of point bot left
        width = dist(bbox[0], bbox[1]) # width = bf - br
        height = dist(bbox[0], bbox[3] ) # heigh = bf - tf
        box_annotations = {
            "original_width": w,
            "original_height": h,
            "image_rotation": 0,
            "value": {
               'x': 100 *x / w,
                'y': 100 *y / h,
                'width': 100 * width / w,
                'height': 100 * height / h,
                'rotation': 0
            },
            "id": id_gen,
            "from_name": "bbox",
            "to_name": "image",
            "type": "rectangle",
            "origin": "manual"
        }
        
        text_annotations = {
            "original_width": w,
            "original_height": h,
            "image_rotation": 0,
            "value": {
                'x': 100 *x / w,
                'y': 100 *y / h,
                'width': 100 * width / w,
                'height': 100 * height / h,
                'rotation': 0,
                'text': [
                    txt
                ]
            },
            "id": id_gen,
            "from_name": "transcription",
            "to_name": "image",
            "type": "textarea",
            "origin": "manual",
        }
        
        results.extend([box_annotations, text_annotations])
    return {
        'data':{
            'ocr': create_image_url(path)
        },
        'predictions': [{
            'result': results
        }]
    }


img_args = utility_func.init_args()
#===================  # Load model

text_det = TextDetector(args) # init model paddel to  TextDetector

config = Cfg.load_config_from_name('vgg_transformer') # load weigh vietocr
detector = Predictor(config) # init model vietocr to OCR


#=================== # convert pdf to image

image_dir = pdf2Image(img_args.pdf_dir, img_args.image_dir) # convert pdf to imgs
image_files = os.listdir(image_dir)

def is_image(path):
    end = ["png", "jpeg", "jpg"]
    return any([path.endswith(e) for e in end])

image_paths = [os.path.join(image_dir, fname) for fname in image_files if is_image(fname)] # get path file img
#===================


# img_paths = ["./imgs/3.jpg"]
# img_cv = cv2.imread(img_path)
# img_pil = Image.open(img_path)

tasks = []
for img_path in image_paths:
    
    img_cv = cv2.imread(img_path)
    img_pil = Image.open(img_path)

    dt_boxes, timeer = text_det(img_cv)
    dt_boxes = sorted_boxes(dt_boxes)

    boxes = [list_point.tolist() for list_point in dt_boxes]
    img_crop_list = [img_pil.crop(point[0] + point[2]) for point in boxes]

    start = time.time()
    rec_result = detector.predict_batch(img_crop_list)
    end = time.time()

    print(f'hi: {end - start}')

    task = create_annotations(img_cv, dt_boxes, rec_result, img_path)
    tasks.append(task)
    with open('ocr_tasks.json', mode='w') as f:
        json.dump(tasks, f, indent=2)

print('Done!')