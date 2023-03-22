import argparse

from paddleocr.tools.infer.predict_det import TextDetector
import paddleocr.tools.infer.utility as utility

from PIL import Image

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


text_det = TextDetector(args)
import cv2 
img_path = "./imgs/1.jpg"
img_cv = cv2.imread(img_path)
img_pil = Image.open(img_path)
dt_boxes, time = text_det(img_cv)
dt_boxes = sorted_boxes(dt_boxes)

boxes = [list_point.tolist() for list_point in dt_boxes]
img_crop_list = [img_pil.crop(point[0] + point[2]) for point in boxes]

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
detector = Predictor(config)


rec_result = detector.predict_batch(img_crop_list)
print(rec_result)
print(len(boxes), len(rec_result))


from uuid import uuid4
from math import dist

results  = []
w, h = img_cv.shape[:2]
for bbox, txt in zip(boxes, rec_result): # for each box
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
            'x': 100 * x / w,
            'y': 100 * y / h,
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

print(results)


