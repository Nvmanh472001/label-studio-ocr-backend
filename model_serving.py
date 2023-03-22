import binascii
import os
import logging
import random
from inspect import trace
import cv2
from PIL import Image
import numpy as np
from math import dist
from uuid import uuid4

import boto3
import io
import json
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

from paddleocr.tools.infer.predict_det import TextDetector
import paddleocr.tools.infer.utility as utility


from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, \
    get_single_tag_keys, DATA_UNDEFINED_NAME
from label_studio_tools.core.utils.io import get_data_dir
from botocore.exceptions import ClientError
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class OCRLabeling(LabelStudioMLBase):
    def __init__(self,
                 lang_list=None,
                 image_dir=None,
                 labels_file=None,
                 score_threshold=0.3,
                 device='cpu',
                 **kwargs):
        
        super(OCRLabeling, self).__init__(**kwargs)

        lang_list = lang_list or ['vi']

        self.labels_file = labels_file
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')

        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        self.from_name, info = list(self.parsed_label_config.items())[0]
        self.to_name = info['to_name'][0]
        self.value = info['inputs'][0]['value']
        self.labels_in_config = set(info['labels'])

        schema = list(self.parsed_label_config.values())[0]

        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name

        # Text Detection
        args = utility.parse_args()
        infer_args = {
            'use_onnx': True,
            'det_model_dir': "./onnx/detection.onnx",
            "use_gpu": False,
        }

        args.__dict__.update(infer_args)
        self.text_det = TextDetector(args)

        # Text Recognition
        config = Cfg.load_config_from_name("vgg_transformer")
        self.text_rec = Predictor(config)


    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)

        if image_url.startswith('s3://'):
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url

    def predict(self, tasks, **kwargs):
        print(tasks)
        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = self.get_local_path(image_url)
        
        results = []
        img_width, img_height = get_image_size(image_path)

        img, ori_img = cv2.imread(image_path), Image.open(image_path)
        dt_boxes, _ = self.text_det(img)
        dt_boxes = sorted_boxes(dt_boxes)

        boxes = [list_point.tolist() for list_point in dt_boxes]
        img_crop_list = [ori_img.crop(point[0] + point[2]) for point in boxes]
        rec_res = self.text_rec.predict_batch(img_crop_list)

        for bbox, txt in zip(boxes, rec_res): # for each box
            id_gen = str(uuid4())[:10]
            x = bbox[0][0] # x of point bot left
            y = bbox[0][1] # y of point bot left
            box_width = dist(bbox[0], bbox[1]) # width = bf - br
            box_height = dist(bbox[0], bbox[3] ) # heigh = bf - tf
            box_annotations = {
                "original_width": img_width,
                "original_height": img_height,
                "image_rotation": 0,
                "value": {
                    'x': 100 * x / img_width,
                    'y': 100 * y / img_height,
                    'width': 100 * box_width / img_width,
                    'height': 100 * box_height / img_height,
                    'rotation': 0
                },
                "id": id_gen,
                "from_name": "bbox",
                "to_name": "image",
                "type": "rectangle",
                "origin": "manual"
            }

            text_annotations = {
                "original_width": img_width,
                "original_height": img_height,
                "image_rotation": 0,
                "value": {
                    'x': 100 * x / img_width,
                    'y': 100 * y / img_height,
                    'width': 100 * box_width / img_width,
                    'height': 100 * box_height / img_height,
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

        return [{
            'result': results,
        }]

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


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
