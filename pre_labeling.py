from txt_det_onnx import DetectionONNX
from txt_rec_onnx import VietOCRONNX
import utility
from preprocess import pdf2Image

from PIL import Image
import numpy as np
import cv2
import copy
import os


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


def get_rotate_crop_image(img, points):
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def formatted_to_json(dtboxes, rec_res):
    results  = []
    

def main():
    args = utility.parse_args()
    
    text_detection = DetectionONNX(args)
    text_recognizer = VietOCRONNX(args)
    

    image_dir = pdf2Image(args.pdf_dir, args.image_dir)
    image_files = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, fname) for fname in image_files if utility.is_image(fname)]
    
    images = [Image.open(image_path) for image_path in image_paths]
    for img in images:
        ori_im = img.copy()
        dtboxes = text_detection(img)
        
        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        
        rec_res = text_recognizer(img_crop_list)
        
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= args.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        