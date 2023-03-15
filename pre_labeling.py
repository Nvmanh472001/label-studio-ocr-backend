from txt_det_onnx import DetectionONNX
from txt_rec_onnx import VietOCRONNX
import utility_func
from preprocess import pdf2Image

from PIL import Image
import random as rd
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

def create_annotations(img, dtboxes, rec_res):
    results  = []
    w, h = img.shape[:1]
    for bbox, txt in zip(dtboxes, rec_res):
        id_gen = rd.randrange(10**10)
        box_annotations = {
            "original_width": w,
            "original_height": h,
            "image_rotation": 0,
            "value": {
                "x": 0,
                "y": 0,
                "width": 0,
                "height": 0,
                "rotation": 0,
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
                "x": 40.09441808289057,
                "y": 11.039487623283758,
                "width": 13.743055254827269,
                "height": 2.225703149855598,
                "rotation": 0,
                "text": [
                    "Work History"
                ]
            },
            "id": id_gen,
            "from_name": "transcription",
            "to_name": "image",
            "type": "textarea",
            "origin": "manual",
        }
        
        results.append([box_annotations, text_annotations])
    
    return results
  

def main():
    args = utility_func.init_args()
    
    text_detection = DetectionONNX(args)
    text_recognizer = VietOCRONNX(args)
    

    image_dir = pdf2Image(args.pdf_dir, args.image_dir)
    image_files = os.listdir(image_dir)
    
    def is_image(path):
        end = ["png", "jpeg", "jpg"]
        return any([path.endswith(e) for e in end])
    
    image_paths = [os.path.join(image_dir, fname) for fname in image_files if is_image(fname)]
    
    images = [cv2.imread(image_path) for image_path in image_paths]
    for img in images:
        ori_img = img.copy()
        dt_boxes = text_detection(img)
        
        dt_boxes = sorted_boxes(dt_boxes)
        img_crop_list = []
        
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_img, tmp_box)
            img_crop = Image.fromarray(img_crop)
            
            img_crop_list.append(img_crop)
        
        rec_res = text_recognizer(img_crop_list)
        
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= args.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

if __name__ == "__main__":
    main()