import argparse

from paddleocr.tools.infer.predict_det import TextDetector
from paddleocr.tools.infer.predict_rec import TextRecognizer
import paddleocr.tools.infer.utility as utility

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


import fasttext
import random

from PIL import Image
# import utility_func
# from preprocess import pdf2Image
import os
import cv2
import numpy as np

import time
import json
from uuid import uuid4
from math import dist
from PIL import Image


args = utility.parse_args()
infer_args = {
    'use_onnx': True,
    'det_model_dir': "./onnx/detection.onnx",
    "use_gpu": False,
}

args.__dict__.update(**infer_args)


rec_args = utility.parse_args()

rec_infer_args = {
    'use_onnx': True,
    'rec_model_dir': "./onnx/Recognizer.onnx",
    # "use_gpu": True,
    'rec_char_dict_path': './config/en_dict.txt'
}
rec_args.__dict__.update(**rec_infer_args)


def sorted_boxes(dt_boxes):
    # real_box = copy.deepcopy(dt_boxes)
    # real_box[:,:,1] = real_box[:,:,1] + height_page
    # real_box = sorted(real_box, key=lambda x: (x[0][1], x[0][0]))

    # #=========
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
    return f'/data/local-files/?d=imgs/{name}' # imgs is folder storage all images

def create_annotations(dtboxes, rec_res, path):
    # "/data/local-files/?d=imageData/{name}.jpg"
    img = cv2.imread(path)

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



def check_and_read(pdf_path):
    # os.path.basename(pdf_path)[-3:] in ['pdf']:
    img_path = './imgs/'+ pdf_path.split('/')[-1][:-4].replace(' - ','-')+'.jpg' # đường dẫn cần lưu ảnh

    import fitz
    from PIL import Image
    imgs = []

    with fitz.open(pdf_path) as pdf:
        for pg in range(0, pdf.pageCount):
            page = pdf[pg]
            mat = fitz.Matrix(2, 2)
            pm = page.get_pixmap(matrix=mat, alpha=False)

            # if width or height > 2000 pixels, don't enlarge the image
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            imgs.append(img)


    vis = imgs[0]
    for i in range(1, len(imgs)):
        vis = np.concatenate((vis, imgs[i]), axis=0)

    cv2.imwrite(f'{img_path}',vis)
    

    return imgs, img_path, True




# img_args = utility_func.init_args()
#===================  # Load model

text_det = TextDetector(args) # init model paddel to  TextDetector

text_rec = TextRecognizer(rec_args)

config = Cfg.load_config_from_name('vgg_transformer') # load weigh vietocr
detector = Predictor(config) # init model vietocr to OCR

model_fasttext = fasttext.load_model('./lid.176.bin')



#=================== # convert pdf to image


pdf_paths = os.listdir("./pdf")
pdf_paths = [os.path.join("./pdf", fname) for fname in pdf_paths if fname.endswith('pdf')]


decode_pdf = [check_and_read(path)[0] for path in pdf_paths]
path_imgs = [check_and_read(path)[1] for path in pdf_paths]


tasks = []
for pdf, img_path in zip(decode_pdf, path_imgs): 
    tolal_box = []
    tolal_rec = []
    height_page = 0
    try:
        for page in pdf:  # page is np.array read with opencv
        # Chuyển đổi mảng NumPy sang đối tượng Image của PIL
            img_pil = Image.fromarray(page)
            
            dt_boxes, timeer = text_det(page)
            dt_boxes = sorted_boxes(dt_boxes)
            
            real_box = np.array(dt_boxes)
            real_box[:,:,1] = real_box[:,:,1] + height_page
            
            tolal_box += list(real_box)
            height_page += page.shape[0]


            boxes = [list_point.tolist() for list_point in dt_boxes]
            img_crop_list = [img_pil.crop(point[0] + point[2]) for point in boxes]
        

            start = time.time()
            rec_result = detector.predict_batch(img_crop_list)

            # check langage
            s = random.choices(rec_result, k = 5)
            s = ' '.join(s)
            print(s)
            lang = model_fasttext.predict(s)

            if lang[0][0][-2:] == 'vi':
                pass
            else: # if en_cv chuyển box qua np.array(định dạng khi đọc bằng cv2) 
                img_crop_list = [cv2.cvtColor(np.array(e), cv2.COLOR_RGB2BGR) for e in img_crop_list]
                rec_result, _ = text_rec(img_crop_list)
                
                rec_result = [e[0] for e in rec_result]
            # print('Done')

            

            end = time.time()
            tolal_rec += rec_result

            print(f'hi: {end - start}')
    except:
        print(img_path)
        continue

    # print(len(tolal_box), len(tolal_rec))
    # print(height_page)
    
    
    task = create_annotations(tolal_box,tolal_rec,img_path)
    tasks.append(task)
    with open('predict.json', mode='w') as f:
        json.dump(tasks, f, indent=2)


print('All is done')




