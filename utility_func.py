import os, sys
from PIL import Image
import argparse

def init_args():
    def str2bool(arg_value):
        return arg_value.lower() in ("true", "t", "1")
    
    parser = argparse.ArgumentParser()
    
    # Params for prelabeling predict engine
    parser.add_argument('--use_gpu', type=str2bool, default=False)
    
    # Params for preprocess convert pdf to image
    parser.add_argument('--pdf_dir', types=str, default="./pdf")
    parser.add_argument('--image_dir', type=str, default="./imgs")

    # Param for text detection onnx
    parser.add_argument('--det_onnx_path', type=str, default='./onnx/detection.onnx')
    
    # Param for ocr onnx
    parser.add_argument('--model_arch', type=str, default="vgg_transformer")
    parser.add_argument('--encoder_onnx_path', type=str, default='./onnx/ocr_encoder.onnx')
    parser.add_argument('--decoder_onnx_path', type=str, default='./onnx/ocr_decoder.onnx')
    
    # score to save value
    parser.add_argument('--drop_score', type=float, default=0.85)
    
    
    return parser.parse_args()

def get_default_config(args):
    return vars(args)


def create_annotations(img, dtboxes, rec_res, img_name):
    
    # path.split('/')[-1]
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
                    txt[0]
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
            'ocr': f'/data/local-files/?d=imgs/{img_name}'
        },
        'predictions': [{
            'result': results
        }]
    }

