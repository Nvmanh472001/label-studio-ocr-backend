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
    parser.add_argument('--pdf_dir', type=str, default="./pdf")
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