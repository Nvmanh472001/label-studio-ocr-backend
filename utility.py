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
    parser.add_argument('--pdf_dir', type=str)
    parser.add_argument('--image_dir', type=str)

    # Param for server ml backend
    parser.add_argument(
        '-p', '--port', dest='port', type=int, default=9090,
        help='Server port')
    parser.add_argument(
        '--host', dest='host', type=str, default='0.0.0.0',
        help='Server host')
    parser.add_argument(
        '--kwargs', '--with', dest='kwargs', metavar='KEY=VAL', nargs='+', type=lambda kv: kv.split('='),
        help='Additional LabelStudioMLBase model initialization kwargs')
    parser.add_argument(
        '-d', '--debug', dest='debug', action='store_true',
        help='Switch debug mode')
    parser.add_argument(
        '--log-level', dest='log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=None,
        help='Logging level')
    
    return parser

def parse_args():
    parser = init_args()
    return parser.parse_args()

def get_default_config(args):
    return vars(args)


def is_image(path):
    end_img = ["jpeg", "jpg", "png"]
    return any([path.lower().endswith(e) for e in end_img])

def is_pdf(path):
    return path.lower().endswith("pdf")
    
    
    
    