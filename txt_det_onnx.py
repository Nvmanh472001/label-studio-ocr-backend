import os
import numpy as np
import onnxruntime as ort

from paddleocr.ppocr.data import create_operators, transform
from paddleocr.ppocr.postprocess import build_post_process
from paddleocr.ppocr.utils.utility import get_image_file_list


class DetectionONNX(object):
    def __init__(self, args) -> None:
        self.onnx_path = args.det_onnx_path
        
        if not os.path.exists(self.onnx_path):
            raise ValueError(f"Not find onnx file path {self.onnx_path}")
        
        self.predictor, self.input_tensor = self._create_predictor() 
        
        preprocess_ops = [{
            'DetResizeForTest': {
                'limit_side_len': 960,
                'det_limit_type': 'max',
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        self.preprocess_op = create_operators(preprocess_ops)
        
        postprocess_params = {
            "name": "DBPostProcess",
            "thresh": 0.3,
            "box_thresh": 0.6,
            "max_candidates": 1000,
            "unclip_ratio": 1.5,
        }
        self.postprocess_op = build_post_process(postprocess_params)
    
    
    def _create_predictor(self):
        import onnxruntime as ort
        model_file_path = self.onnx_path
        if not os.path.exists(model_file_path):
            raise ValueError("not find model file path {}".format(
                model_file_path))
        sess = ort.InferenceSession(model_file_path)
        return sess, sess.get_inputs()[0]
    
    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes
    
    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points
    
    def __call__(self, img):
        ori_image = img.copy()
        data = { "image": img }
        
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        input_dict = {}
        input_dict[self.input_tensor.name] = img
        outputs = self.predictor.run(None, input_dict)

        preds = {}
        preds["maps"] = outputs[0]

        post_result = self.postprocess_op(preds, shape_list)
        
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_image.shape)
        
        return dt_boxes
        
def main():
    import argparse
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_onnx_path", type=str, default="./weights/onnx/txt_detection.onnx")
    parser.add_argument("--image_dir", type=str, default="./imgs")
    
    args = parser.parse_args()
    text_det = DetectionONNX(args)
    
    image_file_list = get_image_file_list(args.image_dir)
    save_results = []
    for idx, image_file in enumerate(image_file_list):
        img = cv2.imread(image_file)
        dtboxes = text_det(img)
        save_results.append([coordinate.tolist() for coordinate in dtboxes])
        
    print(save_results)
    
if __name__ == "__main__":
    main()