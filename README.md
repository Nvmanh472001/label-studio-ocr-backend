# Prelabling for OCR task

## Environment preparation

```bash
    pip3 install -rq requirements.txt
```

## Model conversion

### Paddle model download

```bash
wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar
cd ./inference && tar xf en_PP-OCRv3_det_infer.tar && cd ..
```

### Paddle Convert ONNX

```bash
paddle2onnx --model_dir ./inference/en_PP-OCRv3_det_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./onnx/detection.onnx \
--opset_version 12 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True
```

```bash
rm -rf ./inference
```

## VietOCR Convert ONNX

```bash
python3 txt_rec_onnx.py
```

## Run pipline

```bash
python3 pre_labeling.py
```

## How to customzie source

Customize `init_args` function in `utility_func.py` file with other component you have!
