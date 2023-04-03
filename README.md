# Prelabling for OCR task

## Environment preparation

```bash
sudo apt-get install poppler-utils
```

```bash
pip3 install -r requirements.txt
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

### VietOCR Convert ONNX

```bash
python3 txt_rec_onnx.py
```

## Run pipline

```bash
python3 pre_labeling.py
```

## How to customzie source

Customize `init_args` function in `utility_func.py` file with other component you have!

## How to label

### install tool 
```python
pip install label-studio
```

### Set env variabel

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/yourname/nvmanh-git/label-studio-ocr-backend
```
### Set local storage

1. run label studio on http://localhost:8080/
```python
label-studio
```
---
2. creat project OCR and add list of label in file [label.txt](label.txt)

---
3. open Settings > Cloud Storage.
- **Storage Type -> select Local files**
![image](label-studio.PNG)

- **Storage title**: ImageData
- **Storage path**: /home/yourname/nvmanh-git/label-studio-ocr-backend/imgs

-> save
### Import file json and start labeling


