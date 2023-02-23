from easyocr import Reader
from PIL import Image
import random as rd

ocr = Reader(lang_list=["vi"], detector=True, recognizer=True, download_enabled=True, gpu=False)
model_results = ocr.readtext("data/imgs/2.jpg")

img_width, img_height = Image.open('data/imgs/2.jpg').size
results = []
score_thresh = 0.3
all_scores = []

print(model_results)

for poly in model_results:
    output_label = 'Text'
    if not poly:
        continue
    score = poly[-1]
    if score < score_thresh:
        continue

    rel_pnt = []
    for rp in poly[0]:
        if rp[0] > img_width or rp[1] > img_height:
                continue
        print(rp)
        rel_pnt.append([(rp[0] / img_width) * 100, (rp[1] / img_height) * 100])

        # must add one for the rectangle
        id_gen = rd.randrange(10**10)
        results.append({
            'original_width': img_width,
            'original_height': img_height,
            'image_rotation': 0,
            'value': {
                'points': rel_pnt,
            },
            'id': id_gen,
            'from_name': "polygon",
            'to_name': 'image',
            'type': 'polygon',
            'origin': 'manual',
            'score': score,
        })
        
        # for the transcription
        results.append({
            'original_width': img_width,
            'original_height': img_height,
            'image_rotation': 0,
            'value': {
                'points': rel_pnt,
                'labels': [output_label],
                "text": [
                    poly[1]
                ]
            },
            'id': id_gen,
            'from_name': "transcription",
            'to_name': 'image',
            'type': 'textarea',
            'origin': 'manual',
            'score': score,
        })
        all_scores.append(score)
    avg_score = sum(all_scores) / max(len(all_scores), 1)

print(results[0])