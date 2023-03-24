import json

with open('export_projeck.json') as f:
  data = json.load(f)

LABELS = ["person_name","dob_key","dob_value","gender_key","gender_value","phonenumber_key","phonenumber_value","email_key","email_value","address_key","address_value","socical_address_value","education","education_name","education_time","experience","experience_name","experience_time","information","undefined"]
result = []

for img in data:
  anotations = []
  original_width = 0
  original_height = 0

  for text, label in zip(img['transcription'],img['label']):
    x_box = round((label['x']*label['original_width'])/100, 1)
    y_box = round((label['y']*label['original_height'])/100,1)
    width_box = round((label['width']*label['original_width'])/100,1)
    height_box = round((label['height']*label['original_height'])/100,1)

    anotation = {
        "box": [x_box,y_box,
                x_box + width_box,y_box,
                x_box + width_box, y_box + height_box,
                x_box,y_box +height_box ],
        "text": text,
        "label": LABELS.index(label['labels'][0])
    }
    original_width = label['original_width']
    original_height = label['original_height']
    anotations.append(anotation)

  result.append({
      'file_name': img['ocr'],
      "height": original_height,
      "width": original_width,
      "annotations":  anotations
  })


with open('mm_format.json', mode='w') as f:
  json.dump(result, f, indent=2)
