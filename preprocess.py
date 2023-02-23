import os
import tempfile
from pdf2image import convert_from_path
from PIL import Image

"""
    Convert PDF file with multi pages to one Image
"""
def pdf2Image(file_path, output_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_path(file_path, output_folder=temp_dir)
    
        temp_images = []
        for i in range(len(images)):
            image_path = f'{temp_dir}/{i}.jpg'
            images[i].save(image_path, 'JPEG')
            temp_images.append(image_path)
            
        imgs = list(map(Image.open, temp_images))
        
    min_img_width = min(i.width for i in imgs)
    
    total_height = 0
    for i, img in enumerate(imgs):
        total_height += imgs[i].height
    
    merged_image = Image.new(imgs[0].mode, (min_img_width, total_height))
    
    y = 0
    for img in imgs:
        merged_image.paste(img, (0, y))
        y += img.height

    merged_image.save(output_path)
    return output_path