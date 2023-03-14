import os
import tempfile
from pdf2image import convert_from_path
from PIL import Image

"""
    Convert PDF file with multi pages to one Image
"""
def pdf2Image(pdf_dir, image_dir):
    list_pdf = os.listdir(pdf_dir)
    list_pdf = [os.path.join(pdf_dir, fname) for fname in list_pdf if fname.endswith('pdf')]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, pdf_file in enumerate(list_pdf):
            images = convert_from_path(pdf_file, output_folder=temp_dir)
    
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

            merged_image.save(f"{image_dir}/{idx}.jpg")
    
    return image_dir