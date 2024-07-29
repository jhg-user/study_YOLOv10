from PIL import Image
import os

def fix_image_format(image_path):
    with Image.open(image_path) as img:
        if img.mode in ("P", "LA", "PA"):
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")
        img.save(image_path, format='JPEG')

def fix_images_in_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                fix_image_format(file_path)

if __name__ == "__main__":
    dataset_directory = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/dataset/train/non-human-face'
    fix_images_in_directory(dataset_directory)

