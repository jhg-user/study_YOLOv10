import os
import shutil
import random

# 원본 파일이 있는 디렉토리
src_directory = "/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/crop_human_images"

# 목적지 디렉토리
train_directory = "/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/dataset/train/human-face"
val_directory = "/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/dataset/val/human-face"

# 디렉토리 생성 (이미 존재할 경우 무시)
os.makedirs(train_directory, exist_ok=True)
os.makedirs(val_directory, exist_ok=True)

# .jpg 파일 목록 가져오기
files = [os.path.join(src_directory, f) for f in os.listdir(src_directory) if f.endswith('.jpg')]
total_files = len(files)

# 파일 목록 무작위로 섞기
random.shuffle(files)

# 80%와 20% 계산
train_count = int(total_files * 0.8)

# 파일 옮기기
for i, file in enumerate(files):
    if i < train_count:
        shutil.move(file, train_directory)
    else:
        shutil.move(file, val_directory)

print("Files have been moved successfully.")

