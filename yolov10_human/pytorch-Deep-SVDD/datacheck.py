import os
from PIL import Image

def check_image_file(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # 이미지 파일의 무결성 확인
    except (IOError, SyntaxError) as e:
        print(f"Error with image {image_path}: {e}")
        return False
    return True

def validate_dataset(dataset_dir, log_file='datasetcheck_log.txt'):
    valid_files = []
    with open(log_file, 'w') as log:
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if check_image_file(file_path):
                    valid_files.append(file_path)
                else:
                    log.write(f"Invalid file: {file_path}\n")
                    os.remove(file_path)  # 손상된 파일 삭제
    return valid_files

# 데이터셋 디렉토리 설정
dataset_directory = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/dataset'  # 실제 경로로 변경
valid_image_paths = validate_dataset(dataset_directory)

