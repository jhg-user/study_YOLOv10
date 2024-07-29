from PIL import Image
import os

def validate_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.load()  # 실제 이미지 데이터를 로드하여 파일 유효성 확인
        return True
    except (IOError, SyntaxError, OSError) as e:
        print(f"Invalid image {image_path}: {e}")
        return False

def check_and_clean_dataset(dataset_dir):
    invalid_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if not validate_image(file_path):
                invalid_files.append(file_path)

    for file_path in invalid_files:
        print(f"Removing invalid file: {file_path}")
        os.remove(file_path)  # 손상된 파일 삭제

if __name__ == "__main__":
    dataset_directory = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/dataset/'
    check_and_clean_dataset(dataset_directory)

