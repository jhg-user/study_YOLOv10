import glob
import os
import torch
import cv2
from ultralytics import YOLOv10

# 모델 경로
best_saved_model_path = '/home/hkj/yolov10pj/yolov10_human/trainresult/result_facedetect7/weights/best.pt' # class 2개 학습된 모델 경로

# 모델 로드
model = YOLOv10(best_saved_model_path)

# 테스트 이미지 경로 설정
test_images_path = '/home/hkj/yolov10pj/yolov10_human/dataset/test/*.jpg'

# 결과 저장 폴더 설정
output_dir = 'runs/result_large'
os.makedirs(output_dir, exist_ok=True)

# 모든 테스트 이미지 경로 가져오기
image_paths = glob.glob(test_images_path)

# 이미지 한 개씩 예측 수행
for img_path in image_paths:
    # 이미지 읽기
    image = cv2.imread(img_path)
    height, width, _ = image.shape
    file_size = os.path.getsize(img_path) / 1024  # KB 단위

    # 이미지 정보 출력
    print(f"Image path: {img_path}")
    print(f"Image size: {width}x{height}")
    print(f"File size: {file_size:.2f} KB")

    images = [img_path]  # 리스트로 감싸서 배치 형태로 만듦

    # YOLOv10 모델 예측
    with torch.no_grad():
        try:
            results = model.predict(images, save=True, imgsz=640, conf=0.5, device=1)
            #results = model.predict(images, save=True, imgsz=640, conf=0.5, device='cpu')
            print(f"Image processed: {img_path}")
        except RuntimeError as e:
            print(f"RuntimeError occurred: {e}")
            print("Switching to CPU for this image.")
            #results = model.predict(images, save=True, imgsz=640, conf=0.5, device='cpu')

    # 메모리 클리어
    torch.cuda.empty_cache()

print("Predict images saved successfully")

