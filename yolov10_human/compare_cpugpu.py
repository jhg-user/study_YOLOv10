import glob
import cv2
import os
import torch
from ultralytics import YOLOv10

# 모델 경로
best_saved_model_path = '/home/hkj/yolov10pj/yolov10_human/trainresult/result_facedetect7/weights/best.pt' # class 2개 학습된 l

# 모델 로드
model = YOLOv10(best_saved_model_path)

# 테스트 이미지 경로 설정
test_images_path = '/home/hkj/yolov10pj/yolov10_human/dataset/test/*.jpg'  

# 결과 저장 폴더 설정
output_dir = 'runs/result_large'
os.makedirs(output_dir, exist_ok=True)

# 배치 크기 설정
batch_size = 2  # 한 번에 처리할 이미지 수

# 모든 테스트 이미지 경로 가져오기
image_paths = glob.glob(test_images_path)

# 배치 단위로 예측 수행
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    images = []

    # 배치 내 각 이미지 처리
    for img_path in batch_paths:
        images.append(img_path)

    # YOLOv10 모델 예측
    with torch.no_grad():
        try:
            #results = model.predict(images, save=True, imgsz=640, conf=0.5, device=1)
            results = model.predict(images, save=True, imgsz=640, conf=0.5, device='cpu')
            print(f"image : {images}")
        except RuntimeError as e:
            print(f"RuntimeError occurred: {e}")
            print("Switching to CPU for this batch.")
            #results = model.predict(images, save=True, imgsz=640, conf=0.5, device='cpu')

    # 메모리 클리어
    torch.cuda.empty_cache()

print("Predict images saved successfully")

