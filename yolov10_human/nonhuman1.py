import glob
import cv2
import os
import torch
from ultralytics import YOLOv10

# 모델 경로 설정
best_saved_model_path = '/home/hkj/yolov10pj/yolov10_human/trainresult/result_facedetect7/weights/best.pt'

# 모델 로드
model = YOLOv10(best_saved_model_path)

# 테스트 이미지 경로 설정
test_images_path = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/dataset/train/human-face/*jpg'

# 결과 저장 폴더 및 파일 설정
output_dir = 'nonhuman_list'
os.makedirs(output_dir, exist_ok=True)
nonhuman_list_file = os.path.join(output_dir, 'nonhuman_list2.txt')

# 배치 크기 설정
batch_size = 2  # 한 번에 처리할 이미지 수

# 모든 테스트 이미지 경로 가져오기
image_paths = glob.glob(test_images_path)

# 결과를 저장할 리스트
nonhuman_images = []

# 배치 단위로 예측 수행
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    
    # YOLOv10 모델 예측
    with torch.no_grad():
        try:
            results = model.predict(batch_paths, save=False, conf=0.5, device=1)
        except RuntimeError as e:
            print(f"RuntimeError occurred: {e}")
            print("Switching to CPU for this batch.")
            results = model.predict(batch_paths, save=False, conf=0.5, device='cpu')

    # 메모리 클리어
    torch.cuda.empty_cache()
    
    # 결과 처리
    for img_path, result in zip(batch_paths, results):
        # result.boxes contains detection results
        detections = result.boxes
        for box in detections:
            # box contains coordinates, confidence, and class
            class_id = int(box.cls)  # Get the class id of the detection
            if class_id == 1:  # 특정 클래스 (클래스 1) 체크
                nonhuman_images.append(img_path)
                break

# 결과를 파일에 저장
with open(nonhuman_list_file, 'w') as f:
    for img_path in nonhuman_images:
        f.write(f"{img_path}\n")

print(f"Predict images saved successfully to {nonhuman_list_file}")

