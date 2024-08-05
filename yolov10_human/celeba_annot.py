import glob
import cv2
import os
import torch
from ultralytics import YOLOv10

def get_image_size(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    
    # 현재 이미지 크기 얻기
    original_height, original_width = image.shape[:2]
    
    return original_height, original_width

# 모델 경로
best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect13/weights/best.pt' # class 1개 학습된 l
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10_human/trainresult/result_facedetect4/weights/best.pt' # class 2개 학습된 s

# 모델 로드
model = YOLOv10(best_saved_model_path)

# 테스트 이미지 경로 설정
test_images_path = '/home/hkj/yolov10pj/yolov10_human/dataset/celeba_preannot/*.jpg'

# 결과 저장 폴더 설정
output_dir = 'runs/predict_celebA'
os.makedirs(output_dir, exist_ok=True)

# 배치 크기 설정
batch_size = 2  # 한 번에 처리할 이미지 수

# 모든 테스트 이미지 경로 가져오기
image_paths = glob.glob(test_images_path)

# 배치 단위로 예측 수행
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    original_dims = []

    # 배치 내 각 이미지 처리
    for img_path in batch_paths:
        original_height, original_width = get_image_size(img_path)
        original_dims.append((original_height, original_width))

    # YOLOv10 모델 예측
    with torch.no_grad():
        try:
            results = model.predict(batch_paths, save=True, imgsz=640, conf=0.5, device=0)
        except RuntimeError as e:
            print(f"RuntimeError occurred: {e}")
            print("Switching to CPU for this batch.")
            results = model.predict(batch_paths, save=True, imgsz=640, conf=0.5, device='cpu')

    # 메모리 클리어
    torch.cuda.empty_cache()

    # 결과 저장
    for img_path, result, (original_height, original_width) in zip(batch_paths, results, original_dims):
        boxes = result.boxes
        resized_height, resized_width = result.orig_shape[:2]  # 예측 수행 시의 이미지 크기
        annot_file = img_path.split('/')[-1].replace('jpg', 'txt')
        annot_file = os.path.join(output_dir, annot_file)
        with open(annot_file, 'w') as f:
            for box in boxes:
                x, y, w, h = box.xywhn[0].tolist()
                score = box.conf[0].item()
                class_id = box.cls[0].item()

                # 리사이즈된 이미지 크기를 원본 이미지 크기로 변환
                x_center = (x * resized_width) / original_width
                y_center = (y * resized_height) / original_height
                width = (w * resized_width) / original_width
                height = (h * resized_height) / original_height

                class_id = int(class_id)

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        print(f"Annotations saved for {img_path}")

print("Bounding boxes annotations saved successfully")

