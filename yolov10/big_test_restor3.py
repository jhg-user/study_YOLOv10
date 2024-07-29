import cv2
import numpy as np
from ultralytics import YOLOv10
from PIL import Image

# 모델 경로
best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect13/weights/best.pt'

# 모델 로드
model = YOLOv10(best_saved_model_path)

# 이미지 경로
image_path = '/home/hkj/yolov10pj/yolov10/dataset/bigface/bigface.jpg'

# 이미지 불러오기
image = cv2.imread(image_path)

# 축소할 비율 설정
scale_percent = 0.5  # 예: 원본의 절반 크기로 축소

# 원본 이미지의 가로 세로 길이
original_height, original_width = image.shape[:2]

# 축소할 사이즈 계산
new_width = int(original_width * scale_percent)
new_height = int(original_height * scale_percent)

# 작은 크기로 이미지 리사이즈
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
#resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

# YOLOv10 모델에서 사용할 이미지 크기 설정 (imgsz)
imgsz = (new_width, new_height)

# 예측을 위해 RGB 형식으로 변환 (OpenCV는 BGR 형식으로 이미지를 읽어오므로)
rgb_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# YOLOv10 모델 예측
results = model.predict(rgb_resized_image, save=True, imgsz=imgsz, conf=0.5, device=0)
# 예시로 주어진 정보에서 boxes 객체 추출
print(results[0].boxes)

boxes_tensor = results.tensor('xyxy')  # xyxy 속성을 tensor로 가져오기

# 감지된 객체의 bounding box 좌표를 출력
xmin, ymin, xmax, ymax = boxes_tensor
print(f"Bounding box coordinates: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

# 결과 리스트를 반복하여 처리합니다
output_boxes = []
for result in results:
    boxes_tensor = result.tensor('xyxy')  # 각 결과 객체에서 tensor 메서드 호출
    
    # 원본 이미지의 크기를 가져옵니다
    original_height, original_width = original_image.shape[:2]  # original_image는 원본 이미지입니다.
    
    # 감지된 객체의 각 bounding box에 대해 반복합니다
    for xmin, ymin, xmax, ymax, conf, cls_conf in boxes_tensor:
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        
        # YOLOv10 형식의 상대적인 좌표를 절대적인 좌표로 변환합니다
        xmin_abs = int(xmin * original_width)
        ymin_abs = int(ymin * original_height)
        xmax_abs = int(xmax * original_width)
        ymax_abs = int(ymax * original_height)
        
        # 변환된 좌표를 output_boxes에 추가합니다
        output_boxes.append([xmin_abs, ymin_abs, xmax_abs, ymax_abs])
        
# output_boxes를 파일에 저장합니다
with open('output_boxes.txt', 'w') as f:
    for box in output_boxes:
        xmin, ymin, xmax, ymax = box
        f.write(f'0 {xmin} {ymin} {xmax} {ymax}\n')  # 클래스 ID가 0으로 가정하여 저장

