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
#image = Image.open(image_path).convert('RGB')

# 현재 이미지 크기 얻기
#original_height, original_width = image.shape[:2]
#original_width, original_height = image.size

# 새로운 크기 계산 (가로와 세로 각각 1/2로 줄이기)
#new_width = original_width // 2
#new_height = original_height // 2

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

# bounding box 정보 조정 및 저장할 리스트 생성
output_boxes = []

x_center, y_center, width, height = boxes_tensor

# YOLOv10 형식에서 좌표를 계산합니다
xmin = int((x_center - width / 2) * orig_width)
ymin = int((y_center - height / 2) * orig_height)
xmax = int((x_center + width / 2) * orig_width)
ymax = int((y_center + height / 2) * orig_height)

# 결과 리스트에 추가
output_boxes.append([xmin, ymin, xmax, ymax])
#for box in results:
#    x_center, y_center, width, height = boxes_tensor 

    # YOLOv10 형식에서 좌표를 계산합니다
#    xmin = int((x_center - width / 2) * orig_width)
#    ymin = int((y_center - height / 2) * orig_height)
#    xmax = int((x_center + width / 2) * orig_width)
#    ymax = int((y_center + height / 2) * orig_height)

    # 결과 리스트에 추가
#    output_boxes.append([xmin, ymin, xmax, ymax])

# 결과 리스트를 파일에 저장합니다 (예시)
with open('output_boxes.txt', 'w') as f:
    for box in output_boxes:
        xmin, ymin, xmax, ymax = box
        f.write(f'0 {xmin} {ymin} {xmax} {ymax}\n')  # 클래스 ID가 0으로 가정하여 저장

