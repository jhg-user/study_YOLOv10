import cv2
from ultralytics import YOLOv10

# 모델 경로
best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect13/weights/best.pt'

# 모델 로드
model = YOLOv10(best_saved_model_path)

# 이미지 경로
image_path = 'dataset/bigface/bigface.jpg'

# 이미지 불러오기
image = cv2.imread(image_path)

# 현재 이미지 크기 얻기
original_height, original_width = image.shape[:2]

# 새로운 크기 계산 (가로와 세로 각각 1/2로 줄이기)
new_width = original_width // 2
new_height = original_height // 2

# YOLOv10 모델에서 사용할 이미지 크기 설정 (imgsz)
imgsz = (new_width, new_height)

# 이미지 리사이즈
resized_image = cv2.resize(image, imgsz, interpolation=cv2.INTER_AREA)

# 예측을 위해 RGB 형식으로 변환 (OpenCV는 BGR 형식으로 이미지를 읽어오므로)
rgb_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# YOLOv10 모델 예측
results = model.predict(rgb_resized_image, save=True, imgsz=imgsz, conf=0.5, device=0)

# 결과에서 바운딩 박스 좌표 가져오기
boxes = results[0].boxes

# 바운딩 박스 좌표를 정규화하여 .txt 파일로 저장
with open('dataset/bigface/predict.txt', 'w') as f:
    for box in boxes:
        x, y, w, h = box.xywhn[0].tolist()  # 좌표를 리스트로 변환하여 가져오기
        score = box.conf[0].item()  # 텐서를 Python 숫자로 변환
        class_id = box.cls[0].item()  # 텐서를 Python 숫자로 변환

        # 바운딩 박스를 원본 이미지 크기에 맞게 확장
        x1 = (x - w/2) * original_width
        y1 = (y - h/2) * original_height
        x2 = (x + w/2) * original_width
        y2 = (y + h/2) * original_height

        # 정규화된 좌표값으로 변환하여 파일에 저장
        x_center = (x1 + x2) / (2 * original_width)
        y_center = (y1 + y2) / (2 * original_height)
        width = (x2 - x1) / original_width
        height = (y2 - y1) / original_height
        class_id = int(class_id)

        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        
        # 좌표를 원본 이미지 크기에 맞게 스케일링
        #x1 = x1 * original_width / new_width
        #y1 = y1 * original_height / new_height
        #w = w * original_width / new_width
        #h = h * original_height / new_height

        # Write to txt file in YOLO format
        #f.write(f"{class_id} {x1} {y1} {w} {h}\n")

# 예측 결과 이미지를 BGR 형식으로 변환하여 저장 (OpenCV는 BGR 형식을 사용)
output_image_path = 'dataset/bigface/predict.jpg'
cv2.imwrite(output_image_path, cv2.cvtColor(rgb_resized_image, cv2.COLOR_RGB2BGR))

print(f"Bounding boxes annotations saved success") 

