import cv2
from ultralytics import YOLOv10

# 모델 경로
best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect13/weights/best.pt'

# 모델 로드
model = YOLOv10(best_saved_model_path)

# 이미지 경로
image_path = '/home/hkj/yolov10pj/yolov10/dataset/bigface/bigface.jpg'

# 이미지를 RGB로 읽어오기
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지 흑백으로 읽어오기
#gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 현재 이미지 크기 얻기
#height, width = gray_image.shape[:2]

# 현재 이미지 크기 얻기
height, width = rgb_image.shape[:2]

# 새로운 크기 계산 (가로와 세로 각각 1/2로 줄이기)
new_width = width // 2
new_height = height // 2

# YOLOv10 모델에서 사용할 이미지 크기 설정 (imgsz)
imgsz = (new_width, new_height)

# 이미지 리사이즈 (흑백 이미지를 RGB로 변환)
#resized_image = cv2.resize(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB), imgsz, interpolation=cv2.INTER_AREA)

# 흑백 이미지를 RGB 형식으로 변환
#rgb_resized_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

# 이미지 리사이즈
#resized_image = cv2.resize(rgb_resized_image, imgsz, interpolation=cv2.INTER_AREA)
resized_image = cv2.resize(rgb_image, imgsz, interpolation=cv2.INTER_AREA)

# YOLOv10 모델 예측 및 결과 저장
results = model.predict(resized_image, save=True, imgsz=imgsz, conf=0.5, device=0)

# 예측 결과 파일 경로 출력
print(f"Predictions saved to: {results}")

# 예측 결과를 저장하면 해당 경로에 이미지와 바운딩 박스 정보가 포함된 파일이 생성됩니다.

