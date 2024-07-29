import cv2
from ultralytics import YOLOv10

# 모델 경로
best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect13/weights/best.pt'

# 모델 로드
model = YOLOv10(best_saved_model_path)

# 이미지 경로
image_path = '/home/hkj/yolov10pj/yolov10/dataset/bigface/bigface.jpg'

# 이미지 불러오기
image = cv2.imread(image_path)

# 현재 이미지 크기 얻기
height, width = image.shape[:2]

# 새로운 크기 계산 (가로와 세로 각각 1/2로 줄이기)
new_width = width // 2
new_height = height // 2

# YOLOv10 모델에서 사용할 이미지 크기 설정 (imgsz)
imgsz = (new_width, new_height)

# 이미지 리사이즈
resized_image = cv2.resize(image, imgsz, interpolation=cv2.INTER_AREA)

# 예측을 위해 RGB 형식으로 변환 (OpenCV는 BGR 형식으로 이미지를 읽어오므로)
rgb_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
#rgb_resized_image = resized_image[..., ::-1]

# YOLOv10 모델 예측
results = model.predict(rgb_resized_image, save=True, imgsz=imgsz, conf=0.5, device=0)

