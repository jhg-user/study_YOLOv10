import cv2
import torch
from ultralytics import YOLOv10

#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect3/weights/best.pt' # s
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect6/weights/best.pt' # n
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect9/weights/best.pt' # m
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect10/weights/best.pt' # b
best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect13/weights/best.pt' # l
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect14/weights/best.pt' # x

# 모델 로드
model = YOLOv10(best_saved_model_path)

# 이미지 불러오기
image_path = '/home/hkj/yolov10pj/yolov10/dataset/bigface/bigface.jpg'
image = cv2.imread(image_path)

# 현재 이미지 크기 얻기
height, width = image.shape[:2]

# 새로운 크기 계산 (가로, 세로 각각 절반으로 줄이기)
new_width = width // 2
new_height = height // 2

# 이미지 리사이즈
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# 리사이즈된 이미지를 YOLO 모델에 입력
# OpenCV는 BGR 형식으로 이미지를 불러오기 때문에 RGB로 변환 필요
rgb_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# 예측 수행
model.predict(rgb_resized_image, save=True, conf=0.5, device=0)


# 또는 결과를 저장
#results.save(save_dir='output_directory')  # 결과를 저장할 디렉토리 지정

