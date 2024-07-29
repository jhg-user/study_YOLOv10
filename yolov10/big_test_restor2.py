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

# 이미지 불러오기 (PIL.Image로 열기)
image = Image.open(image_path).convert('RGB')

# 현재 이미지 크기 얻기
original_width, original_height = image.size

# 새로운 크기 계산 (가로와 세로 각각 1/2로 줄이기)
new_width = original_width // 2
new_height = original_height // 2

# 작은 크기로 이미지 리사이즈
resized_image = image.resize((new_width, new_height), Image.LANCZOS)

# YOLOv10 모델에서 사용할 이미지 크기 설정 (imgsz)
imgsz = (new_width, new_height)

# 예측을 위해 numpy 배열로 변환 (PIL.Image를 numpy 배열로 변환)
rgb_resized_image = np.array(resized_image)  # PIL.Image를 numpy 배열로 변환
#rgb_resized_image = rgb_resized_image[:, :, ::-1].copy()  # RGB 형식으로 변환 (BGR -> RGB)
rgb_resized_image = cv2.cvtColor(rgb_resized_image, cv2.COLOR_RGB2BGR)  # PIL.Image에서 OpenCV BGR로 변환

# YOLOv10 모델 예측
results = model.predict(rgb_resized_image, imgsz=imgsz, conf=0.5, device=0)


# 모든 예측 결과 가져오기
for result in results:
    predictions = result.pred[0]  # 첫 번째 이미지의 예측 결과

    # 바운딩 박스를 원래 이미지의 크기로 변환하여 그리기
    scale_x = original_width / new_width
    scale_y = original_height / new_height

    # 원본 이미지 복사
    output_image = np.array(image)

    for pred in predictions:
        x1, y1, x2, y2, conf, cls = map(int, pred[:6])  # 바운딩 박스 좌표, 신뢰도, 클래스
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # 원본 이미지에 바운딩 박스 그리기
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 클래스와 신뢰도 표시 (옵션)
        label = f"{cls}: {conf:.2f}"
        cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 이미지 저장
    output_image_path = 'dataset/bigface/result_image.jpg'
    cv2.imwrite(output_image_path, output_image)

    print(f"Predictions saved to: {output_image_path}")
