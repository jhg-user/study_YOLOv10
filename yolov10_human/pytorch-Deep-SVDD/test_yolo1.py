import glob
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLOv10
from model import network
import os

# 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 경로 설정
svdd_model_path = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/weights/final_deep_svdd.pth'
test_images_path = '/home/hkj/yolov10pj/yolov10_human/dataset/test2/*.jpg'  # 또는 다른 경로
best_saved_model_path = '/home/hkj/yolov10pj/yolov10_human/trainresult/result_facedetect4/weights/best.pt'
output_img_dir = 'runs/predict_deep'  # 예측 결과 이미지를 저장할 디렉터리
os.makedirs(output_img_dir, exist_ok=True)

# 모델 로드
yolo_model = YOLOv10(best_saved_model_path)
net = network().to(device)
state_dict = torch.load(svdd_model_path)
net.load_state_dict(state_dict['net_dict'])
c = torch.tensor(state_dict['center']).to(device)
net.eval()

def anomalous_detection(image_tensor):
    """Deep SVDD 모델을 사용하여 이상 점수 계산"""
    with torch.no_grad():
        z = net(image_tensor)
        score = torch.sum((z - c) ** 2, dim=1)
        return score

def add_text_to_image(img, text, position, font_size=24):
    """이미지에 텍스트 추가"""
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text(position, text, fill='red', font=font)
    return img

def process_image(image_path):
    """이미지를 처리하고 YOLOv10 예측 및 Deep SVDD 이상 탐지를 수행합니다."""
    # YOLOv10 예측
    results = yolo_model.predict(source=image_path,save=True, imgsz=640, conf=0.5, device=device)
    
    # 결과 이미지 저장 경로 가져오기
    result_image_path = results.save_paths[0]
    print(f"Saved result image to {result_image_path}")

    # 결과 이미지 열기
    img = Image.open(image_path)
    detections = results.boxes.xyxy[0]

    anomaly_scores = []

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        
        # 객체 이미지 잘라내기
        object_image = img.crop((x1, y1, x2, y2))

        # Deep SVDD 모델에 맞는 전처리 수행
        object_image = object_image.convert('L')  # 변환할 채널 수에 맞게 (흑백으로 변환)
        object_image = object_image.resize((28, 28))  # Deep SVDD 모델에 맞는 사이즈로 조정
        object_image = np.array(object_image)
        object_image = torch.tensor(object_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # 배치 차원 및 채널 차원 추가

        # 이상 점수 계산
        score = anomalous_detection(object_image)
        anomaly_scores.append(score.item())

        # 라벨 추가
        label = 'Anomalous' if score.item() > 0.5 else 'Normal'  # 점수에 따라 라벨 결정
        img = add_text_to_image(img, label, (x1, y1), font_size=24)
        
        print(f'Image: {image_path}')
        print(f'Bounding box: ({x1}, {y1}, {x2}, {y2})')
        print(f'Anomaly score: {score.item()}')

    # 결과 이미지 저장
    img.save(os.path.join(output_img_dir, os.path.basename(image_path)))
    print(f"Annotated image saved to {os.path.join(output_img_dir, os.path.basename(image_path))}")

# 테스트 이미지 경로에서 이미지 파일을 로드하고 처리합니다.
image_paths = glob.glob(test_images_path)
for image_path in image_paths:
    process_image(image_path)
    print("image predict success")

print("Anomaly scores calculated and results saved successfully")

