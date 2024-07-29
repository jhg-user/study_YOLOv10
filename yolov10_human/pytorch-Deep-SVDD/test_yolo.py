import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLOv10
import numpy as np
import glob
import os
from model import network  # 또는 autoencoder, 선택에 따라

# 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 경로 설정
svdd_model_path = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/weights/final_deep_svdd.pth'
test_images_path = '/home/hkj/yolov10pj/yolov10_human/dataset/test2/printingtshirt5.jpg'
best_saved_model_path = '/home/hkj/yolov10pj/yolov10_human/trainresult/result_facedetect4/weights/best.pt'

# YOLOv10 모델 로드
model = YOLOv10(best_saved_model_path)

# Deep SVDD 모델 로드
net = network().to(device)  # 또는 autoencoder()로 교체
state_dict = torch.load(svdd_model_path)
net.load_state_dict(state_dict['net_dict'])
c = torch.tensor(state_dict['center']).to(device)
net.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Deep SVDD 모델의 입력 사이즈에 맞추기
    transforms.Grayscale(num_output_channels=1),  # Deep SVDD 모델에 맞는 채널 수
    transforms.ToTensor(),
])

def anomalous_detection(image_tensor):
    """Deep SVDD 모델을 사용하여 이상 점수 계산"""
    with torch.no_grad():
        z = net(image_tensor)
        score = torch.sum((z - c) ** 2, dim=1)
        return score

def process_image(image_path):
    """이미지를 로드하고 전처리합니다."""
    # YOLOv10 예측
    results = model.predict(image_path, imgsz=640, conf=0.5, device=0)
    img = Image.open(image_path)
    detections = results.pandas().xyxy[0]

    # 인식된 객체들에 대해 이상 탐지 수행
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        object_image = img.crop((x1, y1, x2, y2))
        object_image = transform(object_image).unsqueeze(0).to(device)

        # 이상 탐지
        score = anomalous_detection(object_image)
        print(f'Image: {image_path}')
        print(f'Bounding box: ({x1}, {y1}, {x2}, {y2})')
        print(f'Anomaly score: {score.item()}')

# 테스트 이미지 경로에서 이미지 파일을 로드하고 처리합니다.
image_paths = glob.glob(test_images_path)
for image_path in image_paths:
    process_image(image_path)

