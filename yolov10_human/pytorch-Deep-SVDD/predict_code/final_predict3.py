import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from ultralytics import YOLOv10
from model import network
from collections import OrderedDict

# YOLO 모델 로딩
def load_yolo_model(model_path):
    model = YOLOv10(model_path)
    return model

# Deep SVDD 모델 로딩
def load_svdd_model(path):
    net = network(z_dim=32).to('cuda')
    checkpoint = torch.load(path)
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    net.eval()
    c = torch.tensor(checkpoint['center']).to('cuda')
    return net, c

# YOLO 모델을 사용하여 이미지에서 객체를 탐지
def predict_yolo(model, image_path):
    results = model.predict(image_path, save=True, imgsz=640, conf=0.5, device=0)
    return results

# 재구성 오차 계산 함수
def compute_reconstruction_error(boxes, model, image_path, c):
    errors = []
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 모델 입력 크기에 맞게 조정
        transforms.ToTensor(),
        transforms.Grayscale(),  # 모델이 그레이스케일 입력을 기대하는 경우
    ])

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        left, top, right, bottom = map(int, xyxy[:4])

        # 이미지를 crop하고 Tensor로 변환
        cropped_image = image.crop((left, top, right, bottom))
        data = transform(cropped_image).unsqueeze(0).to('cuda')  # 배치 차원 추가 및 GPU로 이동

        # 모델을 사용하여 특성 추출 및 재구성 오차 계산
        z = model(data)
        error = torch.mean(torch.sum((z - c) ** 2, dim=1))
        errors.append(error.item())

    return errors

# 이상 탐지 결과 계산 함수
def anomaly_detection(errors, threshold):
    anomalies = ["Anomalous" if error > threshold else "Normal" for error in errors]
    return anomalies

def main():
    yolo_model_path = '/home/hkj/yolov10pj/yolov10_human/runs/detect/train4/weights/best.pt'
    svdd_model_path = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/weights/final_deep_svdd.pth'
    image_path = '/home/hkj/yolov10pj/yolov10_human/dataset/test/printingtshirt5.jpg'

    # YOLO 모델 로딩 및 예측
    yolo_model = load_yolo_model(yolo_model_path)
    yolo_results = predict_yolo(yolo_model, image_path)

    boxes = yolo_results[0].boxes
    print(boxes)
    if len(boxes) == 0:
        print("No objects detected.")
        return

    # Deep SVDD 모델 로딩
    svdd_model, c = load_svdd_model(svdd_model_path)

    # 재구성 오차 계산
    errors = compute_reconstruction_error(boxes, svdd_model, image_path, c)

    # 이상 탐지 임계값 설정
    threshold = 0.5  # 이 값은 모델의 특성에 맞게 조정해야 합니다.

    # 이상 탐지 결과
    anomalies = anomaly_detection(errors, threshold)
    print(f"Anomalies: {anomalies}")

if __name__ == "__main__":
    main()

