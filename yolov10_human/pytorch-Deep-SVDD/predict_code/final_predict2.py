import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from ultralytics import YOLOv10
from torchvision import transforms
from model import network
from test import eval
from collections import OrderedDict


# YOLO 모델 로딩
def load_yolo_model(model_path):
    model = YOLOv10(model_path)
    return model


# 새로운 autoencoder 클래스 정의
class Autoencoder(nn.Module):
    def __init__(self, z_dim=32):
        super(Autoencoder, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

        self.fc2 = nn.Linear(z_dim, 4 * 7 * 7, bias=False)
        self.deconv1 = nn.ConvTranspose2d(4, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def encode(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

    def decode(self, x):
        x = self.fc2(x)
        x = x.view(x.size(0), 4, 7, 7)  # [batch_size, channels, height, width]
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        return torch.sigmoid(x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


# YOLO 모델을 사용하여 이미지에서 객체를 탐지
def predict_yolo(model, image_path):
    #image = cv2.imread(image_path)
    #results = model(image)
    results = model.predict(image_path, save=True, imgsz=640, conf=0.5, device=0)
    #return results.boxes.xyxy[0]  # 결과 반환
    return results
'''

def load_svdd_model(path):
    model = Autoencoder(z_dim=32)
    checkpoint = torch.load(path)
    # state_dict의 키 확인
    print(checkpoint.keys())
    net_dict = checkpoint['net_dict']  # 필요한 키만 추출
    new_state_dict = OrderedDict()

    for k, v in net_dict.items():
        name = k.replace('module.', '')  # 키 이름 변경
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model
'''

# 기존 모델을 로드하는 함수
def load_svdd_model(path):
    model = Autoencoder(z_dim=32)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def compute_reconstruction_error(boxes, model, image_path):
    errors = []
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 모델 입력 크기에 맞게 조정
        transforms.ToTensor(),
        transforms.Grayscale(),  # 모델이 그레이스케일 입력을 기대하는 경우
    ])

    for box in boxes:
        # box 좌표 추출
        xyxy = box.xyxy[0].tolist()
        left, top, right, bottom = map(int, xyxy[:4])

        # 이미지를 crop하고 Tensor로 변환
        cropped_image = image.crop((left, top, right, bottom))
        data = transform(cropped_image).unsqueeze(0)  # 배치 차원 추가

        # 모델을 사용하여 재구성
        reconstructions = model(data)

        # 재구성 오차 계산 (예: MSE)
        error = torch.nn.functional.mse_loss(reconstructions, data)
        errors.append(error.item())

    return errors

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
    svdd_model = load_svdd_model(svdd_model_path)

    # 재구성 오차 계산
    errors = compute_reconstruction_error(boxes, svdd_model, image_path)

    # 이상 탐지 임계값 설정
    threshold = 0.5  # 이 값은 모델의 특성에 맞게 조정해야 합니다.

    # 이상 탐지 결과
    anomalies = anomaly_detection(errors, threshold)
    print(f"Anomalies: {anomalies}")

if __name__ == "__main__":
    main()

