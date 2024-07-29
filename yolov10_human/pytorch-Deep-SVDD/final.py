import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image
import os
from ultralytics import YOLOv10

# Deep SVDD 네트워크 정의
class Network(nn.Module):
    def __init__(self, z_dim=32):
        super(Network, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

# Autoencoder 정의
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

        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
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
        x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)
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

# AnomalyDetector 정의
class AnomalyDetector:
    def __init__(self, autoencoder, center, threshold=0.05):
        self.autoencoder = autoencoder
        self.center = center
        self.device = next(autoencoder.parameters()).device  # 모델의 장치 가져오기
        #self.threshold = threshold
        #self.device = autoencoder.device

    def is_anomalous(self, image):
        self.autoencoder.eval()
        with torch.no_grad():
            image = transform(image).unsqueeze(0).to(self.device)
            reconstructed = self.autoencoder(image)
            z = self.autoencoder.encode(image)
            loss = torch.mean(torch.sum((z - self.center) ** 2, dim=1))
        return loss.item() > self.threshold

# 모델 및 하이퍼스피어 중심 로딩
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Autoencoder 모델 로드
autoencoder = Autoencoder().to(device)
autoencoder.load_state_dict(torch.load('autoencoder_human_face.pth', map_location=device))

# Deep SVDD 하이퍼스피어 중심 로드
checkpoint = torch.load('weights/pretrained_parameters.pth', map_location=device)
center = torch.Tensor(checkpoint['center']).to(device)

# AnomalyDetector 인스턴스 생성
anomaly_detector = AnomalyDetector(autoencoder, center)

# 이미지 전처리 정의
transform = T.Compose([
    T.Grayscale(),  # 얼굴 이미지는 흑백으로 변환
    T.Resize((28, 28)),  # Autoencoder에 맞게 크기 조정
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))  # 정규화 (예시, 필요에 따라 조정)
])

# 예측 및 결과 저장
def predict_and_save(image_path, save_directory):
    # YOLOv10 모델 로드
    model = YOLOv10('/home/hkj/yolov10pj/yolov10_human/runs/detect/train4/weights/best.pt')
    
    # 결과 저장 디렉토리 확인 및 생성
    os.makedirs(save_directory, exist_ok=True)
    
    # 예측 수행 및 결과 저장
    results = model.predict(source=image_path, save=True, imgsz=640, conf=0.5, device=0, save_dir=save_directory)
    
    # 저장된 결과 이미지 파일명 확인
    for result_file in results.files:
        print(f"Result image saved at: {result_file}")
    
    # 결과 이미지 경로를 사용하여 얼굴 감지 및 이상 탐지 수행
    for result_file in results.files:
        result_image = Image.open(result_file)
        detections = results.pandas().xyxy[0]  # bounding boxes in Pandas DataFrame format
        
        anomaly_results = []
        for _, row in detections.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            cropped_face = result_image.crop((x1, y1, x2, y2))
            
            if anomaly_detector.is_anomalous(cropped_face):
                label = "Anomalous Face"
            else:
                label = "Normal Face"
            
            anomaly_results.append({'bbox': [x1, y1, x2, y2], 'label': label, 'score': row['confidence']})
        
        # Print anomaly detection results for each image
        for result in anomaly_results:
            print(f"Bounding Box: {result['bbox']}, Label: {result['label']}, Score: {result['score']}")

# 사용 예시
test_images_path = '/home/hkj/yolov10pj/yolov10_human/dataset/test2/*.jpg'
save_directory = '/home/hkj/yolov10pj/yolov10_human/runs/detect/deep_result'

predict_and_save(test_images_path, save_directory)

