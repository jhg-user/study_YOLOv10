import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from ultralytics import YOLOv10

# Autoencoder 정의
class Autoencoder(nn.Module):
    def __init__(self, z_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*16*16, z_dim)  # Assuming input image size is 64x64
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64*16*16),
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

# Deep SVDD 정의
class DeepSVDD(nn.Module):
    def __init__(self, autoencoder, center, radius):
        super(DeepSVDD, self).__init__()
        self.autoencoder = autoencoder
        self.center = center
        self.radius = radius
    
    def forward(self, encoded_data):
        distances = torch.norm(encoded_data - self.center, dim=1)
        return distances

# 모델 불러오기
def load_autoencoder(model_path, device):
    model = Autoencoder(z_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_deep_svdd(model_path, autoencoder, device):
    svdd_center_radius = torch.load(model_path, map_location=device)
    
    # Print the dictionary to understand its structure
    print("Loaded dictionary keys and types:")
    for key, value in svdd_center_radius.items():
        print(f"Key: {key}, Type: {type(value)}")

    # Convert 'center' to tensor and move to the specified device
    if 'center' not in svdd_center_radius:
        raise ValueError("Missing 'center' in svdd_center_radius.")
    
    center_list = svdd_center_radius['center']
    center_tensor = torch.tensor(center_list, dtype=torch.float32).to(device)
    
    # Check for 'radius' key; if not present, use a default value or handle accordingly
    if 'radius' in svdd_center_radius:
        radius = torch.tensor(svdd_center_radius['radius'], dtype=torch.float32).to(device)
    else:
        radius = torch.tensor([1.0], dtype=torch.float32).to(device)  # Example default value

    return DeepSVDD(autoencoder, center_tensor, radius).to(device)

# YOLOv10 모델로 객체 감지
def detect_objects(model, image_path, device):
    # 이미지 예측
    results = model.predict(image_path, save=True, imgsz=640, conf=0.5, device=device)

    # 예측 결과에서 박스와 신뢰도 추출
    boxes = []
    confidences = []

    # results는 numpy() 메서드를 사용하여 numpy 배열로 변환할 수 있습니다.
    for result in results:
        if result.boxes is not None:
            # result.boxes.numpy()를 사용하여 결과를 numpy 배열로 변환
            boxes_data = result.boxes.numpy()
            for box in boxes_data:
                # 각 박스의 정보 추출 (xyxy 형태)
                xmin, ymin, xmax, ymax, confidence = box[:5]
                boxes.append([xmin, ymin, xmax, ymax])
                confidences.append(confidence)

    return boxes, confidences

# 이미지 패치 추출
def extract_patches(image_path, boxes, size=(64, 64)):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    patches = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        patch = image_rgb[y1:y2, x1:x2]
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            patch_resized = cv2.resize(patch, size)
            patches.append(patch_resized)
    
    return patches

# 패치 전처리
def preprocess_patch(patch, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL Image
        transforms.Grayscale(),  # Ensure the image is in grayscale
        transforms.Resize((64, 64)),  # Resize to match the model's input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize
    ])
    return transform(patch).unsqueeze(0).to(device)

# 재구성 오류 계산
def calculate_reconstruction_error(original, reconstructed):
    return F.mse_loss(reconstructed, original, reduction='none').mean(dim=[1, 2, 3])

# 이상 탐지
def anomaly_detection_deep_svdd(model, patches, device):
    reconstruction_errors = []
    patches_tensor = torch.stack([preprocess_patch(patch, device) for patch in patches])

    with torch.no_grad():
        encoded_patches = model.autoencoder.encode(patches_tensor)
        distances = model(encoded_patches)
        reconstruction_errors = distances.cpu().numpy()

    return reconstruction_errors

def process_results(reconstruction_errors, threshold=0.5):
    anomalies = reconstruction_errors > threshold
    return anomalies

# Main process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#autoencoder_model_path = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/weights/pretrained_parameters.pth'  # Update this path
autoencoder_model_path = 'autoencoder_human_face.pth'  # Update this path
svdd_model_path = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/weights/final_deep_svdd.pth'  # Update this path
test_images_path = '/home/hkj/yolov10pj/yolov10_human/dataset/test2/*.jpg'  # Update this path
best_saved_model_path = '/home/hkj/yolov10pj/yolov10_human/runs/detect/train4/weights/best.pt'


# Load models
autoencoder_model = load_autoencoder(autoencoder_model_path, device)
svdd_model = load_deep_svdd(svdd_model_path, autoencoder_model, device)

# YOLOv10 모델 로드
yolo_model = YOLOv10(best_saved_model_path)

# Object detection
boxes, confidences = detect_objects(yolo_model, test_images_path, device)
patches = extract_patches(test_images_path, boxes)

# Anomaly detection
errors = anomaly_detection_deep_svdd(svdd_model, patches, device)
anomalies = process_results(errors)

# Print results
for idx, (error, is_anomaly) in enumerate(zip(errors, anomalies)):
    print(f'Patch {idx}: Reconstruction Error = {error}, Anomaly Detected = {is_anomaly}')

